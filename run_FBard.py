import os

# AutoDL/containers sometimes set OMP_NUM_THREADS to "" or 0, which libgomp rejects.
# Must run before importing torch/numpy so OpenMP reads a valid value.
_omp = os.environ.get("OMP_NUM_THREADS")
if _omp is not None:
    _omp = _omp.strip()
    try:
        if int(_omp) < 1:
            os.environ["OMP_NUM_THREADS"] = "1"
    except ValueError:
        os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import numpy as np
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
from torch.cuda.amp import autocast, GradScaler
import argparse
from sklearn.cluster import KMeans

# Dataset-Related Modules
from ad_dataset import ADDataset, get_data_transforms
from seg_dataset import SegTrainDataset, train_collate
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Model-Related Modules
from models import vit_encoder
from models.FBard import FBard
from models.FBard_logical import (
    FBard_logical,
    CompositionAutoEncoder,
    CompositionUNet,
)
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

# Training-Related Modules
from optimizers import StableAdamW
from utils import evaluation_batch, setup_seed, get_logger, WarmCosineScheduler, global_cosine_hm_adaptive, str2bool, to_device
from mask_classification_loss import MaskClassificationLoss
from SALAD.eval_logical import compute_normalization_params, evaluation_batch_with_composition
from SALAD.train_composition import train_composition_branch

warnings.filterwarnings("ignore")  # Suppress warning messages from the Python interpreter

print_fn = print
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main(args):
    use_logical_branch = getattr(args, 'use_logical_branch', False)

    # ===== Set random seed
    setup_seed(42)
    # ===== Data preparation
    # ---------- Set paths
    dataset_class_path = os.path.join(args.dataset_path, args.class_name)

    # ---------- Prepare segmentation dataset
    if args.phase == "train":
        seg_train_data = SegTrainDataset(root=dataset_class_path, img_size=(518, 518))
        seg_train_dataloader = torch.utils.data.DataLoader(seg_train_data, batch_size=args.seg_batch_size, shuffle=True, num_workers=4,
                                                       drop_last=True, collate_fn=train_collate)
        num_imgs = len(seg_train_data)
        print_fn(f'find {num_imgs} segmentation training images from path: {dataset_class_path}/train')

        # Compute mask annealing schedule parameters
        total_iters = args.seg_epochs * num_imgs / args.seg_batch_size
        attn_mask_annealing_start_steps = [int(round(total_iters * 2 / 12)), int(round(total_iters * 5 / 12)),
                                           int(round(total_iters * 8 / 12))]
        attn_mask_annealing_end_steps = [int(round(total_iters * 4 / 12)), int(round(total_iters * 7 / 12)),
                                         int(round(total_iters * 10 / 12))]

        print_fn(f"segmentation_training_total_iterations: {int(total_iters)}")
        print_fn(f"mask_annealing_start_steps: {attn_mask_annealing_start_steps}")
        print_fn(f"mask_annealing_end_steps: {attn_mask_annealing_end_steps}")

    # ---------- Prepare anomaly detection dataset
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    if args.phase == "train":
        train_path = os.path.join(dataset_class_path, 'train')
        ad_train_data = ImageFolder(root=train_path, transform=data_transform)
        ad_train_dataloader = torch.utils.data.DataLoader(ad_train_data, batch_size=args.ad_batch_size, shuffle=True,
                                                          num_workers=4,
                                                          drop_last=True)
        num_train_imgs = len(ad_train_data)
        print_fn(f'find {num_train_imgs} anomaly detection training images from path: {train_path}')

    elif args.phase == "test":
        test_data = ADDataset(root=dataset_class_path, transform=data_transform, gt_transform=gt_transform,
                              phase="test")
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.ad_batch_size, shuffle=False,
                                                      num_workers=4)
        num_test_imgs = len(test_data)
        print_fn(f'find {num_test_imgs} anomaly detection testing images from path: {dataset_class_path}/test')

    # ===== Model definition
    # ---------- Set encoder/decoder layers used for feature comparison
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # ---------- Load encoder (dinov2reg_vit_base_14)
    encoder = vit_encoder.load()
    embed_dim, num_heads = 768, 12

    # ---------- Initialize remaining components
    Bottleneck = []
    INP_Guided_Decoder = []
    INP_Extractor = []

    # ---------- Define bottleneck
    Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
    Bottleneck = nn.ModuleList(Bottleneck)

    # ---------- Define learnable INP (prototype) tokens (INP_num, embed_dim)=(6, 768)
    INP = nn.ParameterList(
        [nn.Parameter(torch.randn(args.INP_num, embed_dim))
         for _ in range(1)])

    # ---------- Define INP Extractor
    for i in range(1):
        blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Extractor.append(blk)
    INP_Extractor = nn.ModuleList(INP_Extractor)

    # ---------- Define decoder
    for i in range(8):
        blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Guided_Decoder.append(blk)
    INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)

    # ---------- Build model
    use_segmentaion_branch = not args.disable_segmentation_branch  # Disable when testing INP-Former
    update_bg_prototype = True if args.phase == "train" else False  # Background prototype updates enabled only during training
    use_bg_feature_aggregation = True

    if use_logical_branch:
        n_clusters = getattr(args, 'n_clusters', 4)

        # Load k-means cluster centers (test phase)
        kmeans_centers = None
        if args.phase == "test" and n_clusters > 0:
            dataset_name = args.dataset_path.split('/')[-1]
            kmeans_save_dir = os.path.join(args.save_dir, args.save_name, dataset_name, args.class_name)
            kmeans_path = os.path.join(kmeans_save_dir, 'kmeans_centers.npy')
            if os.path.exists(kmeans_path):
                kmeans_centers = np.load(kmeans_path)
                print_fn(f"Loaded k-means centers from {kmeans_path}")
            else:
                print_fn(f"Warning: k-means centers file not found at {kmeans_path}, clustering will be disabled")
                n_clusters = 0
                args.n_clusters = 0

        # Enable clustering collection only in AD training; disabled during seg training for speed
        enable_clustering_collection = False  # Off by default; enabled after segmentation training

        # Enable composition branch when clustering is on (train and test)
        use_composition_branch = (n_clusters > 0 and not args.disable_segmentation_branch)

        model = FBard_logical(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                            target_layers=target_layers, remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                            fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP, device=device, num_classes=2,
                            num_q=2, num_blocks=3, masked_attn_enabled=False,
                            update_bg_prototypes=update_bg_prototype,
                            use_segmentation_branch=use_segmentaion_branch,
                            use_bg_feature_aggregation=use_bg_feature_aggregation,
                            bg_feature_aggregation_binary=args.bg_feature_aggregation_binary,
                            fg_map_dilate_ksize=args.fg_map_dilate_ksize,
                            n_clusters=n_clusters,
                            kmeans_centers=kmeans_centers,
                            enable_clustering_collection=enable_clustering_collection,
                            use_composition_branch=use_composition_branch,
                            post_fusion=args.post_fusion)
        model = model.to(device)

        # Set loaded cluster centers on model during test
        if kmeans_centers is not None:
            model.set_kmeans_centers(kmeans_centers)
    else:
        model = FBard(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                            target_layers=target_layers, remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                            fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP, device=device, num_classes=2,
                            num_q=2, num_blocks=3, masked_attn_enabled=False,
                            update_bg_prototypes=update_bg_prototype,
                            use_segmentation_branch=use_segmentaion_branch,
                            use_bg_feature_aggregation=use_bg_feature_aggregation,
                            bg_feature_aggregation_binary=args.bg_feature_aggregation_binary,
                            fg_map_dilate_ksize=args.fg_map_dilate_ksize,
                            post_fusion=args.post_fusion)
        model = model.to(device)

    # ===== Training
    if args.phase == "train":
        skip_pretrain = False
        if use_logical_branch:
            # ---------- Check whether to skip pretraining
            dataset_name = args.dataset_path.split('/')[-1]
            save_dir = os.path.join(args.save_dir, args.save_name, dataset_name, args.class_name)
            weight_path = os.path.join(save_dir, args.weight_file_name)
            kmeans_path = os.path.join(save_dir, 'kmeans_centers.npy')

            skip_pretrain = getattr(args, 'skip_pretrain', False) and os.path.exists(weight_path)

            if skip_pretrain:
                print_fn("=" * 80)
                print_fn("Skipping pretraining (segmentation and anomaly detection branches)...")
                print_fn(f"Loading pretrained weights from {weight_path}")

                # Load model weights
                model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
                print_fn("Pretrained weights loaded successfully!")

                # Load cluster centers if available
                if args.n_clusters > 0 and os.path.exists(kmeans_path):
                    kmeans_centers = np.load(kmeans_path)
                    model.set_kmeans_centers(kmeans_centers)
                    print_fn(f"Loaded k-means centers from {kmeans_path}")
                elif args.n_clusters > 0:
                    print_fn(f"Warning: k-means centers file not found at {kmeans_path}")
                    print_fn("Will generate composition maps using existing model, but clustering may not be optimal")

                # Configure model state
                if hasattr(model, "set_segmentation_requires_grad"):
                    model.set_segmentation_requires_grad(False)  # Disable segmentation branch gradients

                print_fn("=" * 80)
            else:
                # ---------- Report clustering status
                if args.n_clusters > 0:
                    print_fn(f"k-means clustering enabled with {args.n_clusters} clusters")
                    print_fn(f"Cluster centers will be collected during anomaly detection training")
                else:
                    print_fn("k-means clustering disabled (n_clusters=0)")

        if not skip_pretrain:
            # ----------- Set up optimizers and schedulers
            seg_lr = 1e-5
            seg_base_lr, seg_final_lr = seg_lr, seg_lr / 10
            seg_optimizer = StableAdamW([{'params': model.parameters()}],  # Parameters to optimize
                                        lr=seg_lr, betas=(0.9, 0.999), weight_decay=0.05, amsgrad=True, eps=1e-10)
            seg_lr_scheduler = WarmCosineScheduler(seg_optimizer, base_value=seg_base_lr, final_value=seg_final_lr,
                                                   total_iters=args.seg_epochs * len(seg_train_dataloader),
                                                   warmup_iters=100)

            ad_trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
            ad_optimizer = StableAdamW([{'params': ad_trainable.parameters()}],  # Parameters to optimize
                                       lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
            ad_lr_scheduler = WarmCosineScheduler(ad_optimizer, base_value=1e-3, final_value=1e-4,
                                                  total_iters=args.ad_epochs * len(ad_train_dataloader),
                                                  warmup_iters=100)

            # ---------- Segmentation training
            if not args.disable_segmentation_branch:
                print("training segmentation branch...")
                global_step = 0
                mask_classification_loss = MaskClassificationLoss(num_labels=2).to(device)
                seg_scaler = GradScaler(enabled=not args.disable_amp)

                for epoch in range(args.seg_epochs):
                    model.train()
                    loss_list = []
                    for batch_idx, (imgs, targets) in enumerate(tqdm(seg_train_dataloader, ncols=80)):
                        imgs = imgs.to(device)
                        targets = to_device(targets, device)

                        if model.attn_mask_annealing_enabled:  # If mask annealing is enabled, compute mask probabilities
                            for i in range(model.num_blocks):
                                model.attn_mask_probs[i] = model.mask_annealing(
                                    attn_mask_annealing_start_steps[i],
                                    global_step,
                                    attn_mask_annealing_end_steps[i],
                                )

                        with autocast(enabled=not args.disable_amp):
                            output = model(imgs)  # Forward pass
                            if use_logical_branch and len(output) == 6:
                                # Cluster centers returned (training with clustering enabled)
                                _, _, _, mask_logits_per_block, class_logits_per_block, _ = output
                            else:
                                _, _, _, mask_logits_per_block, class_logits_per_block = output

                            losses_all_blocks = {}  # Compute loss
                            for i, (mask_logits, class_logits) in enumerate(list(zip(mask_logits_per_block, class_logits_per_block))):
                                losses = mask_classification_loss(
                                    masks_queries_logits=mask_logits,
                                    class_queries_logits=class_logits,
                                    targets=targets)
                                block_postfix = model.block_postfix(i)
                                losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
                                losses_all_blocks.update(losses)
                            loss = mask_classification_loss.loss_total(losses_all_blocks)

                        seg_optimizer.zero_grad(set_to_none=True)  # Zero gradients
                        seg_scaler.scale(loss).backward()  # Backpropagate gradients
                        seg_scaler.unscale_(seg_optimizer)
                        nn.utils.clip_grad_norm(model.parameters(), max_norm=0.1)  # Gradient clipping to cap the norm
                        seg_scaler.step(seg_optimizer)  # Update parameters
                        seg_scaler.update()

                        loss_val = loss.item()  # Record loss for the current batch
                        loss_list.append(loss_val)
                        seg_lr_scheduler.step()  # Update learning rate

                        # Log current batch loss (with epoch and batch index)
                        print_fn(
                            f'epoch [{epoch + 1}/{args.seg_epochs}], batch [{batch_idx + 1}/{len(seg_train_dataloader)}], loss: {loss_val:.4f}')

                        # # <<< Add VRAM monitoring >>>
                        # allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
                        # reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
                        # print_fn(
                        #     f'epoch [{epoch + 1}/{args.seg_epochs}], batch [{batch_idx + 1}/{len(seg_train_dataloader)}], loss: {loss_val:.4f}, Allocated Mem: {allocated:.2f} MB, Reserved Mem: {reserved:.2f} MB')

                        global_step += 1

                del seg_optimizer  # Drop segmentation optimizers (no longer needed)
                del seg_lr_scheduler
                torch.cuda.empty_cache()  # Clear computation graph and release unused GPU memory
                if hasattr(model, "set_segmentation_requires_grad"):
                    model.set_segmentation_requires_grad(False)  # Segmentation branch only needs gradients during seg training; disable for AD to save VRAM
                print("segmentation branch training completed")
                # After seg training, enable clustering collection for AD training
                if use_logical_branch and args.n_clusters > 0:
                    model.set_clustering_collection(True)
                    print_fn(f"Clustering collection enabled for anomaly detection training (n_clusters={args.n_clusters})")

            print("training anomaly detection branch...")
            # ---------- Anomaly detection training
            ad_scaler = GradScaler(enabled=not args.disable_amp)

            # ---------- Collect k-means cluster centers (if enabled)
            all_cluster_centers = [] if (use_logical_branch and args.n_clusters > 0) else None
            if all_cluster_centers is not None:
                print_fn(f"Collecting cluster centers for k-means (n_clusters={args.n_clusters})...")

            for epoch in range(args.ad_epochs):
                model.train()
                for batch_idx, (img, _) in enumerate(tqdm(ad_train_dataloader, ncols=80)):
                    img = img.to(device)

                    with autocast(enabled=not args.disable_amp):
                        if not args.disable_segmentation_branch:  # Forward pass
                            if use_logical_branch:
                                output = model(img)
                                if len(output) == 6 and args.n_clusters > 0:
                                    # Output includes cluster centers
                                    en, de, g_loss, _, _, cluster_centers_batch = output
                                    if all_cluster_centers is not None:
                                        all_cluster_centers.extend(cluster_centers_batch)
                                else:
                                    en, de, g_loss = output[0], output[1], output[2]
                            else:
                                en, de, g_loss, _, _ = model(img)
                        else:
                            en, de, g_loss = model(img)
                        loss = global_cosine_hm_adaptive(en, de, y=3)  # Reconstruction loss
                        loss = loss + 0.2 * g_loss  # Contrastive loss + λ * INP consistency loss

                    ad_optimizer.zero_grad(set_to_none=True)  # Zero gradients
                    ad_scaler.scale(loss).backward()  # Backpropagate gradients

                    ad_scaler.unscale_(ad_optimizer)
                    nn.utils.clip_grad_norm(ad_trainable.parameters(), max_norm=0.1)  # Gradient clipping to cap the norm

                    ad_scaler.step(ad_optimizer)  # Update parameters
                    ad_scaler.update()

                    ad_lr_scheduler.step()  # Update learning rate

                    # Log current batch loss (with epoch and batch index)
                    print_fn(
                        f'epoch [{epoch + 1}/{args.ad_epochs}], batch [{batch_idx + 1}/{len(ad_train_dataloader)}], loss: {loss.item():.4f}')

            if use_logical_branch:
                # ---------- Second-level k-means on all cluster centers
                if all_cluster_centers is not None and len(all_cluster_centers) > 0:
                    print_fn(f"Performing second-level k-means clustering on {len(all_cluster_centers)} cluster centers...")
                    # Flatten per-image cluster centers to (num_images * n_clusters, C)
                    all_means_array = np.array(all_cluster_centers).reshape(-1, all_cluster_centers[0].shape[1])  # (num_images * n_clusters, C)

                    # Run second-level k-means
                    final_kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
                    final_kmeans.fit(all_means_array)
                    final_centers = final_kmeans.cluster_centers_  # (n_clusters, C)

                    # Save final cluster centers
                    dataset_name = args.dataset_path.split('/')[-1]
                    save_dir = os.path.join(args.save_dir, args.save_name, dataset_name, args.class_name)
                    os.makedirs(save_dir, exist_ok=True)
                    kmeans_path = os.path.join(save_dir, 'kmeans_centers.npy')
                    np.save(kmeans_path, final_centers)
                    print_fn(f"Saved final k-means centers ({args.n_clusters} clusters, feature_dim={final_centers.shape[1]}) to {kmeans_path}")

                    # Set cluster centers on model
                    model.set_kmeans_centers(final_centers)
                else:
                    # If no centers collected but clustering enabled, try loading existing centers
                    if args.n_clusters > 0:
                        dataset_name = args.dataset_path.split('/')[-1]
                        kmeans_save_dir = os.path.join(args.save_dir, args.save_name, dataset_name, args.class_name)
                        kmeans_path = os.path.join(kmeans_save_dir, 'kmeans_centers.npy')
                        if os.path.exists(kmeans_path):
                            kmeans_centers = np.load(kmeans_path)
                            model.set_kmeans_centers(kmeans_centers)
                            print_fn(f"Loaded existing k-means centers from {kmeans_path}")

            dataset_name = args.dataset_path.split('/')[-1]
            save_dir = os.path.join(args.save_dir, args.save_name, dataset_name, args.class_name)  # Save model
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, args.weight_file_name))

        # ===== Composition Branch Training =====
        if use_logical_branch and args.n_clusters > 0 and not args.disable_segmentation_branch:
            train_composition_branch(
                model=model,
                dataloader=ad_train_dataloader,
                args=args,
                device=device,
                logger=globals().get("print_fn", print),
                save_dir=save_dir,
            )


    # ===== Testing
    elif args.phase == "test":
        # ---------- Load model weights
        if args.weight_path is not None:
            weight_path = args.weight_path
        else:
            dataset_name = args.dataset_path.split('/')[-1]
            weight_path = f"saved_results/{args.save_name}/{dataset_name}/{args.class_name}/{args.weight_file_name}"
        print(f"Loading weights from {weight_path}")

        if use_logical_branch:
            model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)  # strict=False for composition branch compatibility

            # Load k-means cluster centers if present
            dataset_name = args.dataset_path.split('/')[-1]
            save_dir = os.path.join(args.save_dir, args.save_name, dataset_name, args.class_name)
            kmeans_path = os.path.join(save_dir, 'kmeans_centers.npy')
            if os.path.exists(kmeans_path) and args.n_clusters > 0:
                kmeans_centers = np.load(kmeans_path)
                model.set_kmeans_centers(kmeans_centers)
                print_fn(f"Loaded k-means centers from {kmeans_path}")

            # ---------- Ensure composition branch is initialized (if n_clusters > 0)
            if args.n_clusters > 0 and (not hasattr(model, 'comp_ae') or model.comp_ae is None):
                print_fn("Initializing composition branch for testing...")
                model.comp_ae = CompositionAutoEncoder(n_clusters=args.n_clusters).to(device)
                model.comp_unet = CompositionUNet(n_clusters=args.n_clusters).to(device)
                model.use_composition_branch = True

            # ---------- Compute normalization params on validation set
            normalization_params = None
            if args.n_clusters > 0 and hasattr(model, 'comp_ae') and hasattr(model, 'comp_unet') and model.comp_ae is not None:
                print_fn("=" * 80)
                print_fn("Computing normalization parameters on validation set (10% of training data)...")
                # Build validation set (10% of training data)
                data_transform, _ = get_data_transforms(args.input_size, args.crop_size)
                train_path = os.path.join(dataset_class_path, 'train')
                train_data = ImageFolder(root=train_path, transform=data_transform)

                # Random 10% split for validation
                val_size = int(len(train_data) * 0.1)
                train_size = len(train_data) - val_size
                train_subset, val_subset = torch.utils.data.random_split(
                    train_data, [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)  # Fixed random seed
                )
                val_dataloader = torch.utils.data.DataLoader(
                    val_subset, batch_size=args.ad_batch_size, shuffle=False, num_workers=4
                )

                # Compute normalization parameters
                # Create visualize_analysis directory for histograms
                dataset_name = args.dataset_path.split('/')[-1]
                hist_save_dir = os.path.join('visualize_analysis', args.save_name, dataset_name, args.class_name)
                ad_mean, ad_std, comp_mean, comp_std = compute_normalization_params(
                    val_dataloader, model, device, args.n_clusters, max_ratio=0.01, save_dir=hist_save_dir
                )
                normalization_params = (ad_mean, ad_std, comp_mean, comp_std)
                print_fn("=" * 80)

            # ---------- Test (using fused anomaly maps)
            model.eval()
            vis_path = None if args.disable_visualize else os.path.join(args.visualize_save_dir, args.save_name)

            # Call evaluation_batch_with_composition (composition branch support)
            results = evaluation_batch_with_composition(
                model=model,
                dataloader=test_dataloader,
                device=device,
                max_ratio=0.01,
                resize_mask=512,
                vis_path=vis_path,
                vis_img_num=200,
                compute_metrics=not getattr(args, 'disable_compute_metrics', False),
                visualize_score_distribution=False,
                compute_per_class_auroc=False,
                visualize_pixel_roc=False,
                save_name=args.save_name,
                n_clusters=args.n_clusters if args.n_clusters > 0 else None,
                normalization_params=normalization_params,
                skip_visualization=args.disable_visualize,
                apply_fg_prob_map=args.post_fusion,
            )
        else:
            model.load_state_dict(torch.load(weight_path), strict=True)

            # ---------- Run evaluation
            model.eval()
            vis_path = None if args.disable_visualize else os.path.join(args.visualize_save_dir, args.save_name)
            results = evaluation_batch(model=model,
                                        dataloader=test_dataloader,
                                        device=device,
                                        max_ratio=0.01,
                                        resize_mask=512,
                                        vis_path=vis_path,
                                        vis_img_num=200,  # Number of visualizations per class
                                        compute_metrics=False,  # Whether to compute evaluation metrics
                                        visualize_score_distribution=False,  # Whether to plot anomaly score distribution histogram
                                        compute_per_class_auroc=False,  # Whether to compute per-class pixel-level AUROC
                                        visualize_pixel_roc=False,  # Whether to plot pixel-level ROC curve
                                        save_name=args.save_name,  # Version name (used in ROC curve save path)
                                        apply_fg_prob_map=args.post_fusion)

        if results is not None:
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
            print_fn(  # Print results
                'I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
        else:
            print_fn("Metric computation skipped")

    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Force PyTorch CUDA ops to run synchronously so the CPU waits for the GPU to finish
    parser = argparse.ArgumentParser(description='')  # Create command-line argument parser

    # dataset info
    parser.add_argument('--dataset_path', type=str, default=r'./datasets/MIAD')
    parser.add_argument('--class_name', type=str, default=r'metal_welding')

    # save info
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='MIAD')
    parser.add_argument('--visualize_save_dir', type=str, default='visualize_results')
    parser.add_argument('--weight_file_name', type=str, default='model.pth')

    # model info
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--crop_size', type=int, default=518)
    parser.add_argument('--INP_num', type=int, default=6)
    parser.add_argument('--fg_map_dilate_ksize', type=int, default=5)
    parser.add_argument('--bg_feature_aggregation_binary', type=str2bool, default=False,
                        help='If True: binarize the foreground probability map during background aggregation (foreground keeps encoded features, background is replaced by background prototypes); otherwise use probability-weighted aggregation')
    parser.add_argument('--disable_segmentation_branch', action='store_true', default=False)  # Whether to disable the segmentation branch
    parser.add_argument('--post_fusion', type=str2bool,
                        default=False)  # Whether to skip bg_prob_map feature processing and gate the anomaly map with the foreground probability map at test time
    parser.add_argument('--use_logical_branch', type=str2bool, default=False,
                        help='Enable logical anomaly detection (FBard_logical + composition branch)')
    parser.add_argument('--n_clusters', type=int, default=4,
                        help='Number of k-means clusters (logical branch only)')

    # training info
    parser.add_argument('--ad_epochs', type=int, default=2)
    parser.add_argument('--seg_epochs', type=int, default=1)
    parser.add_argument('--ad_batch_size', type=int, default=16)
    parser.add_argument('--seg_batch_size', type=int, default=8)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--disable_amp', action='store_true', default=False)  # Whether to disable mixed-precision training
    parser.add_argument('--skip_pretrain', action='store_true', default=False,
                        help='Skip seg + AD pretraining and jump to composition training (logical branch only)')

    # testing_info
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--disable_visualize', action='store_true', default=False)  # Whether to skip visualization
    parser.add_argument('--disable_compute_metrics', action='store_true', default=False,
                        help='Skip metric computation during logical-branch testing')

    args = parser.parse_args()

    dataset_name = args.dataset_path.split('/')[-1]
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name, dataset_name, args.class_name))
    print_fn = logger.info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    main(args)
