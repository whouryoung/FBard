import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss
from tqdm import tqdm

from models.FBard_logical import CompositionAutoEncoder, CompositionUNet
from SALAD.composition_anomaly import (
    change_label_same_img_feat,
    cluster_to_onehot,
    combine_strategy_2_and_3,
    copy_label_from_other_image,
    find_background_label,
    onehot_to_cluster,
)
from SALAD.composition_loss import DiceLoss, MultiClassFocalLoss
from utils import visualize_clustering_result


def train_composition_branch(model, dataloader, args, device, logger, save_dir):
    """
    Export composition maps, vote a global background label, then train AE + UNet.
    """
    if logger is None:
        print_fn = print
    elif callable(logger) and not hasattr(logger, "info"):
        print_fn = logger
    else:
        print_fn = logger.info
    if args.n_clusters <= 0 or args.disable_segmentation_branch:
        return

    print_fn("=" * 80)
    print_fn("Starting Composition Branch Training...")

    dataset_name = args.dataset_path.split("/")[-1]
    comp_vis_base_dir = os.path.join(
        "visualize_composition_branch", args.save_name, dataset_name, args.class_name
    )
    os.makedirs(comp_vis_base_dir, exist_ok=True)

    print_fn("Step 1: Generating and saving composition maps...")
    model.eval()
    composition_map_dir = os.path.join(comp_vis_base_dir, "composition_maps")
    os.makedirs(composition_map_dir, exist_ok=True)

    composition_maps = []
    img_count = 0
    with torch.no_grad():
        for _, (img, __) in enumerate(tqdm(dataloader, ncols=80, desc="Generating composition maps")):
            img = img.to(device)
            output = model(img)
            if len(output) == 6:
                cluster_assignments = output[5]
            else:
                print_fn("Warning: No cluster assignments found, skipping composition branch training")
                break

            for cluster_assignment in cluster_assignments:
                onehot = cluster_to_onehot(cluster_assignment, args.n_clusters)
                np_path = os.path.join(composition_map_dir, f"composition_map_{img_count:05d}.npy")
                np.save(np_path, onehot.numpy())
                composition_maps.append(np_path)
                if img_count % 100 == 0:
                    visualize_clustering_result(
                        cluster_assignment.cpu().numpy(),
                        f"composition_map_{img_count:05d}",
                        composition_map_dir,
                        (518, 518),
                        n_clusters=args.n_clusters,
                    )
                img_count += 1

    print_fn(f"Saved {len(composition_maps)} composition maps to {composition_map_dir}")

    global_background_label = None
    if len(composition_maps) > 0:
        print_fn("Determining global background label...")
        sample_size = min(max(10, len(composition_maps) // 10), 100, len(composition_maps))
        sampled_indices = random.sample(range(len(composition_maps)), sample_size)
        background_label_votes = {}
        for idx in sampled_indices:
            seg_sample = np.load(composition_maps[idx])
            seg_cluster = onehot_to_cluster(seg_sample)
            bg_label = find_background_label(seg_cluster)
            background_label_votes[bg_label] = background_label_votes.get(bg_label, 0) + 1
        if background_label_votes:
            global_background_label = max(background_label_votes.items(), key=lambda x: x[1])[0]
            print_fn(f"Global background label determined: {global_background_label} (votes: {background_label_votes})")
        else:
            global_background_label = 0
            print_fn(f"Warning: No background label votes, using default: {global_background_label}")

    print_fn("Step 2: Training composition branch...")
    if model.comp_ae is None or model.comp_unet is None:
        model.comp_ae = CompositionAutoEncoder(n_clusters=args.n_clusters).to(device)
        model.comp_unet = CompositionUNet(n_clusters=args.n_clusters).to(device)
        model.use_composition_branch = True

    weights = [1.0] * args.n_clusters
    dice_loss_f = DiceLoss(weights).to(device)
    multiclass_focal_loss = MultiClassFocalLoss(gamma=2, reduction="mean").to(device)
    focal_loss = sigmoid_focal_loss
    comp_optimizer = torch.optim.Adam(
        [{"params": list(model.comp_ae.parameters()) + list(model.comp_unet.parameters()), "lr": 1e-5}],
        lr=1e-5,
        weight_decay=1e-5,
    )

    comp_vis_dir = os.path.join(comp_vis_base_dir, "composition_branch_vis")
    os.makedirs(comp_vis_dir, exist_ok=True)

    num_epochs = 3
    iteration = 0
    for epoch in range(num_epochs):
        model.train()
        random.shuffle(composition_maps)
        for map_idx in tqdm(range(len(composition_maps)), ncols=80, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            seg = torch.from_numpy(np.load(composition_maps[map_idx])).float().to(device)
            seg = seg.unsqueeze(0)
            random_idx = random.randint(0, len(composition_maps) - 1)
            diff_seg = torch.from_numpy(np.load(composition_maps[random_idx])).float().to(device)
            diff_seg = diff_seg.unsqueeze(0)

            strategy = random.choice([2, 3, 4])
            seg_normal = seg.clone()
            anom_seg = seg.clone()
            if strategy == 2:
                mask, recon_mask, anom_seg = change_label_same_img_feat(
                    anom_seg.squeeze(0), diff_seg.squeeze(0), global_background_label
                )
            elif strategy == 3:
                mask, recon_mask, anom_seg = copy_label_from_other_image(
                    anom_seg.squeeze(0), diff_seg.squeeze(0), global_background_label
                )
            else:
                mask, recon_mask, anom_seg = combine_strategy_2_and_3(
                    anom_seg.squeeze(0), diff_seg.squeeze(0), global_background_label
                )

            mask = mask.unsqueeze(0).to(device)
            anom_seg = anom_seg.unsqueeze(0).to(device)
            seg_normal_vis = seg_normal.squeeze(0).cpu().numpy()
            anom_seg_vis = anom_seg.squeeze(0).cpu().numpy()
            mask_vis = mask.squeeze(0).cpu().numpy()

            input_batch = torch.cat([anom_seg, seg_normal], dim=0)
            mask_batch = torch.cat([mask, torch.zeros_like(mask)], dim=0)
            target_batch = torch.cat([seg_normal, seg_normal], dim=0)
            target_argmax = target_batch.argmax(dim=1)

            seg_recon = model.comp_ae(input_batch).softmax(dim=1)
            unet_input = torch.cat([input_batch, seg_recon], dim=1)
            pred_mask = model.comp_unet(unet_input).squeeze(1)

            loss_comp_recon = multiclass_focal_loss(seg_recon, target_argmax) + dice_loss_f(seg_recon, target_argmax)
            loss_comp_mask = 5 * focal_loss(pred_mask, mask_batch, reduction="mean") + F.l1_loss(
                torch.sigmoid(pred_mask), mask_batch
            )
            loss_total = loss_comp_recon + loss_comp_mask

            comp_optimizer.zero_grad()
            loss_total.backward()
            comp_optimizer.step()
            iteration += 1

            if iteration % 200 == 0:
                model.eval()
                with torch.no_grad():
                    visualize_clustering_result(
                        onehot_to_cluster(seg_normal_vis),
                        f"iter_{iteration}_original",
                        comp_vis_dir,
                        (518, 518),
                        n_clusters=args.n_clusters,
                    )
                    visualize_clustering_result(
                        onehot_to_cluster(anom_seg_vis),
                        f"iter_{iteration}_anomaly",
                        comp_vis_dir,
                        (518, 518),
                        n_clusters=args.n_clusters,
                    )
                    mask_resized = cv2.resize(
                        mask_vis.astype(np.float32), (518, 518), interpolation=cv2.INTER_NEAREST
                    )
                    plt.imsave(
                        os.path.join(comp_vis_dir, f"iter_{iteration}_mask.png"),
                        mask_resized,
                        cmap="gray",
                        vmin=0,
                        vmax=1,
                    )
                    plt.close()
                    visualize_clustering_result(
                        onehot_to_cluster(seg_recon[0].cpu().numpy()),
                        f"iter_{iteration}_recon",
                        comp_vis_dir,
                        (518, 518),
                        n_clusters=args.n_clusters,
                    )
                    pred_mask_resized = cv2.resize(
                        pred_mask[0].cpu().numpy().astype(np.float32),
                        (518, 518),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    plt.imsave(
                        os.path.join(comp_vis_dir, f"iter_{iteration}_disc.png"),
                        pred_mask_resized,
                        cmap="viridis",
                        vmin=0,
                        vmax=1,
                    )
                    plt.close()
                model.train()

            if iteration % 100 == 0:
                print_fn(
                    f"[Composition Branch] Epoch [{epoch + 1}/{num_epochs}], Iteration [{iteration}], "
                    f"Recon Loss: {loss_comp_recon.item():.4f}, Disc Loss: {loss_comp_mask.item():.4f}"
                )

    print_fn("Composition branch training completed!")
    print_fn(f"Visualization results saved to {comp_vis_dir}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, args.weight_file_name))
    print_fn("Model with composition branch saved.")
