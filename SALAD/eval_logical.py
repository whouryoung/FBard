import gc
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from SALAD.composition_infer import run_composition_branch
from SALAD.vis_logical import (
    save_norm_histograms,
    save_test_histograms,
    visualize_composition_eval,
)
from utils import (
    ader_evaluator,
    cal_anomaly_maps,
    evaluation_batch,
    get_gaussian_kernel,
    logit_to_fg_prob,
    to_per_pixel_logits_semantic,
)


def compute_image_level_score(anomaly_map, max_ratio=0.01):
    """
    Compute image-level anomaly score: resize -> Gaussian blur -> mean of top 1% pixel scores.
    anomaly_map: (B, 1, H, W) or (B, H, W)
    """
    if len(anomaly_map.shape) == 4:
        anomaly_map = anomaly_map.squeeze(1)  # (B, H, W)
    
    # Resize to 512x512
    anomaly_map_resized = F.interpolate(
        anomaly_map.unsqueeze(1), 
        size=(512, 512), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(1)  # (B, 512, 512)
    
    # Gaussian blur
    from utils import get_gaussian_kernel
    gaussian_kernel = get_gaussian_kernel(kernel_size=9, sigma=7).to(anomaly_map.device)
    anomaly_map_blurred = gaussian_kernel(anomaly_map_resized.unsqueeze(1)).squeeze(1)  # (B, 512, 512)
    
    # Mean of top 1% pixel scores after sorting
    anomaly_map_flat = anomaly_map_blurred.flatten(1)  # (B, 512*512)
    top_k = int(anomaly_map_flat.shape[1] * max_ratio)
    sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][:, :top_k].mean(dim=1)  # (B,)
    
    return sp_score


def compute_normalization_params(val_dataloader, model, device, n_clusters, max_ratio=0.01, save_dir=None):
    """
    Compute normalization parameters (mean and std) for both branches on the validation set.
    Mean and std are computed from all pixel-level anomaly scores for z-score normalization.
    Returns: (ad_mean, ad_std, comp_mean, comp_std)
    save_dir: directory to save histograms; if None, histograms are not saved
    """
    model.eval()
    ad_pixel_scores = []  # AD branch pixel-level anomaly scores (unnormalized)
    comp_pixel_scores = []  # Composition branch pixel-level anomaly scores (unnormalized)
    ad_pixel_scores_norm = []  # AD branch pixel-level anomaly scores (normalized)
    comp_pixel_scores_norm = []  # Composition branch pixel-level anomaly scores (normalized)
    
    print("Computing normalization parameters on validation set...")
    from utils import cal_anomaly_maps, get_gaussian_kernel
    gaussian_kernel = get_gaussian_kernel(kernel_size=9, sigma=7).to(device)
    
    with torch.no_grad():
        for data in tqdm(val_dataloader, ncols=80, desc="Validation"):
            # ImageFolder returns (image, label), not (img, gt, label, img_path)
            if len(data) == 2:
                img, label = data
                img = img.to(device)
            else:
                img, gt, label, img_path = data
                img = img.to(device)
            B = img.shape[0]
            
            # ===== Original anomaly detection branch =====
            output = model(img)
            if len(output) == 6:
                en, de = output[0], output[1]
            elif len(output) == 5:
                en, de = output[0], output[1]
            else:
                en, de = output[0], output[1]
            
            anomaly_map_ad, _ = cal_anomaly_maps(en, de, img.shape[-1])
            # Resize to 512x512 and apply Gaussian blur
            anomaly_map_ad = F.interpolate(anomaly_map_ad, size=(512, 512), mode='bilinear', align_corners=False)
            anomaly_map_ad = gaussian_kernel(anomaly_map_ad)
            # Collect pixel-level anomaly scores (unnormalized)
            ad_pixel_scores.append(anomaly_map_ad.cpu().flatten())
            
            # ===== Composition Branch =====
            comp_result = run_composition_branch(
                model, img, n_clusters, device, gaussian_kernel, resize_to=512, return_aux=False,
            )
            if comp_result is not None:
                comp_pixel_scores.append(comp_result.anomaly_maps.cpu().flatten())
    
    # Compute mean and std of all pixel-level anomaly scores
    if len(ad_pixel_scores) > 0:
        ad_pixel_all = torch.cat(ad_pixel_scores).numpy()
        ad_mean = float(np.mean(ad_pixel_all))
        ad_std = float(np.std(ad_pixel_all))
    else:
        ad_mean = 0.0
        ad_std = 1.0
    
    if len(comp_pixel_scores) > 0:
        comp_pixel_all = torch.cat(comp_pixel_scores).numpy()
        comp_mean = float(np.mean(comp_pixel_all))
        comp_std = float(np.std(comp_pixel_all))
    else:
        comp_mean = 0.0
        comp_std = 1.0
    
    # Avoid division by zero
    if ad_std < 1e-6:
        ad_std = 1.0
    if comp_std < 1e-6:
        comp_std = 1.0
    
    print(f"Normalization parameters computed (based on pixel-level scores):")
    print(f"  AD branch: mean={ad_mean:.4f}, std={ad_std:.4f}")
    print(f"  Composition branch: mean={comp_mean:.4f}, std={comp_std:.4f}")

    if save_dir is not None and len(ad_pixel_scores) > 0:
        save_norm_histograms(
            save_dir, ad_pixel_scores, comp_pixel_scores, ad_mean, ad_std, comp_mean, comp_std,
        )

    return ad_mean, ad_std, comp_mean, comp_std


def evaluation_batch_with_composition(model, dataloader, device, max_ratio=0, resize_mask=None, vis_path=None, vis_img_num=200,
                                     GPU_accelerate=True, compute_metrics=True, visualize_score_distribution=True, 
                                     compute_per_class_auroc=False, visualize_pixel_roc=False, save_name=None, 
                                     n_clusters=None, normalization_params=None, skip_visualization=False,
                                     apply_fg_prob_map: bool = False):
    """
    Evaluation function with composition branch inference.
    normalization_params: (ad_mean, ad_std, comp_mean, comp_std) or None
    """
    from utils import evaluation_batch, cal_anomaly_maps, get_gaussian_kernel
    
    # Fall back to original evaluation if no composition branch
    if n_clusters is None or n_clusters <= 0:
        return evaluation_batch(
            model=model, dataloader=dataloader, device=device, max_ratio=max_ratio,
            resize_mask=resize_mask, vis_path=vis_path, vis_img_num=vis_img_num,
            GPU_accelerate=GPU_accelerate, compute_metrics=compute_metrics,
            visualize_score_distribution=visualize_score_distribution,
            compute_per_class_auroc=compute_per_class_auroc,
            visualize_pixel_roc=visualize_pixel_roc, save_name=save_name,
            n_clusters=n_clusters,
            apply_fg_prob_map=apply_fg_prob_map,
        )
    
    # Use normalization if params provided; otherwise fuse by direct sum
    use_normalization = (normalization_params is not None)
    if use_normalization:
        ad_mean, ad_std, comp_mean, comp_std = normalization_params
    else:
        print("Warning: Normalization parameters not available, fusing without normalization")
        ad_mean, ad_std, comp_mean, comp_std = 0.0, 1.0, 0.0, 1.0  # No normalization (equivalent to dividing by 1)
    
    # ===== Initialization
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    img_list = []
    seg_map_list = []
    anomaly_map_list = []
    comp_anomaly_map_list = []  # Composition branch anomaly maps
    fused_anomaly_map_list = []  # Fused anomaly maps
    disc_output_list = []  # Discriminator outputs (for visualization)
    cluster_map_list = []  # Cluster maps (for visualization)
    recon_map_list = []  # Reconstruction outputs (for visualization)
    gt_list = []
    img_path_list = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=9, sigma=7).to(device)
    
    # Collect test pixel scores for histograms — only when visualization is enabled
    test_ad_pixel_scores = []  # Test AD branch pixel scores (before normalization)
    test_comp_pixel_scores = []  # Test Composition branch pixel scores (before normalization)
    test_ad_pixel_scores_norm = []  # Test AD branch pixel scores (after normalization)
    test_comp_pixel_scores_norm = []  # Test Composition branch pixel scores (after normalization)
    
    # Collect all test pixel scores for global min/max visualization — only when enabled
    all_ad_scores = []  # AD branch heatmap pixel scores for all test samples
    all_comp_scores = []  # Composition branch heatmap pixel scores for all test samples
    all_disc_scores = []  # Discriminator pixel scores for all test samples (37×37)
    all_fused_scores = []  # Fused heatmap pixel scores for all test samples
    
    # ===== Inference
    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            has_seg_output = False
            img = img.to(device)
            B = img.shape[0]
            
            # ===== Original anomaly detection branch =====
            output = model(img)
            if len(output) == 6:
                en, de, mask_logits, class_logits = output[0], output[1], output[3][-1], output[4][-1]
                has_seg_output = True
            elif len(output) == 5:
                en, de, mask_logits, class_logits = output[0], output[1], output[3][-1], output[4][-1]
                has_seg_output = True
            else:
                en, de = output[0], output[1]
            
            anomaly_map_ad, _ = cal_anomaly_maps(en, de, img.shape[-1])
            if resize_mask is not None:
                anomaly_map_ad = F.interpolate(anomaly_map_ad, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            anomaly_map_ad = gaussian_kernel(anomaly_map_ad)
            if has_seg_output and apply_fg_prob_map:
                pixel_logits = to_per_pixel_logits_semantic(mask_logits, class_logits)
                seg_map = logit_to_fg_prob(pixel_logits)
                seg_map = seg_map.unsqueeze(1)
                seg_map = F.interpolate(
                    seg_map,
                    size=anomaly_map_ad.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_map_ad = anomaly_map_ad * seg_map
            
            # ===== Composition Branch =====
            comp_anomaly_map = None
            disc_output_batch = None
            cluster_map_batch = None
            recon_map_batch = None
            comp_resize = (resize_mask, resize_mask) if resize_mask is not None else img.shape[-1]
            comp_result = run_composition_branch(
                model, img, n_clusters, device, gaussian_kernel,
                resize_to=comp_resize, return_aux=True,
            )
            if comp_result is not None:
                comp_anomaly_map = comp_result.anomaly_maps.to(device)
                disc_output_batch = comp_result.disc_outputs
                cluster_map_batch = comp_result.cluster_maps
                recon_map_batch = comp_result.recon_maps
            
            # ===== Normalization and fusion =====
            # Collect pre-normalization pixel scores for histograms — only when visualization enabled
            if not skip_visualization:
                test_ad_pixel_scores.append(anomaly_map_ad.cpu().flatten())
                if comp_anomaly_map is not None:
                    test_comp_pixel_scores.append(comp_anomaly_map.cpu().flatten())
            
            # Z-score normalize all pixels in both branch anomaly maps (if params provided)
            if use_normalization:
                # Z-score normalize using stored parameters
                anomaly_map_ad_flat = anomaly_map_ad.flatten(1)  # (B, H*W)
                anomaly_map_ad_norm_flat = (anomaly_map_ad_flat - ad_mean) / ad_std
                anomaly_map_ad_norm = anomaly_map_ad_norm_flat.view(anomaly_map_ad.shape)  # (B, 1, H, W)
                # Collect post-normalization pixel scores for histograms — only when visualization enabled
                if not skip_visualization:
                    test_ad_pixel_scores_norm.append(anomaly_map_ad_norm.cpu().flatten())
                
                if comp_anomaly_map is not None:
                    comp_anomaly_map_flat = comp_anomaly_map.flatten(1)  # (B, H*W)
                    comp_anomaly_map_norm_flat = (comp_anomaly_map_flat - comp_mean) / comp_std
                    comp_anomaly_map_norm = comp_anomaly_map_norm_flat.view(comp_anomaly_map.shape)  # (B, 1, H, W)
                    # Collect post-normalization pixel scores for histograms — only when visualization enabled
                    if not skip_visualization:
                        test_comp_pixel_scores_norm.append(comp_anomaly_map_norm.cpu().flatten())
                    # Fuse: sum normalized anomaly maps from both branches
                    fused_anomaly_map = anomaly_map_ad_norm + comp_anomaly_map_norm  # (B, 1, H, W)
                    # Release intermediate variables
                    del comp_anomaly_map_flat, comp_anomaly_map_norm_flat, comp_anomaly_map_norm
                else:
                    fused_anomaly_map = anomaly_map_ad_norm
                    comp_anomaly_map = torch.zeros_like(anomaly_map_ad)
                # Release intermediate variables
                del anomaly_map_ad_flat, anomaly_map_ad_norm_flat, anomaly_map_ad_norm
            else:
                # No normalization; fuse by direct sum
                if comp_anomaly_map is not None:
                    fused_anomaly_map = anomaly_map_ad + comp_anomaly_map  # (B, 1, H, W)
                else:
                    fused_anomaly_map = anomaly_map_ad
                    comp_anomaly_map = torch.zeros_like(anomaly_map_ad)
            
            # Collect pixel scores for global min/max visualization
            # Note: store flattened data only; release after percentile computation to save memory
            # Collect only when visualization is enabled
            if vis_path is not None and not skip_visualization:
                all_ad_scores.append(anomaly_map_ad.detach().cpu().flatten())
                if comp_anomaly_map is not None:
                    all_comp_scores.append(comp_anomaly_map.detach().cpu().flatten())
                if disc_output_batch is not None:
                    # Discriminator output is 37×37; resize to match anomaly heatmap size
                    disc_resized_for_vis = F.interpolate(
                        disc_output_batch.unsqueeze(1).detach().cpu(),
                        size=anomaly_map_ad.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                    all_disc_scores.append(disc_resized_for_vis.flatten())
                    del disc_resized_for_vis  # Release immediately
                # Collect fused heatmap pixel scores
                all_fused_scores.append(fused_anomaly_map.detach().cpu().flatten())
            
            # ===== Organize predictions and ground truth =====
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            
            # Accumulate data (move to CPU immediately to save GPU memory)
            if compute_metrics:
                gt_list_px.append(gt.detach().cpu())
                pr_list_px.append(fused_anomaly_map.detach().cpu())  # Use fused anomaly map
                gt_list_sp.append(label)
            
            # Store visualization metadata — only when visualization is enabled
            if vis_path is not None and not skip_visualization:
                # Use detach() and cpu() to free GPU memory
                anomaly_map_ad_cpu = anomaly_map_ad.detach().cpu()
                comp_anomaly_map_cpu = comp_anomaly_map.detach().cpu()
                fused_anomaly_map_cpu = fused_anomaly_map.detach().cpu()
                gt_cpu = gt.detach().cpu()
                img_cpu = img.detach().cpu()
                
                img_list.append(img_cpu)
                if has_seg_output:
                    pixel_logits = to_per_pixel_logits_semantic(mask_logits, class_logits)
                    seg_map = logit_to_fg_prob(pixel_logits)
                    seg_map = seg_map.unsqueeze(1)
                    seg_map = F.interpolate(seg_map, size=anomaly_map_ad.shape[-2:], mode='bilinear', align_corners=False)
                    seg_map_list.append(seg_map.detach().cpu())
                    # Release intermediate variables immediately
                    del pixel_logits, seg_map
                else:
                    seg_map_list.append(None)
                anomaly_map_list.append(anomaly_map_ad_cpu)
                comp_anomaly_map_list.append(comp_anomaly_map_cpu)
                fused_anomaly_map_list.append(fused_anomaly_map_cpu)
                if disc_output_batch is not None:
                    disc_output_list.append(disc_output_batch.detach().cpu())
                else:
                    disc_output_list.append(None)
                if cluster_map_batch is not None:
                    cluster_map_list.append(cluster_map_batch)  # List of numpy arrays
                else:
                    cluster_map_list.append(None)
                if recon_map_batch is not None:
                    recon_map_list.append(recon_map_batch)  # List of numpy arrays
                else:
                    recon_map_list.append(None)
                gt_list.append(gt_cpu)
                img_path_list.append(img_path)
                
                # After storing visualization data, release GPU tensor copies
                del anomaly_map_ad_cpu, comp_anomaly_map_cpu, fused_anomaly_map_cpu, gt_cpu, img_cpu
                if disc_output_batch is not None:
                    del disc_output_batch
            
            # Compute image-level scores from fused map; move to CPU immediately
            if compute_metrics:
                if max_ratio == 0:
                    sp_score = torch.max(fused_anomaly_map.flatten(1), dim=1)[0]
                else:
                    fused_anomaly_map_flat = fused_anomaly_map.flatten(1)
                    sp_score = torch.sort(fused_anomaly_map_flat, dim=1, descending=True)[0][:, :int(fused_anomaly_map_flat.shape[1] * max_ratio)]
                    sp_score = sp_score.mean(dim=1)
                    del fused_anomaly_map_flat  # Release immediately
                pr_list_sp.append(sp_score.detach().cpu())
                del sp_score  # Release immediately
            
            # After each batch, release GPU intermediates once data is stored
            del en, de
            if has_seg_output:
                del mask_logits, class_logits
            # Release GPU copies (already stored on CPU)
            if vis_path is None or skip_visualization:
                # Release immediately when visualization is disabled
                if 'anomaly_map_ad' in locals():
                    del anomaly_map_ad
                if 'comp_anomaly_map' in locals() and comp_anomaly_map is not None:
                    del comp_anomaly_map
                if 'fused_anomaly_map' in locals():
                    del fused_anomaly_map
                if 'gt' in locals():
                    del gt
                if 'img' in locals():
                    del img
            else:
                # When visualizing, release GPU copies after CPU storage
                if 'anomaly_map_ad' in locals():
                    del anomaly_map_ad
                if 'comp_anomaly_map' in locals() and comp_anomaly_map is not None:
                    del comp_anomaly_map
                if 'fused_anomaly_map' in locals():
                    del fused_anomaly_map
                if 'gt' in locals():
                    del gt
                if 'img' in locals():
                    del img
            # Periodically clear GPU cache
            if (len(gt_list_px) % 50 == 0):
                torch.cuda.empty_cache()
                import gc
                gc.collect()
    
    # ===== Inference loop finished; clean up all inference variables =====
    print("Inference completed. Cleaning up inference-related variables...")
    # Release any remaining inference variables
    if 'gaussian_kernel' in locals():
        del gaussian_kernel
    if 'has_seg_output' in locals():
        del has_seg_output
    if 'output' in locals():
        del output
    if 'B' in locals():
        del B
    # Release all intermediate variable references
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("Inference cleanup completed.")
    

    if vis_path is not None and not skip_visualization and len(img_list) > 0:
        print(f"Saving visualization results to {vis_path}...")
        visualize_composition_eval(
            vis_path=vis_path,
            vis_img_num=vis_img_num,
            n_clusters=n_clusters,
            img_list=img_list,
            img_path_list=img_path_list,
            seg_map_list=seg_map_list,
            anomaly_map_list=anomaly_map_list,
            comp_anomaly_map_list=comp_anomaly_map_list,
            fused_anomaly_map_list=fused_anomaly_map_list,
            disc_output_list=disc_output_list,
            cluster_map_list=cluster_map_list,
            recon_map_list=recon_map_list,
            gt_list=gt_list,
            all_ad_scores=all_ad_scores,
            all_comp_scores=all_comp_scores,
            all_disc_scores=all_disc_scores,
            all_fused_scores=all_fused_scores,
        )

    if not skip_visualization and len(test_ad_pixel_scores) > 0:
        save_test_histograms(
            vis_path,
            save_name,
            test_ad_pixel_scores,
            test_comp_pixel_scores,
            test_ad_pixel_scores_norm,
            test_comp_pixel_scores_norm,
        )
        del test_ad_pixel_scores, test_comp_pixel_scores, test_ad_pixel_scores_norm, test_comp_pixel_scores_norm

    # ===== Release visualization data; keep only what metrics need
    if vis_path is not None and not skip_visualization:
        print("Releasing visualization data to free memory...")
        # Release all visualization-related lists
        if 'img_list' in locals():
            del img_list
        if 'seg_map_list' in locals():
            del seg_map_list
        if 'anomaly_map_list' in locals():
            del anomaly_map_list
        if 'comp_anomaly_map_list' in locals():
            del comp_anomaly_map_list
        if 'fused_anomaly_map_list' in locals():
            del fused_anomaly_map_list
        if 'disc_output_list' in locals():
            del disc_output_list
        if 'cluster_map_list' in locals():
            del cluster_map_list
        if 'recon_map_list' in locals():
            del recon_map_list
        if 'gt_list' in locals():
            del gt_list
        if 'img_path_list' in locals():
            del img_path_list
        if 'all_ad_scores' in locals():
            del all_ad_scores
        if 'all_comp_scores' in locals():
            del all_comp_scores
        if 'all_disc_scores' in locals():
            del all_disc_scores
        if 'all_fused_scores' in locals():
            del all_fused_scores
        # Release global percentile variables (including min/max)
        if 'global_ad_p25' in locals():
            del global_ad_p25, global_ad_p99
        if 'global_ad_min' in locals():
            del global_ad_min, global_ad_max
        if 'global_comp_p25' in locals():
            del global_comp_p25, global_comp_p99
        if 'global_comp_min' in locals():
            del global_comp_min, global_comp_max
        if 'global_disc_p25' in locals():
            del global_disc_p25, global_disc_p99
        if 'global_disc_min' in locals():
            del global_disc_min, global_disc_max
        if 'global_fused_p25' in locals():
            del global_fused_p25, global_fused_p99
        if 'global_fused_min' in locals():
            del global_fused_min, global_fused_max
        # Close all matplotlib figures
        plt.close('all')
        import matplotlib
        matplotlib.pyplot.close('all')
        # Free GPU and CPU memory
        torch.cuda.empty_cache()
        import gc
        gc.collect()  # Force garbage collection
        print("Visualization data released.")
    
    # ===== Compute quantitative metrics (last, after visualization cleanup)
    if compute_metrics:
        # Release any remaining inference variables
        print("Releasing inference-related variables before metric calculation...")
        # Release all variable references
        if 'gaussian_kernel' in locals():
            del gaussian_kernel
        if 'has_seg_output' in locals():
            del has_seg_output
        if 'output' in locals():
            del output
        if 'B' in locals():
            del B
        if 'use_normalization' in locals():
            del use_normalization
        if 'ad_mean' in locals():
            del ad_mean, ad_std, comp_mean, comp_std
        # Free GPU memory
        torch.cuda.empty_cache()
        import gc
        gc.collect()  # Force garbage collection
        print("Inference variables released.")
        
        print("Calculating evaluation metrics...")
        # Process in batches to limit memory peaks
        # Note: gt_list_px and pr_list_px were moved to CPU during collection
        print("  Concatenating pixel-level predictions and ground truth...")
        # Concatenate and release source lists to avoid duplicate storage
        gt_list_px_cat = torch.cat(gt_list_px, dim=0)[:, 0]  # Already on CPU
        del gt_list_px  # Release immediately
        torch.cuda.empty_cache()
        gc.collect()
        
        pr_list_px_cat = torch.cat(pr_list_px, dim=0)[:, 0]  # Already on CPU
        del pr_list_px  # Release immediately
        torch.cuda.empty_cache()
        gc.collect()
        
        gt_list_sp_cat = torch.cat(gt_list_sp).flatten()  # Already on CPU
        del gt_list_sp  # Release immediately
        torch.cuda.empty_cache()
        gc.collect()
        
        pr_list_sp_cat = torch.cat(pr_list_sp).flatten()  # Already on CPU
        del pr_list_sp  # Release immediately
        torch.cuda.empty_cache()
        gc.collect()
        
        print("  Original lists released.")
        
        # Convert to numpy in batches to limit memory peaks
        print("  Converting to numpy arrays...")
        # For large arrays, convert in chunks and release tensors immediately
        batch_size_np = 1000  # Convert 1000 images per chunk
        num_images = gt_list_px_cat.shape[0]
        
        if num_images > batch_size_np:
            # Batch-convert to numpy; release tensor slices immediately
            gt_list_px_np_list = []
            pr_list_px_np_list = []
            for i in range(0, num_images, batch_size_np):
                end_idx = min(i + batch_size_np, num_images)
                # Convert slice and release tensor immediately
                gt_slice = gt_list_px_cat[i:end_idx]
                pr_slice = pr_list_px_cat[i:end_idx]
                gt_list_px_np_list.append(gt_slice.numpy())
                pr_list_px_np_list.append(pr_slice.numpy())
                del gt_slice, pr_slice  # Release tensor slice immediately
                if (i // batch_size_np) % 10 == 0:  # Clean up every 10 batches
                    torch.cuda.empty_cache()
                    gc.collect()
            gt_list_px_np = np.concatenate(gt_list_px_np_list, axis=0)
            pr_list_px_np = np.concatenate(pr_list_px_np_list, axis=0)
            del gt_list_px_np_list, pr_list_px_np_list
        else:
            gt_list_px_np = gt_list_px_cat.numpy()
            pr_list_px_np = pr_list_px_cat.numpy()
        
        # Release pixel-level tensors after numpy conversion
        del gt_list_px_cat, pr_list_px_cat
        torch.cuda.empty_cache()
        gc.collect()
        
        # Convert image-level data
        gt_list_sp_np = gt_list_sp_cat.numpy()
        pr_list_sp_np = pr_list_sp_cat.numpy()
        
        # Release image-level tensor memory
        del gt_list_sp_cat, pr_list_sp_cat
        torch.cuda.empty_cache()
        gc.collect()
        
        print("  Tensor memory released.")
        
        print("  Computing metrics...")
        from utils import ader_evaluator
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px_np, pr_list_sp_np, gt_list_px_np, gt_list_sp_np)
        
        # Release numpy arrays
        print("  Releasing numpy arrays...")
        del gt_list_px_np, pr_list_px_np, gt_list_sp_np, pr_list_sp_np
        torch.cuda.empty_cache()
        gc.collect()
        
        return auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px
    else:
        return None

