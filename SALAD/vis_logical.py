import gc
import os
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from SALAD.composition_anomaly import onehot_to_cluster
from utils import (
    denormalize,
    save_anomaly_jet_heatmap,
    save_anomaly_overlay,
    visualize_clustering_result,
)


def _maybe_downsample(arr, max_samples=10_000_000):
    if len(arr) > max_samples:
        indices = np.random.choice(len(arr), size=max_samples, replace=False)
        return arr[indices]
    return arr


def _save_hist(data, path, xlabel, title, color=None):
    plt.figure(figsize=(10, 6))
    kwargs = dict(bins=100, alpha=0.7, edgecolor="black")
    if color is not None:
        kwargs["color"] = color
    plt.hist(data, **kwargs)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_norm_histograms(save_dir, ad_pixel_scores, comp_pixel_scores, ad_mean, ad_std, comp_mean, comp_std):
    """Validation-set pixel-score histograms before/after z-score normalization."""
    if save_dir is None or len(ad_pixel_scores) == 0:
        return
    os.makedirs(save_dir, exist_ok=True)
    max_samples = 10_000_000
    ad_pixel_all = _maybe_downsample(torch.cat(ad_pixel_scores).numpy(), max_samples)
    if len(comp_pixel_scores) > 0:
        comp_pixel_all = _maybe_downsample(torch.cat(comp_pixel_scores).numpy(), max_samples)
    else:
        comp_pixel_all = np.array([])

    _save_hist(
        ad_pixel_all,
        os.path.join(save_dir, "ad_branch_pixel_score_histogram_before_norm.png"),
        "Pixel-level Anomaly Score (AD Branch, Before Normalization)",
        f"Distribution of Pixel-level Anomaly Scores (AD Branch, Validation Set, Before Normalization)\nMean={np.mean(ad_pixel_all):.4f}, Std={np.std(ad_pixel_all):.4f}",
    )
    if len(comp_pixel_all) > 0:
        _save_hist(
            comp_pixel_all,
            os.path.join(save_dir, "composition_branch_pixel_score_histogram_before_norm.png"),
            "Pixel-level Anomaly Score (Composition Branch, Before Normalization)",
            f"Distribution of Pixel-level Anomaly Scores (Composition Branch, Validation Set, Before Normalization)\nMean={np.mean(comp_pixel_all):.4f}, Std={np.std(comp_pixel_all):.4f}",
            color="orange",
        )

    ad_pixel_all_norm = _maybe_downsample((torch.cat(ad_pixel_scores).numpy() - ad_mean) / ad_std, max_samples)
    _save_hist(
        ad_pixel_all_norm,
        os.path.join(save_dir, "ad_branch_pixel_score_histogram_after_norm.png"),
        "Pixel-level Anomaly Score (AD Branch, After Normalization)",
        f"Distribution of Pixel-level Anomaly Scores (AD Branch, Validation Set, After Normalization)\nMean={np.mean(ad_pixel_all_norm):.4f}, Std={np.std(ad_pixel_all_norm):.4f}",
        color="green",
    )
    if len(comp_pixel_scores) > 0:
        comp_pixel_all_norm = _maybe_downsample((torch.cat(comp_pixel_scores).numpy() - comp_mean) / comp_std, max_samples)
        _save_hist(
            comp_pixel_all_norm,
            os.path.join(save_dir, "composition_branch_pixel_score_histogram_after_norm.png"),
            "Pixel-level Anomaly Score (Composition Branch, After Normalization)",
            f"Distribution of Pixel-level Anomaly Scores (Composition Branch, Validation Set, After Normalization)\nMean={np.mean(comp_pixel_all_norm):.4f}, Std={np.std(comp_pixel_all_norm):.4f}",
            color="red",
        )
    print(f"Histograms saved to {save_dir}")


def save_test_histograms(
    vis_path,
    save_name,
    test_ad_pixel_scores,
    test_comp_pixel_scores,
    test_ad_pixel_scores_norm,
    test_comp_pixel_scores_norm,
):
    """Test-set pixel-score histograms before/after z-score normalization."""
    if vis_path is not None:
        vis_path_parts = vis_path.replace("\\", "/").split("/")
        if len(vis_path_parts) >= 2:
            dataset_name = vis_path_parts[-2] if len(vis_path_parts) >= 2 else "unknown"
            save_dir = os.path.join("visualize_analysis", vis_path_parts[-1], dataset_name)
        else:
            save_dir = os.path.join("visualize_analysis", "test_set")
    elif save_name is not None:
        save_dir = os.path.join("visualize_analysis", save_name, "test_set")
    else:
        save_dir = os.path.join("visualize_analysis", "test_set")

    os.makedirs(save_dir, exist_ok=True)
    print("Drawing test set histograms...")
    print(f"  Merging pixel scores... (AD: {len(test_ad_pixel_scores)} batches, Comp: {len(test_comp_pixel_scores)} batches)")
    sys.stdout.flush()
    max_samples = 10_000_000
    test_ad_pixel_all = torch.cat(test_ad_pixel_scores).numpy()
    print(f"  AD branch: {len(test_ad_pixel_all)} pixels (before sampling)")
    sys.stdout.flush()
    test_ad_pixel_all = _maybe_downsample(test_ad_pixel_all, max_samples)
    if len(test_comp_pixel_scores) > 0:
        test_comp_pixel_all = torch.cat(test_comp_pixel_scores).numpy()
        print(f"  Composition branch: {len(test_comp_pixel_all)} pixels (before sampling)")
        sys.stdout.flush()
        test_comp_pixel_all = _maybe_downsample(test_comp_pixel_all, max_samples)
    else:
        test_comp_pixel_all = np.array([])

    print("  Drawing AD branch histogram (before normalization)...")
    _save_hist(
        test_ad_pixel_all,
        os.path.join(save_dir, "test_ad_branch_pixel_score_histogram_before_norm.png"),
        "Pixel-level Anomaly Score (AD Branch, Before Normalization)",
        f"Distribution of Pixel-level Anomaly Scores (AD Branch, Test Set, Before Normalization)\nMean={np.mean(test_ad_pixel_all):.4f}, Std={np.std(test_ad_pixel_all):.4f}",
        color="blue",
    )
    print("  AD branch histogram saved.")
    if len(test_comp_pixel_all) > 0:
        print("  Drawing Composition branch histogram (before normalization)...")
        _save_hist(
            test_comp_pixel_all,
            os.path.join(save_dir, "test_composition_branch_pixel_score_histogram_before_norm.png"),
            "Pixel-level Anomaly Score (Composition Branch, Before Normalization)",
            f"Distribution of Pixel-level Anomaly Scores (Composition Branch, Test Set, Before Normalization)\nMean={np.mean(test_comp_pixel_all):.4f}, Std={np.std(test_comp_pixel_all):.4f}",
            color="orange",
        )
        print("  Composition branch histogram saved.")
    if len(test_ad_pixel_scores_norm) > 0:
        print("  Drawing AD branch histogram (after normalization)...")
        test_ad_pixel_all_norm = _maybe_downsample(torch.cat(test_ad_pixel_scores_norm).numpy(), max_samples)
        _save_hist(
            test_ad_pixel_all_norm,
            os.path.join(save_dir, "test_ad_branch_pixel_score_histogram_after_norm.png"),
            "Pixel-level Anomaly Score (AD Branch, After Normalization)",
            f"Distribution of Pixel-level Anomaly Scores (AD Branch, Test Set, After Normalization)\nMean={np.mean(test_ad_pixel_all_norm):.4f}, Std={np.std(test_ad_pixel_all_norm):.4f}",
            color="green",
        )
        print("  AD branch histogram (after normalization) saved.")
    if len(test_comp_pixel_scores_norm) > 0:
        print("  Drawing Composition branch histogram (after normalization)...")
        test_comp_pixel_all_norm = _maybe_downsample(torch.cat(test_comp_pixel_scores_norm).numpy(), max_samples)
        _save_hist(
            test_comp_pixel_all_norm,
            os.path.join(save_dir, "test_composition_branch_pixel_score_histogram_after_norm.png"),
            "Pixel-level Anomaly Score (Composition Branch, After Normalization)",
            f"Distribution of Pixel-level Anomaly Scores (Composition Branch, Test Set, After Normalization)\nMean={np.mean(test_comp_pixel_all_norm):.4f}, Std={np.std(test_comp_pixel_all_norm):.4f}",
            color="red",
        )
        print("  Composition branch histogram (after normalization) saved.")
    print(f"Test set histograms saved to {save_dir}")
    plt.close("all")
    matplotlib.pyplot.close("all")
    torch.cuda.empty_cache()
    gc.collect()
    print("Histogram data released.")


def visualize_composition_eval(
    vis_path,
    vis_img_num,
    n_clusters,
    img_list,
    img_path_list,
    seg_map_list,
    anomaly_map_list,
    comp_anomaly_map_list,
    fused_anomaly_map_list,
    disc_output_list,
    cluster_map_list,
    recon_map_list,
    gt_list,
    all_ad_scores,
    all_comp_scores,
    all_disc_scores,
    all_fused_scores,
):
    """Save per-image heatmaps / cluster maps / discriminator outputs for the composition eval."""
    
    # Compute global 25th and 99th percentiles (for viridis colormap)
    # Free GPU memory
    torch.cuda.empty_cache()
    
    global_ad_p25, global_ad_p99 = None, None
    global_comp_p25, global_comp_p99 = None, None
    global_disc_p25, global_disc_p99 = None, None
    global_fused_p25, global_fused_p99 = None, None
    
    if len(all_ad_scores) > 0:
        all_ad_scores_flat = torch.cat(all_ad_scores).numpy()
        global_ad_p25 = float(np.percentile(all_ad_scores_flat, 25))
        global_ad_p99 = float(np.percentile(all_ad_scores_flat, 99))
        global_ad_min = float(all_ad_scores_flat.min())
        global_ad_max = float(all_ad_scores_flat.max())
        print(f"Global AD branch score range: [{global_ad_min:.4f}, {global_ad_max:.4f}], 25%-99% percentile: [{global_ad_p25:.4f}, {global_ad_p99:.4f}]")
        # Release memory immediately
        del all_ad_scores_flat
        del all_ad_scores
        all_ad_scores = None  # Set to None instead of empty list
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    if len(all_comp_scores) > 0:
        all_comp_scores_flat = torch.cat(all_comp_scores).numpy()
        global_comp_p25 = float(np.percentile(all_comp_scores_flat, 25))
        global_comp_p99 = float(np.percentile(all_comp_scores_flat, 99))
        global_comp_min = float(all_comp_scores_flat.min())
        global_comp_max = float(all_comp_scores_flat.max())
        print(f"Global Composition branch score range: [{global_comp_min:.4f}, {global_comp_max:.4f}], 25%-99% percentile: [{global_comp_p25:.4f}, {global_comp_p99:.4f}]")
        # Release memory immediately
        del all_comp_scores_flat
        del all_comp_scores
        all_comp_scores = None
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    if len(all_disc_scores) > 0:
        all_disc_scores_flat = torch.cat(all_disc_scores).numpy()
        global_disc_p25 = float(np.percentile(all_disc_scores_flat, 25))
        global_disc_p99 = float(np.percentile(all_disc_scores_flat, 99))
        global_disc_min = float(all_disc_scores_flat.min())
        global_disc_max = float(all_disc_scores_flat.max())
        print(f"Global Discriminator output score range: [{global_disc_min:.4f}, {global_disc_max:.4f}], 25%-99% percentile: [{global_disc_p25:.4f}, {global_disc_p99:.4f}]")
        # Release memory immediately
        del all_disc_scores_flat
        del all_disc_scores
        all_disc_scores = None
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    if len(all_fused_scores) > 0:
        all_fused_scores_flat = torch.cat(all_fused_scores).numpy()
        global_fused_p25 = float(np.percentile(all_fused_scores_flat, 25))
        global_fused_p99 = float(np.percentile(all_fused_scores_flat, 99))
        global_fused_min = float(all_fused_scores_flat.min())
        global_fused_max = float(all_fused_scores_flat.max())
        print(f"Global Fused anomaly map score range: [{global_fused_min:.4f}, {global_fused_max:.4f}], 25%-99% percentile: [{global_fused_p25:.4f}, {global_fused_p99:.4f}]")
        # Release memory immediately
        del all_fused_scores_flat
        del all_fused_scores
        all_fused_scores = None
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    # Group by defect type
    defect_type_to_items = {}  # {defect_type: [(batch_idx, item_idx, img_path, ...), ...]}
    
    # Collect all images and their defect types
    for idx in range(len(img_list)):
        img_batch = img_list[idx]
        img_path_batch = img_path_list[idx]
        batch_size = img_batch.shape[0]
        
        for i in range(batch_size):
            img_path_item = img_path_batch[i].replace('\\', '/').split('/')
            # Extract defect type from path: .../test/good/xxx.png or .../test/broken/xxx.png
            defect_type = None
            if 'test' in img_path_item:
                test_idx = img_path_item.index('test')
                if test_idx + 1 < len(img_path_item):
                    defect_type = img_path_item[test_idx + 1]  # Directory name after test/ is the defect type
            
            if defect_type is None:
                # If path parsing fails, try fallback heuristics
                if len(img_path_item) >= 2:
                    defect_type = img_path_item[-2]  # Second-to-last path segment may be defect type
                else:
                    defect_type = 'unknown'
            
            if defect_type not in defect_type_to_items:
                defect_type_to_items[defect_type] = []
            
            defect_type_to_items[defect_type].append((idx, i, img_path_item))
        
        # Release loop intermediate references
        del img_batch, img_path_batch
    
    # Visualize per defect type; cap at vis_img_num images per class
    for defect_type, items in defect_type_to_items.items():
        # Limit number of images visualized per class
        items_to_vis = items[:vis_img_num]
        
        # Save directory: vis_path/MIAD/nut_and_bolt/good/, etc.
        # Extract dataset_name and class_name from first item path
        if len(items_to_vis) > 0:
            _, _, img_path_item = items_to_vis[0]
            if len(img_path_item) >= 5:
                dataset_name = img_path_item[-5]
                class_name = img_path_item[-4]
                save_dir = os.path.join(vis_path, dataset_name, class_name, defect_type)
                os.makedirs(save_dir, exist_ok=True)
            elif len(img_path_item) >= 4:
                # If path is too short, use second/third-to-last segments
                dataset_name = img_path_item[-3] if len(img_path_item) >= 3 else 'unknown'
                class_name = img_path_item[-2] if len(img_path_item) >= 2 else 'unknown'
                save_dir = os.path.join(vis_path, dataset_name, class_name, defect_type)
                os.makedirs(save_dir, exist_ok=True)
            else:
                save_dir = os.path.join(vis_path, defect_type)
                os.makedirs(save_dir, exist_ok=True)
            del img_path_item, dataset_name, class_name  # Release intermediate variables
        else:
            del items_to_vis  # Release empty list
            continue
        
        print(f"Visualizing {len(items_to_vis)} images for defect type: {defect_type}")
        
        for batch_idx, item_idx, img_path_item in tqdm(items_to_vis, ncols=80, desc=f"Visualizing {defect_type}"):
            img_batch = img_list[batch_idx]
            seg_map_batch = seg_map_list[batch_idx] if seg_map_list[batch_idx] is not None else None
            anomaly_map_ad_batch = anomaly_map_list[batch_idx]
            comp_anomaly_map_batch = comp_anomaly_map_list[batch_idx]
            fused_anomaly_map_batch = fused_anomaly_map_list[batch_idx]
            disc_output_batch = disc_output_list[batch_idx] if disc_output_list[batch_idx] is not None else None
            cluster_map_batch = cluster_map_list[batch_idx] if cluster_map_list[batch_idx] is not None else None
            recon_map_batch = recon_map_list[batch_idx] if recon_map_list[batch_idx] is not None else None
            gt_batch = gt_list[batch_idx]
            
            img_name = os.path.splitext(os.path.basename(img_path_item[-1]))[0]
            cv2_img = np.array(
                denormalize(img_batch[item_idx].clone().squeeze(0).cpu().detach().numpy()),
                dtype=np.uint8,
            )
            
            # Save original image
            plt.imsave(os.path.join(save_dir, f'{img_name}_img.png'), cv2_img)
            
            # Save ground truth
            gt_np = gt_batch[item_idx].squeeze(0).numpy()
            plt.imsave(os.path.join(save_dir, f'{img_name}_gt.png'), gt_np, cmap='gray')
            del gt_np  # Release immediately
            
            # Save segmentation mask
            if seg_map_batch is not None:
                seg_np = seg_map_batch[item_idx].squeeze(0).numpy()
                plt.imsave(os.path.join(save_dir, f'{img_name}_seg.png'), seg_np, cmap='gray')
                del seg_np  # Release immediately
            
            # Save AD branch anomaly map (global 25th–99th percentile scaling)
            ad_np = anomaly_map_ad_batch[item_idx].squeeze(0).numpy()
            if global_ad_p25 is not None and global_ad_p99 is not None:
                plt.imsave(os.path.join(save_dir, f'{img_name}_ad.png'), ad_np, cmap='viridis', vmin=global_ad_p25, vmax=global_ad_p99)
            else:
                # Fallback to per-image min-max normalization if no global stats
                ad_np_normalized = (ad_np - ad_np.min()) / (ad_np.max() - ad_np.min() + 1e-8)
                plt.imsave(os.path.join(save_dir, f'{img_name}_ad.png'), ad_np_normalized, cmap='viridis', vmin=0, vmax=1)
                del ad_np_normalized  # Release immediately
            save_anomaly_overlay(
                cv2_img, ad_np,
                os.path.join(save_dir, f'{img_name}_ad_overlay.png'),
                vmin=global_ad_p25, vmax=global_ad_p99,
            )
            del ad_np  # Release immediately
            
            # Save composition branch anomaly map (global 25th–99th percentile, JET colormap)
            comp_np = comp_anomaly_map_batch[item_idx].squeeze(0).numpy()
            h_img, w_img = cv2_img.shape[:2]
            if global_comp_p25 is not None and global_comp_p99 is not None:
                save_anomaly_jet_heatmap(
                    comp_np,
                    os.path.join(save_dir, f'{img_name}_comp.png'),
                    target_size=(w_img, h_img),
                    vmin=global_comp_p25,
                    vmax=global_comp_p99,
                )
            else:
                # Fallback to per-image min-max normalization if no global stats
                save_anomaly_jet_heatmap(
                    comp_np,
                    os.path.join(save_dir, f'{img_name}_comp.png'),
                    target_size=(w_img, h_img),
                )
            save_anomaly_overlay(
                cv2_img, comp_np,
                os.path.join(save_dir, f'{img_name}_comp_overlay.png'),
                vmin=global_comp_p25, vmax=global_comp_p99,
            )
            del comp_np  # Release immediately
            
            # Save fused anomaly map (global 25th–99th percentile scaling)
            fused_np = fused_anomaly_map_batch[item_idx].squeeze(0).numpy()
            if global_fused_p25 is not None and global_fused_p99 is not None:
                plt.imsave(os.path.join(save_dir, f'{img_name}_fused.png'), fused_np, cmap='viridis', vmin=global_fused_p25, vmax=global_fused_p99)
            else:
                # Fallback to per-image min-max normalization if no global stats
                fused_np_normalized = (fused_np - fused_np.min()) / (fused_np.max() - fused_np.min() + 1e-8)
                plt.imsave(os.path.join(save_dir, f'{img_name}_fused.png'), fused_np_normalized, cmap='viridis', vmin=0, vmax=1)
                del fused_np_normalized  # Release immediately
            save_anomaly_overlay(
                cv2_img, fused_np,
                os.path.join(save_dir, f'{img_name}_fused_overlay.png'),
                vmin=global_fused_p25, vmax=global_fused_p99,
            )
            del fused_np  # Release immediately
            
            # Store cluster map (37×37, upscaled to image size via nearest-neighbor)
            if cluster_map_batch is not None and item_idx < len(cluster_map_batch):
                cluster_map_np = cluster_map_batch[item_idx]  # (37, 37)
                target_size = img_batch[item_idx].shape[1:]  # (H, W)
                visualize_clustering_result(
                    cluster_map_np,
                    f'{img_name}_cluster',
                    save_dir,
                    target_size,
                    n_clusters=n_clusters
                )
                del cluster_map_np, target_size  # Release immediately
            
            # Save reconstruction output (37×37, upscaled via nearest-neighbor)
            if recon_map_batch is not None and item_idx < len(recon_map_batch):
                recon_map_np = recon_map_batch[item_idx]  # (n_clusters, 37, 37)
                recon_cluster = onehot_to_cluster(recon_map_np)  # (37, 37)
                target_size = img_batch[item_idx].shape[1:]  # (H, W)
                visualize_clustering_result(
                    recon_cluster,
                    f'{img_name}_recon',
                    save_dir,
                    target_size,
                    n_clusters=n_clusters
                )
                del recon_map_np, recon_cluster, target_size  # Release immediately
            
            # Save discriminator output (37×37, nearest-neighbor upscale, viridis, global 25th–99th percentile)
            if disc_output_batch is not None:
                disc_np = disc_output_batch[item_idx].numpy()  # (37, 37)
                # Upscale to image size via nearest-neighbor (no blur)
                target_size = img_batch[item_idx].shape[1:]  # (H, W)
                disc_np_resized = cv2.resize(
                    disc_np.astype(np.float32), 
                    (target_size[1], target_size[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
                # Map with global 25th–99th percentiles (computed after resizing to heatmap size)
                # For visualization, upscale to original image size via nearest-neighbor
                if global_disc_p25 is not None and global_disc_p99 is not None:
                    plt.imsave(os.path.join(save_dir, f'{img_name}_disc.png'), disc_np_resized, cmap='viridis', vmin=global_disc_p25, vmax=global_disc_p99)
                else:
                    # Fallback to per-image min-max normalization if no global stats
                    disc_np_normalized = (disc_np_resized - disc_np_resized.min()) / (disc_np_resized.max() - disc_np_resized.min() + 1e-8)
                    plt.imsave(os.path.join(save_dir, f'{img_name}_disc.png'), disc_np_normalized, cmap='viridis', vmin=0, vmax=1)
                    del disc_np_normalized  # Release immediately
                del disc_np, disc_np_resized, target_size  # Release immediately
            
            # Release batch references (references only; frees loop scope)
            del cv2_img, img_batch, seg_map_batch, anomaly_map_ad_batch, comp_anomaly_map_batch, fused_anomaly_map_batch
            del disc_output_batch, cluster_map_batch, recon_map_batch, gt_batch
            
            # Clean memory every 10 images
            if (item_idx % 10 == 0):
                plt.close('all')
                import gc
                gc.collect()
        
        # After each defect type, release related variables
        del items_to_vis, save_dir
        plt.close('all')
        import gc
        gc.collect()
    
    plt.close('all')
    # Release matplotlib memory
    import matplotlib
    matplotlib.pyplot.close('all')
    # Release visualization loop intermediates and batch references
    if 'defect_type_to_items' in locals():
        del defect_type_to_items
    # Release all batch data references created in the loop
    # Note: items_to_vis and save_dir are released per defect-type loop
    # Double-check cleanup for completeness
    if 'items_to_vis' in locals():
        del items_to_vis
    if 'save_dir' in locals():
        del save_dir
    if 'items' in locals():
        del items
    if 'defect_type' in locals():
        del defect_type
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("Visualization loop completed and memory cleaned.")

