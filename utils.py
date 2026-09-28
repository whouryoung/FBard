from collections import defaultdict
import torch
from numpy.random import normal
import random
import logging
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, roc_curve
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
import os
from functools import partial
import math
from tqdm import tqdm
from aug_funcs import rot_img, translation_img, hflip_img, grey_img, rot90_img
import torch.backends.cudnn as cudnn
from adeval import EvalAccumulatorCuda
from pathlib import Path


# Compute evaluation metrics with optional GPU acceleration
def ader_evaluator(pr_px, pr_sp, gt_px, gt_sp,
                   use_metrics=['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO']):
    if len(gt_px.shape) == 4:
        gt_px = gt_px.squeeze(1)
    if len(pr_px.shape) == 4:
        pr_px = pr_px.squeeze(1)

    score_min = min(pr_sp)
    score_max = max(pr_sp)
    anomap_min = pr_px.min()
    anomap_max = pr_px.max()

    accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max, skip_pixel_aupro=False, nstrips=200)
    accum.add_anomap_batch(torch.tensor(pr_px).cuda(non_blocking=True),
                           torch.tensor(gt_px.astype(np.uint8)).cuda(non_blocking=True))

    # for i in range(torch.tensor(pr_px).size(0)):
    #     accum.add_image(torch.tensor(pr_sp[i]), torch.tensor(gt_sp[i]))

    metrics = accum.summary()
    metric_results = {}
    for metric in use_metrics:
        if metric.startswith('I-AUROC'):
            auroc_sp = roc_auc_score(gt_sp, pr_sp)
            metric_results[metric] = auroc_sp
        elif metric.startswith('I-AP'):
            ap_sp = average_precision_score(gt_sp, pr_sp)
            metric_results[metric] = ap_sp
        elif metric.startswith('I-F1_max'):
            best_f1_score_sp = f1_score_max(gt_sp, pr_sp)
            metric_results[metric] = best_f1_score_sp
        elif metric.startswith('P-AUROC'):
            metric_results[metric] = metrics['p_auroc']
        elif metric.startswith('P-AP'):
            metric_results[metric] = metrics['p_aupr']
        elif metric.startswith('P-F1_max'):
            best_f1_score_px = f1_score_max(gt_px.ravel(), pr_px.ravel())
            metric_results[metric] = best_f1_score_px
        elif metric.startswith('AUPRO'):
            metric_results[metric] = metrics['p_aupro']
    return list(metric_results.values())


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    return logger


def setup_seed(seed):
    torch.manual_seed(seed)  # Set PyTorch CPU random seed
    torch.cuda.manual_seed(seed)  # Set random seed for the current GPU
    torch.cuda.manual_seed_all(seed)  # Set random seeds for all GPUs (multi-GPU)
    np.random.seed(seed)  # Set NumPy random seed
    random.seed(seed)  # Set Python stdlib random seed
    torch.backends.cudnn.deterministic = True  # Ensure CuDNN uses deterministic algorithms
    torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmark optimization


def augmentation(img):
    img = img.unsqueeze(0)
    augment_img = img
    for angle in [-np.pi / 4, -3 * np.pi / 16, -np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, 3 * np.pi / 16,
                  np.pi / 4]:
        rotate_img = rot_img(img, angle)
        augment_img = torch.cat([augment_img, rotate_img], dim=0)
        # translate img
    for a, b in [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2), (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1),
                 (0.1, -0.1)]:
        trans_img = translation_img(img, a, b)
        augment_img = torch.cat([augment_img, trans_img], dim=0)
        # hflip img
    flipped_img = hflip_img(img)
    augment_img = torch.cat([augment_img, flipped_img], dim=0)
    # rgb to grey img
    greyed_img = grey_img(img)
    augment_img = torch.cat([augment_img, greyed_img], dim=0)
    # rotate img in 90 degree
    for angle in [1, 2, 3]:
        rotate90_img = rot90_img(img, angle)
        augment_img = torch.cat([augment_img, rotate90_img], dim=0)
    augment_img = (augment_img[torch.randperm(augment_img.size(0))])
    return augment_img


def modify_grad(x, inds, factor=0.):
    # print(inds.shape)
    inds = inds.expand_as(x)
    # print(x.shape)
    # print(inds.shape)
    x[inds] *= factor
    return x


def modify_grad_v2(x, factor):  # x is the gradient tensor, factor is the scale; returns x * factor
    factor = factor.expand_as(x)
    x *= factor
    return x


# Overall: use flattened global cosine similarity as the base; weight each point by its cosine similarity relative to the mean
def global_cosine_hm_adaptive(a, b, y=3):
    cos_loss = torch.nn.CosineSimilarity()  # Cosine similarity module (per-point)
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()  # (1, 768, 28, 28)
        b_ = b[item]  # (1, 768, 28, 28)
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach()  # Per-point cosine distance (1, 1, 28, 28)
        mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        # thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        factor = (point_dist / mean_dist) ** (y)  # y controls penalty strength; factor scales gradients via multiplication
        # factor = factor/torch.max(factor)
        # factor = torch.clip(factor, min=min_grad)
        # print(thresh)
        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),  # (B, C*H*W)
                                        b_.reshape(b_.shape[0], -1)))
        partial_func = partial(modify_grad_v2, factor=factor)  # Bind factor to modify_grad_v2 via partial
        b_.register_hook(partial_func)  # Scale decoder backprop gradients by per-point cosine similarity
    loss = loss / len(a)
    return loss


# Custom global weighted rank pooling: larger values receive higher weights
def global_weighted_rank_pooling(x, alpha=5.0):
    x_ = x.detach()
    sorted_indices = torch.argsort(x_, descending=True)
    n = x.size(0)
    ranks = torch.arange(1, n + 1, dtype=torch.float32, device=x.device)
    weights = torch.exp(-alpha * (ranks - 1) / n)
    weights = weights / weights.sum()
    return torch.dot(x, weights[sorted_indices.argsort()])


def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        # mse_map = torch.mean((fs-ft)**2, dim=1)
        # a_map = mse_map
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N


def denormalize(img):
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    x = (((img.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def anomaly_map_to_jet_rgb(ano_map, target_size=None, vmin=None, vmax=None):
    """Map an anomaly map to JET pseudo-color RGB (same as overlay heatmap mapping).

    Normalization:
    - If vmin/vmax are given: linear scale (ano_map - vmin) / (vmax - vmin) and clip to [0, 1].
      visualize_img should pass vmin=0, vmax=1 (anomaly map already normalized in evaluation_batch
      using global GT=0/GT=1 pixel means).
    - If not given: per-image min-max normalization (for other callers).
    """
    ano_map = np.asarray(ano_map, dtype=np.float32)
    if target_size is not None:
        w_img, h_img = target_size
        if ano_map.shape[0] != h_img or ano_map.shape[1] != w_img:
            ano_map = cv2.resize(ano_map, (w_img, h_img), interpolation=cv2.INTER_CUBIC)

    if vmin is not None and vmax is not None and vmax > vmin:
        ano_norm = np.clip((ano_map - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        ano_norm = (ano_map - ano_map.min()) / (ano_map.max() - ano_map.min() + 1e-8)

    ano_map_u8 = np.clip(ano_norm * 255.0, 0, 255).astype(np.uint8)
    ano_map_jet_bgr = cv2.applyColorMap(ano_map_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(ano_map_jet_bgr, cv2.COLOR_BGR2RGB)


def save_anomaly_jet_heatmap(ano_map, save_path, target_size=None, vmin=None, vmax=None):
    """Save a JET pseudo-color anomaly heatmap (without overlay on the original image)."""
    ano_map_jet_rgb = anomaly_map_to_jet_rgb(ano_map, target_size=target_size, vmin=vmin, vmax=vmax)
    plt.imsave(save_path, ano_map_jet_rgb)


def save_anomaly_overlay(cv2_img, ano_map, save_path, vmin=None, vmax=None, overlay_alpha=0.4):
    """Overlay a JET pseudo-color anomaly heatmap on the original image and save."""
    h_img, w_img = cv2_img.shape[:2]
    ano_map_jet_rgb = anomaly_map_to_jet_rgb(ano_map, target_size=(w_img, h_img), vmin=vmin, vmax=vmax)
    ano_overlay_rgb = cv2.addWeighted(ano_map_jet_rgb, overlay_alpha, cv2_img, 1 - overlay_alpha, 0)
    plt.imsave(save_path, ano_overlay_rgb)


def visualize_img_inp_former(imgs, anomaly_map, gt, save_root, img_path):
    batch_num = imgs.shape[0]
    for i in range(batch_num):
        img_path_list = img_path[i].replace('\\', '/').split('/')
        class_name, category, idx_name = img_path_list[-4], img_path_list[-2], img_path_list[-1].split('.')[0]

        if int(idx_name) < 500:  # Visualize at most 500 images per category
            os.makedirs(os.path.join(save_root, class_name, category), exist_ok=True)
            input_frame = denormalize(imgs[i].clone().squeeze(0).cpu().detach().numpy())
            cv2_input = np.array(input_frame, dtype=np.uint8)
            plt.imsave(os.path.join(save_root, class_name, category, fr'{idx_name}_0.png'), cv2_input)
            ano_map = anomaly_map[i].squeeze(0).cpu().detach().numpy()

            plt.imsave(os.path.join(save_root, class_name, category, fr'{idx_name}_1.png'), ano_map, cmap='viridis',
                       vmin=0, vmax=1)
            gt_map = gt[i].squeeze(0).cpu().detach().numpy()
            plt.imsave(os.path.join(save_root, class_name, category, fr'{idx_name}_2.png'), gt_map, cmap='gray')
            plt.close()


'''
Purpose: Visualize class-level anomaly score distribution histograms.
scores_gt0: Array of anomaly scores for GT=0 pixels
scores_gt1: Array of anomaly scores for GT=1 pixels
class_name: Class name
dataset_name: Dataset name
vis_path: Path to save visualization outputs
version_name: Version name
'''


def visualize_class_score_distribution(scores_gt0, scores_gt1, class_name, dataset_name, vis_path, version_name):
    '''
    Plot anomaly score distribution histograms.
    Args:
    scores_gt0: Normalized anomaly scores for GT=0 pixels (normalized = (score - pr_mean_of_good_px) / (pr_mean_of_abnormal_px - pr_mean_of_good_px))
    scores_gt1: Normalized anomaly scores for GT=1 pixels (normalized = (score - pr_mean_of_good_px) / (pr_mean_of_abnormal_px - pr_mean_of_good_px))
    class_name: Class name
    dataset_name: Dataset name
    vis_path: Path to save visualization outputs (may be None)
    version_name: Version name
    '''
    # Input scores are already normalized (normalized_anomaly_map)
    # Bin with fixed resolution 0.005
    if len(scores_gt0) > 0 and len(scores_gt1) > 0:
        all_scores = np.concatenate([scores_gt0, scores_gt1])
    elif len(scores_gt0) > 0:
        all_scores = scores_gt0
    elif len(scores_gt1) > 0:
        all_scores = scores_gt1
    else:
        # Skip visualization if there are no pixels
        return

    score_min = all_scores.min()
    score_max = all_scores.max()

    # Bin with fixed resolution 0.005
    resolution = 0.005
    # Bin range: floor score_min to 0.005 multiple, ceil score_max to 0.005 multiple
    bin_start = np.floor(score_min / resolution) * resolution
    bin_end = np.ceil(score_max / resolution) * resolution
    # Create bins: bin_start, bin_start+0.005, bin_start+0.010, ..., bin_end
    bins = np.arange(bin_start, bin_end + resolution, resolution)

    # Total pixel count per class (for normalization)
    total_gt0 = len(scores_gt0)
    total_gt1 = len(scores_gt1)
    
    # Compute histograms (raw frequencies)
    hist_gt0, bin_edges = np.histogram(scores_gt0, bins=bins)
    hist_gt1, _ = np.histogram(scores_gt1, bins=bins)
    
    # Normalize to within-class frequencies
    # Divide each bin count by total pixels in that class
    if total_gt0 > 0:
        hist_gt0_norm = hist_gt0.astype(float) / total_gt0
    else:
        hist_gt0_norm = hist_gt0.astype(float)

    if total_gt1 > 0:
        hist_gt1_norm = hist_gt1.astype(float) / total_gt1
    else:
        hist_gt1_norm = hist_gt1.astype(float)

    # Define masks from normalized frequencies (for area computation)
    # Note: overlap detection uses raw frequencies (hist_gt0 > 0); area uses normalized frequencies
    overlap_mask = (hist_gt0 > 0) & (hist_gt1 > 0)  # Overlapping bins (from raw frequencies)
    gt0_only_mask = (hist_gt0 > 0) & (hist_gt1 == 0)  # GT=0-only bins (blue)
    gt1_only_mask = (hist_gt1 > 0) & (hist_gt0 == 0)  # GT=1-only bins (red)

    # Compute areas for the three color regions (from normalized frequencies)
    # Bar height is normalized frequency, so area is based on normalized frequencies
    # Blue: GT=0-only bins + GT=0 excess in overlapping bins
    blue_area = hist_gt0_norm[gt0_only_mask].sum() if gt0_only_mask.any() else 0
    if overlap_mask.any():
        # GT=0 excess in overlap = GT=0 norm freq minus overlap (min of both)
        overlap_heights = np.minimum(hist_gt0_norm[overlap_mask], hist_gt1_norm[overlap_mask])
        blue_area += (hist_gt0_norm[overlap_mask] - overlap_heights).sum()  # GT=0 excess beyond overlap
    
    # Red: GT=1-only bins + GT=1 excess in overlapping bins
    red_area = hist_gt1_norm[gt1_only_mask].sum() if gt1_only_mask.any() else 0
    if overlap_mask.any():
        # GT=1 excess in overlap = GT=1 norm freq minus overlap (min of both)
        overlap_heights = np.minimum(hist_gt0_norm[overlap_mask], hist_gt1_norm[overlap_mask])
        red_area += (hist_gt1_norm[overlap_mask] - overlap_heights).sum()  # GT=1 excess beyond overlap
    
    # Purple (overlap): overlapping portion (min of normalized frequencies)
    if overlap_mask.any():
        purple_area = np.minimum(hist_gt0_norm[overlap_mask], hist_gt1_norm[overlap_mask]).sum()  # Overlap area
    else:
        purple_area = 0
    
    total_area = blue_area + red_area + purple_area  # Sum of three color areas (normalized)
    if total_area > 0:
        overlap_ratio = purple_area / total_area
    else:
        overlap_ratio = 0.0

    # Bin centers and width
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = resolution  # Fixed resolution 0.005
    bar_width = bin_width  # Bar width equals bin width (no gaps)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Overlap bar height (min of both, normalized frequencies)
    overlap_heights_norm = np.zeros_like(hist_gt0_norm)
    if overlap_mask.any():
        overlap_heights_norm[overlap_mask] = np.minimum(hist_gt0_norm[overlap_mask], hist_gt1_norm[overlap_mask])

    # Draw GT=0-only bins first (blue)
    label_gt0_added = False
    if gt0_only_mask.any():
        hist_gt0_only = np.zeros_like(hist_gt0_norm)
        hist_gt0_only[gt0_only_mask] = hist_gt0_norm[gt0_only_mask]
        ax.bar(bin_centers, hist_gt0_only, width=bar_width, alpha=0.7,
               color='blue', label='GT=0 (Normal)', align='center', zorder=1)
        label_gt0_added = True
    
    # Draw GT=1-only bins (red)
    label_gt1_added = False
    if gt1_only_mask.any():
        hist_gt1_only = np.zeros_like(hist_gt1_norm)
        hist_gt1_only[gt1_only_mask] = hist_gt1_norm[gt1_only_mask]
        ax.bar(bin_centers, hist_gt1_only, width=bar_width, alpha=0.7,
               color='red', label='GT=1 (Anomaly)', align='center', zorder=1)
        label_gt1_added = True
    
    # For overlapping bins, draw full blue and red bars first
    if overlap_mask.any():
        # GT=0 bars in overlapping bins (blue)
        hist_gt0_overlap = np.zeros_like(hist_gt0_norm)
        hist_gt0_overlap[overlap_mask] = hist_gt0_norm[overlap_mask]
        if hist_gt0_overlap.sum() > 0:
            ax.bar(bin_centers, hist_gt0_overlap, width=bar_width, alpha=0.7,
                   color='blue', label='GT=0 (Normal)' if not label_gt0_added else '', 
                   align='center', zorder=1)
        
        # GT=1 bars in overlapping bins (red)
        hist_gt1_overlap = np.zeros_like(hist_gt1_norm)
        hist_gt1_overlap[overlap_mask] = hist_gt1_norm[overlap_mask]
        if hist_gt1_overlap.sum() > 0:
            ax.bar(bin_centers, hist_gt1_overlap, width=bar_width, alpha=0.7,
                   color='red', label='GT=1 (Anomaly)' if not label_gt1_added else '', 
                   align='center', zorder=1)
        
        # Overlay purple on overlap height (from bottom)
        overlap_bin_centers = bin_centers[overlap_mask]
        overlap_heights = overlap_heights_norm[overlap_mask]
        ax.bar(overlap_bin_centers, overlap_heights, width=bar_width,
               bottom=0, alpha=0.6, color='purple', label='Overlap', align='center', zorder=2)

    # Save path: visualize_analysis/score_distribution/version_name/dataset_name/class_name/
    # visualize_analysis should be sibling to visualize_results
    # vis_path is usually visualize_results/version_name or visualize_save_dir/version_name
    # Parent of visualize_results (sibling to visualize_analysis)
    if vis_path is not None:
        vis_results_dir = os.path.dirname(vis_path)  # visualize_results or visualize_save_dir
        base_dir = os.path.dirname(vis_results_dir)  # Parent of visualize_results
    else:
        # If vis_path is None, use current working directory
        base_dir = os.getcwd()
    save_dir = os.path.join(base_dir, 'visualize_analysis', 'score_distribution', version_name, dataset_name,
                            class_name)
    os.makedirs(save_dir, exist_ok=True)

    # Normalized mean scores
    mean_gt0 = scores_gt0.mean() if len(scores_gt0) > 0 else 0.0
    mean_gt1 = scores_gt1.mean() if len(scores_gt1) > 0 else 0.0

    # Three plots: 1) GT=0 only, 2) GT=1 only, 3) full (with overlap)
    
    # ===== Plot 1: GT=0 only histogram =====
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    if gt0_only_mask.any():
        hist_gt0_only = np.zeros_like(hist_gt0_norm)
        hist_gt0_only[gt0_only_mask] = hist_gt0_norm[gt0_only_mask]
        ax1.bar(bin_centers, hist_gt0_only, width=bar_width, alpha=0.7,
               color='blue', label='GT=0 (Normal)', align='center', zorder=1)
    if overlap_mask.any():
        hist_gt0_overlap = np.zeros_like(hist_gt0_norm)
        hist_gt0_overlap[overlap_mask] = hist_gt0_norm[overlap_mask]
        if hist_gt0_overlap.sum() > 0:
            ax1.bar(bin_centers, hist_gt0_overlap, width=bar_width, alpha=0.7,
                   color='blue', align='center', zorder=1)
    ax1.set_xlabel('Normalized Anomaly Score', fontsize=12)
    ax1.set_ylabel('Normalized Frequency (within class)', fontsize=12)
    ax1.set_title(f'Score Distribution (GT=0 only): {class_name}', fontsize=14)
    ax1.set_ylim([0, 0.01])  # Fixed y-axis range 0~0.01
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    mean_text = f'Mean: {mean_gt0:.4f}'
    ax1.text(0.02, 0.98, mean_text, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    save_path1 = os.path.join(save_dir, 'score_distribution_gt0_only.png')
    plt.savefig(save_path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # ===== Plot 2: GT=1 only histogram =====
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    if gt1_only_mask.any():
        hist_gt1_only = np.zeros_like(hist_gt1_norm)
        hist_gt1_only[gt1_only_mask] = hist_gt1_norm[gt1_only_mask]
        ax2.bar(bin_centers, hist_gt1_only, width=bar_width, alpha=0.7,
               color='red', label='GT=1 (Anomaly)', align='center', zorder=1)
    if overlap_mask.any():
        hist_gt1_overlap = np.zeros_like(hist_gt1_norm)
        hist_gt1_overlap[overlap_mask] = hist_gt1_norm[overlap_mask]
        if hist_gt1_overlap.sum() > 0:
            ax2.bar(bin_centers, hist_gt1_overlap, width=bar_width, alpha=0.7,
                   color='red', align='center', zorder=1)
    ax2.set_xlabel('Normalized Anomaly Score', fontsize=12)
    ax2.set_ylabel('Normalized Frequency (within class)', fontsize=12)
    ax2.set_title(f'Score Distribution (GT=1 only): {class_name}', fontsize=14)
    ax2.set_ylim([0, 0.01])  # Fixed y-axis range 0~0.01
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    mean_text = f'Mean: {mean_gt1:.4f}'
    ax2.text(0.02, 0.98, mean_text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    save_path2 = os.path.join(save_dir, 'score_distribution_gt1_only.png')
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # ===== Plot 3: Full histogram (with overlap) =====
    # Labels and title
    ax.set_xlabel('Normalized Anomaly Score', fontsize=12)
    ax.set_ylabel('Normalized Frequency (within class)', fontsize=12)
    ax.set_title(f'Score Distribution: {class_name}', fontsize=14)
    ax.set_ylim([0, 0.01])  # Fixed y-axis range 0~0.01
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Show overlap ratio and mean scores on plot
    info_text = f'Overlap Ratio: {overlap_ratio:.4f} ({overlap_ratio*100:.2f}%)\nGT=0 Mean: {mean_gt0:.4f}\nGT=1 Mean: {mean_gt1:.4f}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save figure
    save_path = os.path.join(save_dir, 'score_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Overlap ratio: {overlap_ratio*100:.2f}%")


def visualize_pixel_roc_curve(gt_pixels, pr_pixels, vis_path, version_name, dataset_name=None, class_name=None):
    '''
    Plot pixel-level ROC curve.
    Args:
    gt_pixels: Pixel-level GT (flattened 1D array)
    pr_pixels: Pixel-level predictions (flattened 1D array)
    vis_path: Path to save visualization outputs (may be None)
    version_name: Version name
    dataset_name: Dataset name (optional; inferred from vis_path or default if None)
    class_name: Class name (optional; inferred from vis_path or default if None)
    '''
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(gt_pixels, pr_pixels)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Pixel-level ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Show AUC on plot
    info_text = f'AUC = {roc_auc:.4f}'
    plt.text(0.6, 0.2, info_text, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save path: visualize_analysis/pixel_roc/version_name/dataset_name/class_name/ (same layout as score_distribution)
    if vis_path is not None:
        vis_results_dir = os.path.dirname(vis_path)  # visualize_results or visualize_save_dir
        base_dir = os.path.dirname(vis_results_dir)  # Parent of visualize_results
    else:
        # If vis_path is None, use current working directory
        base_dir = os.getcwd()
    
    # If dataset_name and class_name are None, try to extract from vis_path
    if dataset_name is None or class_name is None:
        if vis_path is not None:
            # Try to extract from vis_path (if path format includes this info)
            vis_path_parts = vis_path.replace('\\', '/').split('/')
            # Typical path: visualize_results/version_name or visualize_save_dir/version_name
            # Cannot extract dataset_name/class_name directly from vis_path; get from elsewhere
            pass
    
    # Default if still None
    if dataset_name is None:
        dataset_name = 'default'
    if class_name is None:
        class_name = 'default'
    
    save_dir = os.path.join(base_dir, 'visualize_analysis', 'pixel_roc', version_name, dataset_name, class_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save figure
    save_path = os.path.join(save_dir, 'pixel_roc_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pixel-level ROC curve saved to: {save_path}")


'''
Purpose: Model evaluation.
Args:
model: Inference model
dataloader: Test data dataloader
device: CPU & GPU
max_ratio: Fraction of highest-scoring pixels used for image-level anomaly score
resize_mask: Resize anomaly map and labels to this size
vis_path: Path to save visualization outputs
vis_img_num: Max images to visualize per class
GPU_accelerate: Use GPU acceleration for metric computation
compute_metrics: If False, skip metrics and return None
visualize_score_distribution: If False, skip score distribution histograms
compute_per_class_auroc: If True, compute and print per-class pixel AUROC
visualize_pixel_roc: If True, plot pixel-level ROC and save under visualize_analysis
'''


def evaluation_batch(model, dataloader, device, max_ratio=0, resize_mask=None, vis_path=None, vis_img_num=200,
                     GPU_accelerate=True, compute_metrics=True, visualize_score_distribution=True, compute_per_class_auroc=False, visualize_pixel_roc=False, save_name=None, enable_clustering=False, n_clusters=None,
                     apply_fg_prob_map: bool = False):
    # ===== Initialization
    model.eval()
    # ---------- GT and predictions for quantitative metrics
    gt_list_px = []  # Pixel-level GT list
    pr_list_px = []  # Pixel-level prediction list
    gt_list_sp = []  # Image-level GT list
    pr_list_sp = []  # Image-level prediction list
    # ---------- Data for visualization
    img_list = []
    seg_map_list = []
    anomaly_map_list = []
    gt_list = []
    img_path_list = []
    cluster_assignments_list = []  # Cluster assignment results
    # ---------- Per-class anomaly scores for distribution plots
    # Only init when needed to avoid memory blow-up
    if visualize_score_distribution:
        class_scores_dict = defaultdict(lambda: {'gt0': [], 'gt1': []})  # Scores by class
    else:
        class_scores_dict = None
    class_dataset_dict = {}  # Dataset name per class
    # ---------- Per-image class in each batch (for per-class AUROC)
    if compute_per_class_auroc:
        batch_class_info = []  # Class name per image per batch
    gaussian_kernel = get_gaussian_kernel(kernel_size=9, sigma=7).to(device)  # Gaussian blur kernel
    # ===== Inference
    with torch.no_grad():
        has_seg_output = False
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            # ---------- Forward pass
            img = img.to(device)
            output = model(img)

            cluster_assignments = None
            if len(output) == 3:  # No segmentation output
                en, de = output[0], output[1]  # Encoder features, decoder features
            elif len(output) == 5:  # Segmentation output (clustering disabled)
                has_seg_output = True
                en, de, mask_logits, class_logits = output[0], output[1], output[3][-1], output[4][
                    -1]  # Encoder, decoder, mask/class logits; -1 = last layer
            elif len(output) == 6:  # Segmentation and clustering output
                has_seg_output = True
                en, de, mask_logits, class_logits, cluster_assignments = output[0], output[1], output[3][-1], output[4][
                    -1], output[5]  # Encoder, decoder, mask/class logits, cluster assignments
            # ---------- Anomaly map
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            if resize_mask is not None:  # Resize anomaly map and GT
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            anomaly_map = gaussian_kernel(anomaly_map)  # Blur anomaly map

            # ---------- Foreground probability map from segmentation
            if has_seg_output:
                pixel_logits = to_per_pixel_logits_semantic(mask_logits,
                                                            class_logits)  # (B, 2, 148, 148) per-pixel fg/bg logits
                seg_map = logit_to_fg_prob(pixel_logits)  # (B, 148, 148) foreground probability
                seg_map = seg_map.unsqueeze(1)  # (B, 1, 148, 148)
                seg_map = F.interpolate(  # (B, 1, 148, 148) -> (B, 1, 512, 512)
                    seg_map,
                    size=anomaly_map.shape[-2:],
                    mode='bilinear',  # Bilinear interpolation
                    align_corners=False
                )
                if apply_fg_prob_map:
                    anomaly_map = anomaly_map * seg_map

            # ---------- Pack predictions and GT
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            
            # Accumulate only when metrics, per-class AUROC, or ROC viz needed (stay on GPU until end)
            if compute_metrics or compute_per_class_auroc or visualize_pixel_roc:
                gt_list_px.append(gt)  # Pixel GT (on GPU)
                pr_list_px.append(anomaly_map)  # Anomaly map (on GPU)
                if compute_metrics:
                    gt_list_sp.append(label)  # Image GT (metrics only)

            # ---------- Defect type per image in batch (for per-defect-type AUROC)
            # Note: defect type (missing, broken, etc.), not dataset class (catenary_dropper)
            if compute_per_class_auroc:
                batch_size = img.shape[0]
                batch_defect_types = []
                for i in range(batch_size):
                    img_path_item = img_path[i].replace('\\', '/').split('/')
                    if len(img_path_item) >= 3:
                        defect_type = img_path_item[-2]  # Defect type (missing, broken, good, etc.)
                        batch_defect_types.append(defect_type)
                    else:
                        batch_defect_types.append(None)
                batch_class_info.append(batch_defect_types)

            # Visualization-related storage (when needed)
            # Also collect paths when visualize_pixel_roc or visualize_score_distribution
            if vis_path is not None or visualize_pixel_roc or visualize_score_distribution:
                # Extract dataset_name and class_name from paths (ROC / distribution save paths)
                batch_size = img.shape[0]
                for i in range(batch_size):
                    img_path_list_item = img_path[i].replace('\\', '/').split('/')
                    if len(img_path_list_item) >= 5:
                        dataset_name = img_path_list_item[-5]
                        class_name = img_path_list_item[-4]  # Class name
                        
                        # Map class to dataset (for ROC / distribution paths)
                        if class_name not in class_dataset_dict:
                            class_dataset_dict[class_name] = dataset_name
                
                # Full batch data only when saving images
                if vis_path is not None:
                    anomaly_map_cpu = anomaly_map.cpu()
                    gt_cpu = gt.cpu()
                    img_cpu = img.cpu()

                    img_list.append(img_cpu)  # One element per batch
                    seg_map_list.append(seg_map.cpu()) if has_seg_output else seg_map_list.append(None)
                    anomaly_map_list.append(anomaly_map_cpu)
                    gt_list.append(gt_cpu)
                    img_path_list.append(img_path)
                    # Cluster results
                    if cluster_assignments is not None:
                        # List of per-batch cluster results
                        if isinstance(cluster_assignments, list):
                            cluster_assignments_list.append([ca.cpu() if isinstance(ca, torch.Tensor) else ca for ca in cluster_assignments])
                        else:
                            cluster_assignments_list.append(cluster_assignments.cpu() if isinstance(cluster_assignments, torch.Tensor) else cluster_assignments)
                    else:
                        cluster_assignments_list.append(None)
                else:
                    # vis_path None but distribution plot: CPU transfer for scores
                    if visualize_score_distribution:
                        anomaly_map_cpu = anomaly_map.cpu()
                        gt_cpu = gt.cpu()
                
                # ---------- Collect scores in loop (distribution plot; avoid storing all data)
                # Raw scores here; normalized scores used later for visualization
                if visualize_score_distribution:
                    for i in range(batch_size):
                        img_path_list_item = img_path[i].replace('\\', '/').split('/')
                        if len(img_path_list_item) >= 5:
                            class_name = img_path_list_item[-4]  # Class name
                            
                            # Current image anomaly map and GT
                            if vis_path is not None:
                                # Already on CPU
                                ano_map = anomaly_map_cpu[i].squeeze(0).numpy()  # (H, W)
                                gt_map = gt_cpu[i].squeeze(0).numpy()  # (H, W)
                            else:
                                # Temporary CPU transfer for scores
                                ano_map = anomaly_map[i].squeeze(0).cpu().numpy()  # (H, W)
                                gt_map = gt[i].squeeze(0).cpu().numpy()  # (H, W)
                            
                            # Flatten and split gt=0 vs gt=1 pixel scores
                            ano_flat = ano_map.flatten()
                            gt_flat = gt_map.flatten()
                            
                            # Per-class collection (only when distribution viz enabled)
                            if class_scores_dict is not None:
                                class_scores_dict[class_name]['gt0'].extend(ano_flat[gt_flat == 0])
                                class_scores_dict[class_name]['gt1'].extend(ano_flat[gt_flat == 1])

            # ---------- Image-level anomaly score (metrics only)
            if compute_metrics:
                if max_ratio == 0:
                    sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
                else:
                    anomaly_map_flat = anomaly_map.flatten(1)
                    sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][:, :int(anomaly_map_flat.shape[1] * max_ratio)]
                    sp_score = sp_score.mean(dim=1)
                pr_list_sp.append(sp_score)

        # ===== Quantitative metrics
        if compute_metrics:
            gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
            pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
            gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
            pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

            print("Calculating evaluation metrics...")

            if GPU_accelerate:  # GPU-accelerated metrics
                auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp,
                                                                                          gt_list_px, gt_list_sp)
                # Flattened data for ROC curve
                gt_list_px_flat = gt_list_px.ravel() if gt_list_px.ndim > 1 else gt_list_px
                pr_list_px_flat = pr_list_px.ravel() if pr_list_px.ndim > 1 else pr_list_px
            else:  # CPU-only metrics
                aupro_px = compute_pro(gt_list_px, pr_list_px)
                gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()
                auroc_px = roc_auc_score(gt_list_px, pr_list_px)
                auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
                ap_px = average_precision_score(gt_list_px, pr_list_px)
                ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
                f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
                f1_px = f1_score_max(gt_list_px, pr_list_px)
                # Data already flattened on CPU path
                gt_list_px_flat = gt_list_px
                pr_list_px_flat = pr_list_px
            
            # ---------- Pixel-level ROC curve
            if visualize_pixel_roc:
                # Version name from vis_path, or save_name if vis_path is None
                if vis_path is not None:
                    version_name = os.path.basename(vis_path.rstrip('/\\'))  # Version name
                    if not version_name:  # vis_path ends with /; use parent dir name
                        version_name = os.path.basename(os.path.dirname(vis_path.rstrip('/\\')))
                else:
                    version_name = save_name if save_name is not None else 'default'  # save_name as version when no vis_path
                
                # dataset_name and class_name from class_dataset_dict or img_path_list
                dataset_name = None
                class_name = None
                if class_dataset_dict and len(class_dataset_dict) > 0:
                    # First class and its dataset from class_dataset_dict
                    first_class = list(class_dataset_dict.keys())[0]
                    class_name = first_class
                    dataset_name = class_dataset_dict[first_class]
                elif img_path_list and len(img_path_list) > 0 and len(img_path_list[0]) > 0:
                    # From first image path
                    first_img_path = img_path_list[0][0].replace('\\', '/').split('/')
                    if len(first_img_path) >= 5:
                        dataset_name = first_img_path[-5]
                        class_name = first_img_path[-4]
                
                visualize_pixel_roc_curve(gt_list_px_flat, pr_list_px_flat, vis_path, version_name, dataset_name, class_name)
        else:
            # Skip metric computation
            # Still need data if compute_per_class_auroc or visualize_pixel_roc
            if compute_per_class_auroc or visualize_pixel_roc:
                gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
                pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
                gt_list_px_flat = gt_list_px.ravel() if gt_list_px.ndim > 1 else gt_list_px
                pr_list_px_flat = pr_list_px.ravel() if pr_list_px.ndim > 1 else pr_list_px
            else:
                # Release memory when not needed
                gt_list_px = None
                pr_list_px = None
            auroc_sp = ap_sp = f1_sp = auroc_px = ap_px = f1_px = aupro_px = None
            
            # ---------- Pixel-level ROC (even when metrics disabled)
            if visualize_pixel_roc:
                # Version name from vis_path, or save_name if vis_path is None
                if vis_path is not None:
                    version_name = os.path.basename(vis_path.rstrip('/\\'))  # Version name
                    if not version_name:  # vis_path ends with /; use parent dir name
                        version_name = os.path.basename(os.path.dirname(vis_path.rstrip('/\\')))
                else:
                    version_name = save_name if save_name is not None else 'default'  # save_name as version when no vis_path
                
                # dataset_name and class_name from class_dataset_dict or img_path_list
                dataset_name = None
                class_name = None
                if class_dataset_dict and len(class_dataset_dict) > 0:
                    # First class and its dataset from class_dataset_dict
                    first_class = list(class_dataset_dict.keys())[0]
                    class_name = first_class
                    dataset_name = class_dataset_dict[first_class]
                elif img_path_list and len(img_path_list) > 0 and len(img_path_list[0]) > 0:
                    # From first image path
                    first_img_path = img_path_list[0][0].replace('\\', '/').split('/')
                    if len(first_img_path) >= 5:
                        dataset_name = first_img_path[-5]
                        class_name = first_img_path[-4]
                
                visualize_pixel_roc_curve(gt_list_px_flat, pr_list_px_flat, vis_path, version_name, dataset_name, class_name)

        # ---------- Per-defect-type pixel AUROC (same method as global metrics)
        # Defect types (missing, broken, etc.); excludes good
        if compute_per_class_auroc and len(batch_class_info) > 0 and gt_list_px is not None:
            print("\nComputing per-defect-type pixel-level AUROC:")
            
            # Unique defect types (exclude good)
            all_defect_types = set()
            for batch_defect_types in batch_class_info:
                all_defect_types.update([dt for dt in batch_defect_types if dt is not None and dt != 'good'])
            
            if len(all_defect_types) == 0:
                print("  No defect types found (good class excluded)")
            else:
                # Image size (assume uniform)
                if gt_list_px.ndim == 3:  # (N, H, W)
                    h, w = gt_list_px.shape[1], gt_list_px.shape[2]
                    pixels_per_image = h * w
                else:
                    pixels_per_image = None
                
                # Process each defect type; extract from numpy arrays
                # Check shape: may be ravel()'d already
                is_3d_shape = (gt_list_px.ndim == 3 and pr_list_px.ndim == 3)
                
                # 1D flattened data: use flat arrays
                if not is_3d_shape:
                    gt_list_px_flat = gt_list_px.ravel() if gt_list_px.ndim > 1 else gt_list_px
                    pr_list_px_flat = pr_list_px.ravel() if pr_list_px.ndim > 1 else pr_list_px
                
                for defect_type in sorted(all_defect_types):
                    # Image indices for this defect type
                    defect_image_indices = []
                    image_idx = 0
                    for batch_idx, batch_defect_types in enumerate(batch_class_info):
                        batch_size = len(batch_defect_types)
                        for i in range(batch_size):
                            if batch_defect_types[i] == defect_type:
                                defect_image_indices.append(image_idx)
                            image_idx += 1
                    
                    # Extract data if images exist for this defect type
                    if len(defect_image_indices) > 0 and pixels_per_image is not None:
                        # Extract and flatten for CPU metrics
                        if is_3d_shape:
                            # 3D arrays: index and ravel
                            gt_defect_images = gt_list_px[defect_image_indices]  # (M, H, W) M = images for this defect
                            pr_defect_images = pr_list_px[defect_image_indices]  # (M, H, W)
                            gt_pixels = gt_defect_images.ravel()
                            pr_pixels = pr_defect_images.ravel()
                            del gt_defect_images, pr_defect_images
                        else:
                            # 1D flattened: slice from flat arrays
                            gt_pixels_list = []
                            pr_pixels_list = []
                            for img_idx in defect_image_indices:
                                start_idx = img_idx * pixels_per_image
                                end_idx = start_idx + pixels_per_image
                                gt_pixels_list.append(gt_list_px_flat[start_idx:end_idx])
                                pr_pixels_list.append(pr_list_px_flat[start_idx:end_idx])
                            
                            # Concatenate
                            gt_pixels = np.concatenate(gt_pixels_list)
                            pr_pixels = np.concatenate(pr_pixels_list)
                            del gt_pixels_list, pr_pixels_list
                        
                        # CPU metrics (same as global evaluation)
                        if len(gt_pixels) > 0 and len(np.unique(gt_pixels)) > 1:
                            defect_auroc = roc_auc_score(gt_pixels, pr_pixels)
                            # Print only
                            print(f"  Defect type [{defect_type}]: pixel-level AUROC = {defect_auroc:.4f} (GT=0: {np.sum(gt_pixels == 0)} pixels, GT=1: {np.sum(gt_pixels == 1)} pixels)")
                        else:
                            print(f"  Defect type [{defect_type}]: skipped (insufficient data or no positive/negative samples)")
                        
                        # Free defect-type data
                        del gt_pixels, pr_pixels
                    
                    # Free index list
                    del defect_image_indices
                
                # Free flat array refs if created
                if not is_3d_shape and 'gt_list_px_flat' in locals():
                    del gt_list_px_flat, pr_list_px_flat
            
            print()

        # Visualization =====
        # Anomaly map normalization params (for distribution plots)
        if visualize_score_distribution or vis_path is not None:
            if gt_list_px is not None and pr_list_px is not None:
                pr_mean_of_good_px = pr_list_px[gt_list_px == 0].mean()  # Mean prediction for GT=0 pixels
                pr_mean_of_abnormal_px = pr_list_px[gt_list_px == 1].mean()  # Mean prediction for GT=1 pixels
            elif vis_path is not None and len(anomaly_map_list) > 0 and len(gt_list) > 0:
                # Recompute from anomaly_map_list and gt_list if gt_list_px/pr_list_px unavailable
                all_pr_good = []
                all_pr_abnormal = []
                for anomaly_map, gt in zip(anomaly_map_list, gt_list):
                    ano_np = anomaly_map[0].cpu().numpy() if isinstance(anomaly_map, torch.Tensor) else anomaly_map[0]
                    gt_np = gt[0].cpu().numpy() if isinstance(gt, torch.Tensor) else gt[0]
                    all_pr_good.extend(ano_np[gt_np == 0].flatten())
                    all_pr_abnormal.extend(ano_np[gt_np == 1].flatten())
                pr_mean_of_good_px = np.mean(all_pr_good) if len(all_pr_good) > 0 else 0.0
                pr_mean_of_abnormal_px = np.mean(all_pr_abnormal) if len(all_pr_abnormal) > 0 else 1.0
            else:
                # Default fallback
                pr_mean_of_good_px = 0.0
                pr_mean_of_abnormal_px = 1.0
        
        if vis_path is not None:
            print("Visualizing results...")
            # ---------- Version name from vis_path
            version_name = os.path.basename(vis_path.rstrip('/\\'))  # Version name
            if not version_name:  # vis_path ends with /; use parent dir name
                version_name = os.path.basename(os.path.dirname(vis_path.rstrip('/\\')))
            # ---------- Normalize anomaly maps and save visualizations
            if pr_mean_of_abnormal_px > pr_mean_of_good_px:  # Expected: abnormal mean > normal mean
                # Only when image visualization is enabled
                if vis_img_num > 0 and len(img_list) > 0:
                    class_vis_count = defaultdict(int)  # Per-class vis count

                    for batch_idx, (img, seg_map, anomaly_map, gt, img_path) in enumerate(zip(img_list, seg_map_list, anomaly_map_list, gt_list,
                                                                       img_path_list)):
                        defect_type = img_path[0].replace('\\', '/').split('/')[-2]
                        if class_vis_count[defect_type] < vis_img_num:  # Under per-class vis limit

                            normalized_anomaly_map = (anomaly_map - pr_mean_of_good_px) / (
                                        pr_mean_of_abnormal_px - pr_mean_of_good_px)  # Map normal pixels to 0, abnormal to 1
                            cluster_assignments_batch = cluster_assignments_list[batch_idx] if batch_idx < len(cluster_assignments_list) else None
                            visualize_img(imgs=img, segmentation_map=seg_map, anomaly_map=normalized_anomaly_map, gt=gt,
                                          img_path=img_path, save_root=vis_path, cluster_assignments=cluster_assignments_batch, n_clusters=n_clusters)

                            class_vis_count[defect_type] += img.shape[0]
                            print(f"Class [{defect_type}] visualization progress: {class_vis_count[defect_type]}/{vis_img_num}")

                # ---------- Per-class score distribution histograms
                if visualize_score_distribution and class_scores_dict is not None:
                    total_classes = len(class_scores_dict)
                    if total_classes > 0:
                        print(f"\nGenerating anomaly score distribution bar charts ({total_classes} classes)...")
                        # Use save_name as version when vis_path is None
                        if vis_path is None:
                            version_name = save_name if save_name is not None else 'default'
                        for idx, (class_name, scores) in enumerate(class_scores_dict.items(), 1):
                            dataset_name = class_dataset_dict[class_name]
                            scores_gt0_raw = np.array(scores['gt0'])
                            scores_gt1_raw = np.array(scores['gt1'])
                            
                            # Normalize: GT=0 mean -> 0, GT=1 mean -> 1
                            # Raw means for GT=0 and GT=1
                            mean_gt0_raw = scores_gt0_raw.mean() if len(scores_gt0_raw) > 0 else 0.0
                            mean_gt1_raw = scores_gt1_raw.mean() if len(scores_gt1_raw) > 0 else 0.0
                            norm_denominator = mean_gt1_raw - mean_gt0_raw
                            if norm_denominator > 0:
                                # GT=0 mean -> 0, GT=1 mean -> 1
                                scores_gt0 = (scores_gt0_raw - mean_gt0_raw) / norm_denominator
                                scores_gt1 = (scores_gt1_raw - mean_gt0_raw) / norm_denominator
                            else:
                                # Zero/negative denominator: use raw scores
                                scores_gt0 = scores_gt0_raw
                                scores_gt1 = scores_gt1_raw
                            
                            # Print normalized means
                            mean_gt0 = scores_gt0.mean() if len(scores_gt0) > 0 else 0.0
                            mean_gt1 = scores_gt1.mean() if len(scores_gt1) > 0 else 0.0
                            print(f"  [{idx}/{total_classes}] Class: {class_name} (GT=0: {len(scores_gt0)} pixels, GT=1: {len(scores_gt1)} pixels)")
                            print(f"      Normalized mean score: GT=0={mean_gt0:.4f}, GT=1={mean_gt1:.4f}")
                            visualize_class_score_distribution(
                                scores_gt0=scores_gt0,
                                scores_gt1=scores_gt1,
                                class_name=class_name,
                                dataset_name=dataset_name,
                                vis_path=vis_path,
                                version_name=version_name
                            )
                        print(f"Done! Generated anomaly score distribution plots for {total_classes} classes\n")

                # Free visualization data and GPU memory
                del img_list, seg_map_list, anomaly_map_list, gt_list, img_path_list
                if class_scores_dict is not None:
                    del class_scores_dict
                del class_dataset_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            else:
                print("Model prediction anomaly: mean anomaly score for anomalous pixels <= normal pixels")
                import sys
                sys.exit(1)
        
        # ---------- Per-class distribution histograms (even when vis_path is None)
        if visualize_score_distribution and class_scores_dict is not None:
            total_classes = len(class_scores_dict)
            if total_classes > 0:
                print(f"\nGenerating anomaly score distribution bar charts ({total_classes} classes)...")
                # Use save_name as version when vis_path is None
                if vis_path is None:
                    version_name = save_name if save_name is not None else 'default'
                for idx, (class_name, scores) in enumerate(class_scores_dict.items(), 1):
                    dataset_name = class_dataset_dict[class_name]
                    scores_gt0_raw = np.array(scores['gt0'])
                    scores_gt1_raw = np.array(scores['gt1'])
                    
                    # Normalize: GT=0 mean -> 0, GT=1 mean -> 1
                    # Raw means for GT=0 and GT=1
                    mean_gt0_raw = scores_gt0_raw.mean() if len(scores_gt0_raw) > 0 else 0.0
                    mean_gt1_raw = scores_gt1_raw.mean() if len(scores_gt1_raw) > 0 else 0.0
                    norm_denominator = mean_gt1_raw - mean_gt0_raw
                    if norm_denominator > 0:
                        # GT=0 mean -> 0, GT=1 mean -> 1
                        scores_gt0 = (scores_gt0_raw - mean_gt0_raw) / norm_denominator
                        scores_gt1 = (scores_gt1_raw - mean_gt0_raw) / norm_denominator
                    else:
                        # Zero/negative denominator: use raw scores
                        scores_gt0 = scores_gt0_raw
                        scores_gt1 = scores_gt1_raw
                    
                    # Print normalized means
                    mean_gt0 = scores_gt0.mean() if len(scores_gt0) > 0 else 0.0
                    mean_gt1 = scores_gt1.mean() if len(scores_gt1) > 0 else 0.0
                    print(f"  [{idx}/{total_classes}] Class: {class_name} (GT=0: {len(scores_gt0)} pixels, GT=1: {len(scores_gt1)} pixels)")
                    print(f"      Normalized mean score: GT=0={mean_gt0:.4f}, GT=1={mean_gt1:.4f}")
                    visualize_class_score_distribution(
                        scores_gt0=scores_gt0,
                        scores_gt1=scores_gt1,
                        class_name=class_name,
                        dataset_name=dataset_name,
                        vis_path=vis_path,
                        version_name=version_name
                    )
                print(f"Done! Generated anomaly score distribution plots for {total_classes} classes\n")

    # Return evaluation metrics
    if compute_metrics:
        return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]
    else:
        return None


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

'''
base_value: Base LR after warmup; start of cosine annealing
final_value: Final LR at end of training; end of cosine annealing
total_iters: Total training iterations
warmup_iters: Warmup iterations (iter, not epoch)
start_warmup_value: LR at start of warmup
Warmup: first warmup_iters steps, LR linearly from start_warmup_value to base_value
Cosine annealing: remaining total_iters - warmup_iters steps, LR cosine decay from base_value to final_value
'''


class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]


# Combine mask and class predictions into per-pixel class probabilities
def to_per_pixel_logits_semantic(mask_logits, class_logits):
    return torch.einsum(
        "bqhw, bqc -> bchw",  # (B, 2, 148, 148), (B, 2, 2) -> (B, 2, 148, 148) per-pixel class probabilities
        mask_logits.sigmoid(),
        class_logits.softmax(dim=-1)[..., :-1])


# Per-pixel class logits -> 2D foreground probability map; fg_prob = fg_logit / (fg_logit + bg_logit)
def logit_to_fg_prob(logit):  # (B, 2, H, W) -> # (B, H, W)
    # Sum over channel dim, keep batch dim
    channel_sum = logit[:, 0] + logit[:, 1]
    # Avoid division by zero; same device as logit
    channel_sum = torch.where(channel_sum == 0, torch.tensor(1.0, device=logit.device), channel_sum)
    # Foreground probability
    return logit[:, 1] / channel_sum


def visualize_seg(logit, img_path, save_dir):
    """
    Visualize segmentation as a heatmap.
    """
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    # Linear scale so channels sum to 1; use second channel ratio
    channel_sum = logit[0] + logit[1]
    channel_sum = torch.where(channel_sum == 0, torch.tensor(1.0), channel_sum)
    prob_channel1 = logit[1] / channel_sum

    # Map to 0-255
    heatmap = (prob_channel1.cpu().numpy() * 255).astype(np.uint8)

    # Resize heatmap
    h, w = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    # Path metadata
    path_obj = Path(img_path)
    img_name = path_obj.stem
    defect_type = path_obj.parent.name

    # Save heatmap
    os.makedirs(os.path.join(save_dir, defect_type), exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, defect_type, f"{img_name}.png"), heatmap_resized)

    return heatmap_resized


'''
Purpose: Visualize images.
imgs: Batch of images
seg_map: Segmentation probability maps for imgs
anomaly_map: Anomaly heatmaps for imgs
gt: Binary GT masks for imgs
img_path: Paths for imgs
save_root: Root directory for visualization outputs
'''


def visualize_img(imgs, segmentation_map, anomaly_map, gt, img_path, save_root, cluster_assignments=None, n_clusters=None):
    batch_size = imgs.shape[0]

    for i in range(batch_size):

        img_path_list = img_path[i].replace('\\', '/').split('/')  # Split img_path into components
        dataset_name, class_name, defect_type, img_file_name = (
            img_path_list[-5], img_path_list[-4], img_path_list[-2],
            img_path_list[-1].split('.')[0])  # Dataset, class, defect type, filename
        save_dir = os.path.join(save_root, dataset_name, class_name, defect_type)  # Output directory
        os.makedirs(save_dir, exist_ok=True)
        # ---------- Render and save
        img = denormalize(imgs[i].clone().squeeze(0).cpu().detach().numpy())
        cv2_img = np.array(img, dtype=np.uint8)  # Original image
        plt.imsave(os.path.join(save_dir, fr'{img_file_name}.png'), cv2_img)

        gt_map = gt[i].squeeze(0).cpu().detach().numpy()  # GT mask
        plt.imsave(os.path.join(save_dir, fr'{img_file_name}_gt.png'), gt_map, cmap='gray')

        ano_map = anomaly_map[i].squeeze(0).cpu().detach().numpy()  # Anomaly map (normalized in evaluation_batch via global GT=0/GT=1 means)
        h_img, w_img = cv2_img.shape[:2]
        save_anomaly_jet_heatmap(
            ano_map,
            os.path.join(save_dir, fr'{img_file_name}_ano.png'),
            target_size=(w_img, h_img),
            vmin=0,
            vmax=1,
        )

        save_anomaly_overlay(
            cv2_img, ano_map,
            os.path.join(save_dir, fr'{img_file_name}_ano_overlay.png'),
            vmin=0,
            vmax=1,
        )

        if segmentation_map is not None:
            seg_map = segmentation_map[i].squeeze(0).cpu().detach().numpy()  # Segmentation map
            plt.imsave(os.path.join(save_dir, fr'{img_file_name}_seg.png'), seg_map, cmap='gray', vmin=0, vmax=1)

        # ---------- Clustering visualization
        if cluster_assignments is not None:
            # cluster_assignments may be a list (batch) or a tensor
            # If list, take element i; otherwise use directly
            if isinstance(cluster_assignments, list):
                cluster_assignment_single = cluster_assignments[i] if i < len(cluster_assignments) else None
            else:
                cluster_assignment_single = cluster_assignments
            if cluster_assignment_single is not None:
                # img is (H, W, C); img.shape[:2] is (H, W)
                img_size = img.shape[:2]  # (H, W)
                visualize_clustering_result(cluster_assignment_single, img_file_name, save_dir, img_size, n_clusters=n_clusters)

        plt.close()


def visualize_clustering_result(cluster_assignments, img_file_name, save_dir, img_size, n_clusters=None):
    """
    Visualize clustering results.
    Args:
    cluster_assignments: Cluster map (H, W), values 0~n_clusters-1 (every pixel assigned to a cluster)
    img_file_name: Image filename without extension
    save_dir: Save directory
    img_size: Image size (H, W)
    n_clusters: Expected cluster count (if None, use actual unique clusters)
    """
    if cluster_assignments is None:
        return
    
    # Convert to numpy
    if isinstance(cluster_assignments, torch.Tensor):
        cluster_map = cluster_assignments.cpu().numpy()
    else:
        cluster_map = cluster_assignments
    
    # Handle list or batch data
    # cluster_assignments may be a list (one map per image) or tensor (H, W)
    if isinstance(cluster_map, list) and len(cluster_map) > 0:
        # List: take first element
        cluster_map = cluster_map[0]
        if isinstance(cluster_map, torch.Tensor):
            cluster_map = cluster_map.cpu().numpy()
    
    # cluster_map must be 2D (H, W)
    if cluster_map.ndim != 2:
        raise ValueError(f"cluster_map should be 2D (H, W), but got shape {cluster_map.shape} (ndim={cluster_map.ndim})")
    
    # Resize to image size
    # img_size is (H, W); cv2.resize expects (W, H)
    # cv2.resize size is (width, height) = (W, H)
    target_size = (int(img_size[1]), int(img_size[0]))  # (W, H)
    if cluster_map.shape != img_size:
        # INTER_NEAREST preserves discrete cluster labels
        cluster_map_resized = cv2.resize(cluster_map.astype(np.float32), target_size, 
                                         interpolation=cv2.INTER_NEAREST)
        cluster_map = cluster_map_resized.astype(np.int64)
    
    # Unique cluster IDs (use unique, not max; some IDs may be missing)
    unique_clusters = np.unique(cluster_map)
    n_clusters_actual = len(unique_clusters)
    
    # Use n_clusters if given; else actual count
    if n_clusters is not None:
        n_clusters_used = n_clusters
        # Color all clusters 0..n_clusters-1 even if some IDs missing
        cluster_ids_to_color = list(range(n_clusters))
    else:
        n_clusters_used = n_clusters_actual
        cluster_ids_to_color = unique_clusters.tolist()
    
    # Color image: every pixel belongs to a cluster (no background)
    colored_map = np.zeros((cluster_map.shape[0], cluster_map.shape[1], 3), dtype=np.uint8)
    
    # Assign color per cluster (HSV colormap)
    # Generate n_clusters distinct colors in HSV
    for idx, cluster_id in enumerate(cluster_ids_to_color):
        # Uniform hue in HSV
        # Split 180° into n_clusters steps; hue at 0, step, 2*step, ...
        # n_clusters=4 -> hue=[0, 45, 90, 135]
        hue_step = 180 / n_clusters_used  # Degrees per step
        hue = int(hue_step * idx)  # Start at 0, step by hue_step
        saturation = 255  # Max saturation
        value = 255  # Max value
        
        # HSV to RGB
        color_hsv = np.uint8([[[hue, saturation, value]]])
        color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0]
        
        # Color pixels in this cluster
        mask = (cluster_map == cluster_id)
        colored_map[mask] = color_rgb
    
    # Save with cv2.imwrite; shape (H, W, 3); cv2 expects BGR but we have RGB
    # Convert to BGR or use RGB (cv2.IMWRITE_PNG_COLOR)
    save_path = os.path.join(save_dir, f'{img_file_name}_cluster.png')
    
    # Verify colored_map shape
    if colored_map.ndim != 3 or colored_map.shape[2] != 3:
        raise ValueError(f"colored_map should be (H, W, 3), but got shape {colored_map.shape}")
    
    # cv2.imwrite expects BGR; convert from RGB
    colored_map_bgr = cv2.cvtColor(colored_map, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(save_path, colored_map_bgr)
    if not success:
        raise RuntimeError(f"Failed to save image to {save_path}")