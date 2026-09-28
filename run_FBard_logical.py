import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
from torch.cuda.amp import autocast, GradScaler
import argparse
from sklearn.cluster import KMeans
from copy import deepcopy
import random
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_dilation, generate_binary_structure

# Dataset-Related Modules
from ad_dataset import ADDataset, get_data_transforms
from seg_dataset import SegTrainDataset, train_collate
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Model-Related Modules
from models import vit_encoder
from models.FBard_logical import FBard_logical, CompositionAutoEncoder_CNN, CompositionUNet_CNN, CompositionAutoEncoder_Light_v4, CompositionUNet_Light_v4
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

# Training-Related Modules
from optimizers import StableAdamW
from utils import evaluation_batch, setup_seed, get_logger, WarmCosineScheduler, global_cosine_hm_adaptive, visualize_clustering_result, to_per_pixel_logits_semantic, logit_to_fg_prob, denormalize, save_anomaly_overlay, save_anomaly_jet_heatmap
from mask_classification_loss import MaskClassificationLoss
from torchvision.ops.focal_loss import sigmoid_focal_loss

warnings.filterwarnings("ignore")  # Suppress all Python warnings


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    v = str(v).strip().lower()
    if v in {"true", "1", "yes", "y", "t"}:
        return True
    if v in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


# ===== Composition Branch Loss Functions =====
class DiceLoss(nn.Module):
    """Dice Loss for multi-class segmentation"""
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight)
        self.smooth = 1e-5

    def forward(self, predict, target):
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1)
        target = target.view(N, 1, -1)
        predict = F.softmax(predict, dim=1)
        target_onehot = torch.zeros(predict.size(), device=predict.device, dtype=predict.dtype)
        target_onehot.scatter_(1, target.long(), 1)
        intersection = torch.sum(predict * target_onehot, dim=2)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)
        if hasattr(self, 'weight'):
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
            dice_coef = dice_coef * self.weight * C
        dice_loss = 1 - torch.mean(dice_coef)
        return dice_loss


class MultiClassFocalLoss(nn.Module):
    """Multi-class Focal Loss implementation"""
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        if inputs.dim() == 4:
            N, C, H, W = inputs.shape
            inputs = inputs.view(N, C, -1)
            if targets.dim() == 4:
                targets = targets.view(N, C, -1)
            elif targets.dim() == 3:
                targets = targets.view(N, -1)
                targets_one_hot = torch.zeros(N, C, H*W, device=inputs.device, dtype=inputs.dtype)
                targets_one_hot.scatter_(1, targets.unsqueeze(1).long(), 1)
                targets = targets_one_hot
        else:
            if targets.dim() == 1:
                N, C = inputs.shape
                targets_one_hot = torch.zeros(N, C, device=inputs.device, dtype=inputs.dtype)
                targets_one_hot.scatter_(1, targets.unsqueeze(1).long(), 1)
                targets = targets_one_hot
        
        probs = F.softmax(inputs, dim=1)
        p_t = (probs * targets).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        log_probs = F.log_softmax(inputs, dim=1)
        ce_loss = -(targets * log_probs).sum(dim=1)
        
        if self.alpha is not None:
            if self.alpha.dim() == 1:
                alpha_t = (self.alpha.unsqueeze(0).unsqueeze(-1) * targets).sum(dim=1)
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ===== Anomaly Generation Strategy 2 (37×37 adapted) =====
def get_connected_components_37x37(mask):
    """Extract connected components (8-connectivity, 10-pixel threshold, adapted for 37×37)."""
    components_list = []
    components_ci = []
    for c in range(mask.shape[0]):
        mask_numpy = mask[c].cpu().numpy() if isinstance(mask, torch.Tensor) else mask[c]
        structure = generate_binary_structure(rank=2, connectivity=2)  # 8-connectivity
        labeled_mask, num_features = label(mask_numpy, structure=structure)
        for i in range(1, num_features + 1):
            component = labeled_mask == i
            if component.sum() > 10:  # 10-pixel threshold
                components_list.append(component)
                components_ci.append(c)
    return components_list, components_ci


def change_label_same_img_feat(seg_mask, diff_seg_mask, background_label):
    """Strategy 2: change labels within the same image (37×37), excluding background label."""
    # Ensure seg_mask is a tensor and get device info
    if isinstance(seg_mask, torch.Tensor):
        device = seg_mask.device
        seg_mask = seg_mask.clone()  # Clone to avoid modifying the original tensor
    else:
        device = torch.device('cpu')
        seg_mask = torch.from_numpy(seg_mask).float()
    
    H, W = seg_mask.shape[1], seg_mask.shape[2]
    mask = torch.zeros((H, W), device=device)
    recon_mask = torch.zeros((H, W), device=device)
    anom_seg = seg_mask.clone()
    
    # Record original labels
    original_cluster = onehot_to_cluster(anom_seg.cpu().numpy())
    
    component_list, ci_list = get_connected_components_37x37(anom_seg)
    # Filter out connected components belonging to background label
    filtered_components = []
    filtered_cis = []
    for comp, ci in zip(component_list, ci_list):
        if ci != background_label:
            filtered_components.append(comp)
            filtered_cis.append(ci)
    
    if len(filtered_components) == 0:
        return mask, recon_mask, anom_seg
    
    i = torch.randint(0, len(filtered_components), (1,)).item()
    connected_component, ci_orig = filtered_components[i], filtered_cis[i]
    
    # Convert numpy boolean array to torch tensor on the correct device
    if isinstance(connected_component, np.ndarray):
        connected_component = torch.from_numpy(connected_component).bool().to(device)
    else:
        connected_component = connected_component.to(device)
    
    recon_mask[connected_component] = 1
    
    # Find neighboring region of connected component (via dilation)
    temp_mask = torch.zeros((H, W), device=device)
    temp_mask[connected_component] = 1
    temp_mask_np = temp_mask.cpu().numpy()
    dilated_mask = binary_dilation(temp_mask_np, structure=[[1,1,1], [1,1,1], [1,1,1]])
    dilated_mask = torch.from_numpy(dilated_mask).bool().to(device)
    # Exclude original component region; keep only neighbors
    dilated_mask[connected_component] = False
    
    # Find labels present in neighbor region (background label may replace other labels)
    neighbor_channels = torch.where((anom_seg[:, dilated_mask]).sum(dim=1) > 0)[0]
    if len(neighbor_channels) > 0:
        # If neighbors have labels, randomly pick one as new label (including background)
        ci_new = neighbor_channels[torch.randint(0, len(neighbor_channels), (1,)).item()].item()
    else:
        # If no neighbor labels, randomly pick a label (not original, may include background)
        available_labels = [c for c in range(anom_seg.shape[0]) if c != ci_orig]
        if len(available_labels) > 0:
            ci_new = available_labels[torch.randint(0, len(available_labels), (1,)).item()]
        else:
            ci_new = ci_orig  # If only one label exists, keep unchanged
    
    # Update anom_seg: zero all channels at connected_component pixels, then set new label channel to 1
    anom_seg[:, connected_component] = 0
    anom_seg[ci_new, connected_component] = 1
    
    # Compute mask: pixels where labels changed
    new_cluster = onehot_to_cluster(anom_seg.cpu().numpy())
    mask = torch.from_numpy((original_cluster != new_cluster).astype(np.float32)).to(device)
    
    return mask, recon_mask, anom_seg


def cluster_to_onehot(cluster_map, n_clusters):
    """Convert cluster map to one-hot encoding."""
    if isinstance(cluster_map, torch.Tensor):
        cluster_map = cluster_map.cpu().numpy()
    H, W = cluster_map.shape
    onehot = np.zeros((n_clusters, H, W), dtype=np.float32)
    for c in range(n_clusters):
        onehot[c] = (cluster_map == c).astype(np.float32)
    return torch.from_numpy(onehot).float()


def onehot_to_cluster(onehot):
    """Convert one-hot encoding to cluster map."""
    if isinstance(onehot, torch.Tensor):
        onehot = onehot.cpu().numpy()
    return np.argmax(onehot, axis=0).astype(np.int64)


def find_background_label(cluster_map):
    """
    Find background label: the label with the most boundary pixels
    (pixels where h=0 or H-1, or w=0 or W-1).
    Returns the background label index.
    """
    H, W = cluster_map.shape
    # Collect boundary pixels
    boundary_pixels = []
    boundary_pixels.extend(cluster_map[0, :].flatten())  # Top boundary
    boundary_pixels.extend(cluster_map[H-1, :].flatten())  # Bottom boundary
    boundary_pixels.extend(cluster_map[:, 0].flatten())  # Left boundary
    boundary_pixels.extend(cluster_map[:, W-1].flatten())  # Right boundary
    # Deduplicate (avoid double-counting corner pixels)
    boundary_pixels = np.array(boundary_pixels)
    
    # Count occurrences of each label on boundary pixels
    unique_labels, counts = np.unique(boundary_pixels, return_counts=True)
    # Pick the label with the highest count
    background_label = unique_labels[np.argmax(counts)]
    return int(background_label)


# ===== Anomaly Generation Strategy 3 (copy label cluster from another image) =====
def copy_label_from_other_image(seg_mask, diff_seg_mask, background_label):
    """
    Strategy 3: copy a label cluster from another image to the current image.
    Only copies connected components above the threshold, not all pixels of the label.
    seg_mask: composition map of the current image (n_clusters, H, W)
    diff_seg_mask: composition map of the other image (n_clusters, H, W)
    background_label: background label, never copied
    Returns: (mask, recon_mask, anom_seg)
    """
    if isinstance(seg_mask, torch.Tensor):
        device = seg_mask.device
        seg_mask = seg_mask.clone()
        diff_seg_mask = diff_seg_mask.clone()
    else:
        device = torch.device('cpu')
        seg_mask = torch.from_numpy(seg_mask).float()
        diff_seg_mask = torch.from_numpy(diff_seg_mask).float()
    
    H, W = seg_mask.shape[1], seg_mask.shape[2]
    mask = torch.zeros((H, W), device=device)
    recon_mask = torch.zeros((H, W), device=device)
    anom_seg = seg_mask.clone()
    
    # Record original labels
    original_cluster = onehot_to_cluster(anom_seg.cpu().numpy())
    
    # Get labels in diff_seg_mask (exclude background; background is never copied)
    available_labels = [c for c in range(diff_seg_mask.shape[0]) if c != background_label]
    if len(available_labels) == 0:
        return mask, recon_mask, anom_seg
    
    # Randomly select a label to copy from the other image
    ci_copy = available_labels[torch.randint(0, len(available_labels), (1,)).item()]
    
    # Get pixel locations of that label in diff_seg_mask
    label_mask = diff_seg_mask[ci_copy] > 0.5  # (H, W)
    if not label_mask.any():
        return mask, recon_mask, anom_seg
    
    # Connected-component analysis on label; keep only components above threshold
    label_mask_np = label_mask.cpu().numpy() if isinstance(label_mask, torch.Tensor) else label_mask
    structure = generate_binary_structure(rank=2, connectivity=2)  # 8-connectivity
    labeled_mask, num_features = label(label_mask_np, structure=structure)
    
    # Collect connected components above threshold
    valid_components = []
    for i in range(1, num_features + 1):
        component = labeled_mask == i
        if component.sum() > 10:  # 10-pixel threshold
            valid_components.append(component)
    
    if len(valid_components) == 0:
        return mask, recon_mask, anom_seg
    
    # Randomly select one connected component to copy
    selected_component = valid_components[torch.randint(0, len(valid_components), (1,)).item()]
    copy_mask = torch.from_numpy(selected_component).bool().to(device)  # (H, W)
    
    # Copy pixels at copy_mask into current image (background label may be overwritten)
    # Zero out pixels at these locations first
    anom_seg[:, copy_mask] = 0
    # Then assign the new label
    anom_seg[ci_copy, copy_mask] = 1
    
    # Compute mask: pixels where labels changed
    new_cluster = onehot_to_cluster(anom_seg.cpu().numpy())
    mask = torch.from_numpy((original_cluster != new_cluster).astype(np.float32)).to(device)
    recon_mask[copy_mask] = 1
    
    return mask, recon_mask, anom_seg


# ===== Anomaly Generation Strategy 4 (combine strategies 2 and 3) =====
def combine_strategy_2_and_3(seg_mask, diff_seg_mask, background_label):
    """
    Strategy 4: combine strategies 2 and 3.
    First apply strategy 2 (change labels within the same image), then strategy 3 (copy label from another image).
    seg_mask: composition map of the current image (n_clusters, H, W)
    diff_seg_mask: composition map of the other image (n_clusters, H, W)
    background_label: background label, never modified or copied
    Returns: (mask, recon_mask, anom_seg)
    """
    if isinstance(seg_mask, torch.Tensor):
        device = seg_mask.device
        seg_mask = seg_mask.clone()
        diff_seg_mask = diff_seg_mask.clone()
    else:
        device = torch.device('cpu')
        seg_mask = torch.from_numpy(seg_mask).float()
        diff_seg_mask = torch.from_numpy(diff_seg_mask).float()
    
    H, W = seg_mask.shape[1], seg_mask.shape[2]
    mask = torch.zeros((H, W), device=device)
    recon_mask = torch.zeros((H, W), device=device)
    anom_seg = seg_mask.clone()
    
    # Record original labels
    original_cluster = onehot_to_cluster(anom_seg.cpu().numpy())
    
    # Apply strategy 2: change labels within same image (exclude background)
    component_list, ci_list = get_connected_components_37x37(anom_seg)
    if len(component_list) > 0:
        # Filter out connected components belonging to background label
        filtered_components = []
        filtered_cis = []
        for comp, ci in zip(component_list, ci_list):
            if ci != background_label:
                filtered_components.append(comp)
                filtered_cis.append(ci)
        
        if len(filtered_components) > 0:
            i = torch.randint(0, len(filtered_components), (1,)).item()
            connected_component, ci_orig = filtered_components[i], filtered_cis[i]
            
            if isinstance(connected_component, np.ndarray):
                connected_component = torch.from_numpy(connected_component).bool().to(device)
            else:
                connected_component = connected_component.to(device)
            
            recon_mask[connected_component] = 1
            
            # Find neighboring region of connected component
            temp_mask = torch.zeros((H, W), device=device)
            temp_mask[connected_component] = 1
            temp_mask_np = temp_mask.cpu().numpy()
            dilated_mask = binary_dilation(temp_mask_np, structure=[[1,1,1], [1,1,1], [1,1,1]])
            dilated_mask = torch.from_numpy(dilated_mask).bool().to(device)
            dilated_mask[connected_component] = False
            
            # Find labels present in neighbor region (background label may replace other labels)
            neighbor_channels = torch.where((anom_seg[:, dilated_mask]).sum(dim=1) > 0)[0]
            if len(neighbor_channels) > 0:
                ci_new = neighbor_channels[torch.randint(0, len(neighbor_channels), (1,)).item()].item()
            else:
                available_labels = [c for c in range(anom_seg.shape[0]) if c != ci_orig]
                if len(available_labels) > 0:
                    ci_new = available_labels[torch.randint(0, len(available_labels), (1,)).item()]
                else:
                    ci_new = ci_orig
            
            anom_seg[:, connected_component] = 0
            anom_seg[ci_new, connected_component] = 1
    
    # Apply strategy 3: copy label from other image (exclude background; thresholded components only)
    available_labels = [c for c in range(diff_seg_mask.shape[0]) if c != background_label]
    if len(available_labels) > 0:
        ci_copy = available_labels[torch.randint(0, len(available_labels), (1,)).item()]
        # Get pixel locations of that label
        label_mask = diff_seg_mask[ci_copy] > 0.5
        if label_mask.any():
            # Connected-component analysis on label; keep only components above threshold
            label_mask_np = label_mask.cpu().numpy() if isinstance(label_mask, torch.Tensor) else label_mask
            structure = generate_binary_structure(rank=2, connectivity=2)  # 8-connectivity
            labeled_mask, num_features = label(label_mask_np, structure=structure)
            
            # Collect connected components above threshold
            valid_components = []
            for i in range(1, num_features + 1):
                component = labeled_mask == i
                if component.sum() > 10:  # 10-pixel threshold
                    valid_components.append(component)
            
            if len(valid_components) > 0:
                # Randomly select one connected component to copy
                selected_component = valid_components[torch.randint(0, len(valid_components), (1,)).item()]
                copy_mask = torch.from_numpy(selected_component).bool().to(device)
                anom_seg[:, copy_mask] = 0
                anom_seg[ci_copy, copy_mask] = 1
    
    # Compute mask: pixels where labels changed
    new_cluster = onehot_to_cluster(anom_seg.cpu().numpy())
    mask = torch.from_numpy((original_cluster != new_cluster).astype(np.float32)).to(device)
    
    return mask, recon_mask, anom_seg


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
            if n_clusters > 0 and hasattr(model, 'comp_ae') and hasattr(model, 'comp_unet') and model.comp_ae is not None:
                # Extract features for composition map generation
                en_list, feat_hw = model.extract_features_for_composition_map(img)
                
                # Generate composition map via k-means clustering
                if model.kmeans_centers is not None:
                    kmeans_centers_np = model.kmeans_centers.cpu().numpy() if isinstance(model.kmeans_centers, torch.Tensor) else model.kmeans_centers
                    
                    comp_anomaly_maps = []
                    for b in range(B):
                        # Cluster features for each image
                        x = model.fuse_feature(en_list)  # (B, N, C)
                        features = x[b].detach().cpu().numpy()  # (N, C)
                        
                        # Compute distance to cluster centers
                        distances = np.linalg.norm(features[:, None, :] - kmeans_centers_np[None, :, :], axis=2)  # (N, n_clusters)
                        labels = np.argmin(distances, axis=1)  # (N,)
                        
                        # Convert to one-hot composition map
                        feat_hw = int(np.sqrt(features.shape[0]))
                        cluster_map = labels.reshape(feat_hw, feat_hw)  # (37, 37)
                        seg_onehot = cluster_to_onehot(cluster_map, n_clusters)  # (n_clusters, 37, 37)
                        # cluster_to_onehot returns a Tensor; no further conversion needed
                        if isinstance(seg_onehot, torch.Tensor):
                            seg_onehot_tensor = seg_onehot.unsqueeze(0).to(device)  # (1, n_clusters, 37, 37)
                        else:
                            seg_onehot_tensor = torch.from_numpy(seg_onehot).float().unsqueeze(0).to(device)  # (1, n_clusters, 37, 37)
                        
                        # Pass through reconstruction and discriminator networks
                        seg_recon = model.comp_ae(seg_onehot_tensor).softmax(dim=1)  # (1, n_clusters, 37, 37)
                        unet_input = torch.cat([seg_onehot_tensor, seg_recon], dim=1)  # (1, n_clusters*2, 37, 37)
                        pred_mask = model.comp_unet(unet_input).squeeze(1)  # (1, 37, 37)
                        
                        # Resize to 512x512 and apply Gaussian blur
                        pred_mask_resized = F.interpolate(
                            pred_mask.unsqueeze(1), 
                            size=(512, 512), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(1)  # (1, 512, 512)
                        pred_mask_blurred = gaussian_kernel(pred_mask_resized.unsqueeze(1)).squeeze(1)  # (1, 512, 512)
                        
                        comp_anomaly_maps.append(pred_mask_blurred)
                    
                    if len(comp_anomaly_maps) > 0:
                        comp_anomaly_map_batch = torch.cat(comp_anomaly_maps, dim=0).unsqueeze(1)  # (B, 1, 512, 512)
                        # Collect pixel-level anomaly scores (unnormalized)
                        comp_pixel_scores.append(comp_anomaly_map_batch.cpu().flatten())
    
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
    
    # Compute normalized pixel-level scores (for post-normalization histograms)
    # Note: normalize each batch after mean/std are computed
    # All batches are collected, so normalize once at the end
    if len(ad_pixel_scores) > 0:
        ad_pixel_all = torch.cat(ad_pixel_scores).numpy()
        ad_pixel_all_norm = (ad_pixel_all - ad_mean) / ad_std
        ad_pixel_scores_norm = [ad_pixel_all_norm]  # Wrap as list for consistency
    else:
        ad_pixel_scores_norm = []
    if len(comp_pixel_scores) > 0:
        comp_pixel_all = torch.cat(comp_pixel_scores).numpy()
        comp_pixel_all_norm = (comp_pixel_all - comp_mean) / comp_std
        comp_pixel_scores_norm = [comp_pixel_all_norm]  # Wrap as list for consistency
    else:
        comp_pixel_scores_norm = []
    
    # Plot pixel-level anomaly score histograms
    if save_dir is not None and len(ad_pixel_scores) > 0:
        os.makedirs(save_dir, exist_ok=True)
        
        # Merge all pixel scores (pre-normalization) with downsampling to avoid OOM
        max_samples = 10_000_000
        ad_pixel_all = torch.cat(ad_pixel_scores).numpy()
        if len(ad_pixel_all) > max_samples:
            indices = np.random.choice(len(ad_pixel_all), size=max_samples, replace=False)
            ad_pixel_all = ad_pixel_all[indices]
        
        if len(comp_pixel_scores) > 0:
            comp_pixel_all = torch.cat(comp_pixel_scores).numpy()
            if len(comp_pixel_all) > max_samples:
                indices = np.random.choice(len(comp_pixel_all), size=max_samples, replace=False)
                comp_pixel_all = comp_pixel_all[indices]
        else:
            comp_pixel_all = np.array([])
        
        # Plot AD branch histogram (before normalization)
        plt.figure(figsize=(10, 6))
        plt.hist(ad_pixel_all, bins=100, alpha=0.7, edgecolor='black')
        plt.xlabel('Pixel-level Anomaly Score (AD Branch, Before Normalization)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Distribution of Pixel-level Anomaly Scores (AD Branch, Validation Set, Before Normalization)\nMean={np.mean(ad_pixel_all):.4f}, Std={np.std(ad_pixel_all):.4f}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'ad_branch_pixel_score_histogram_before_norm.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot Composition branch histogram (before normalization)
        if len(comp_pixel_all) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(comp_pixel_all, bins=100, alpha=0.7, edgecolor='black', color='orange')
            plt.xlabel('Pixel-level Anomaly Score (Composition Branch, Before Normalization)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'Distribution of Pixel-level Anomaly Scores (Composition Branch, Validation Set, Before Normalization)\nMean={np.mean(comp_pixel_all):.4f}, Std={np.std(comp_pixel_all):.4f}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'composition_branch_pixel_score_histogram_before_norm.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Plot post-normalization histograms
        if len(ad_pixel_scores_norm) > 0:
            # ad_pixel_scores_norm is now a list containing a single array
            ad_pixel_all_norm = ad_pixel_scores_norm[0] if isinstance(ad_pixel_scores_norm[0], np.ndarray) else np.concatenate(ad_pixel_scores_norm)
            # Downsample to avoid OOM
            if len(ad_pixel_all_norm) > max_samples:
                indices = np.random.choice(len(ad_pixel_all_norm), size=max_samples, replace=False)
                ad_pixel_all_norm = ad_pixel_all_norm[indices]
            plt.figure(figsize=(10, 6))
            plt.hist(ad_pixel_all_norm, bins=100, alpha=0.7, edgecolor='black', color='green')
            plt.xlabel('Pixel-level Anomaly Score (AD Branch, After Normalization)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'Distribution of Pixel-level Anomaly Scores (AD Branch, Validation Set, After Normalization)\nMean={np.mean(ad_pixel_all_norm):.4f}, Std={np.std(ad_pixel_all_norm):.4f}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'ad_branch_pixel_score_histogram_after_norm.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        if len(comp_pixel_scores_norm) > 0:
            # comp_pixel_scores_norm is now a list containing a single array
            comp_pixel_all_norm = comp_pixel_scores_norm[0] if isinstance(comp_pixel_scores_norm[0], np.ndarray) else np.concatenate(comp_pixel_scores_norm)
            # Downsample to avoid OOM
            if len(comp_pixel_all_norm) > max_samples:
                indices = np.random.choice(len(comp_pixel_all_norm), size=max_samples, replace=False)
                comp_pixel_all_norm = comp_pixel_all_norm[indices]
            plt.figure(figsize=(10, 6))
            plt.hist(comp_pixel_all_norm, bins=100, alpha=0.7, edgecolor='black', color='red')
            plt.xlabel('Pixel-level Anomaly Score (Composition Branch, After Normalization)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'Distribution of Pixel-level Anomaly Scores (Composition Branch, Validation Set, After Normalization)\nMean={np.mean(comp_pixel_all_norm):.4f}, Std={np.std(comp_pixel_all_norm):.4f}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'composition_branch_pixel_score_histogram_after_norm.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Histograms saved to {save_dir}")
    
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
            disc_output_batch = None  # Discriminator outputs (for visualization)
            cluster_map_batch = None  # Cluster maps (for visualization)
            recon_map_batch = None  # Reconstruction outputs (for visualization)
            if hasattr(model, 'comp_ae') and hasattr(model, 'comp_unet') and model.comp_ae is not None:
                # Extract features for composition map generation
                en_list, feat_hw = model.extract_features_for_composition_map(img)
                
                # Generate composition map via k-means clustering
                if model.kmeans_centers is not None:
                    kmeans_centers_np = model.kmeans_centers.cpu().numpy() if isinstance(model.kmeans_centers, torch.Tensor) else model.kmeans_centers
                    
                    comp_anomaly_maps = []
                    disc_outputs = []  # Store raw discriminator outputs (for visualization)
                    cluster_maps = []  # Store cluster maps (for visualization)
                    recon_maps = []  # Store reconstruction outputs (for visualization)
                    # Precompute fused features to avoid redundant work in the loop
                    x = model.fuse_feature(en_list)  # (B, N, C)
                    for b in range(B):
                        # Cluster features for each image
                        features = x[b].detach().cpu().numpy()  # (N, C)
                        
                        # Compute distance to cluster centers
                        distances = np.linalg.norm(features[:, None, :] - kmeans_centers_np[None, :, :], axis=2)  # (N, n_clusters)
                        labels = np.argmin(distances, axis=1)  # (N,)
                        
                        # Convert to one-hot composition map
                        feat_hw = int(np.sqrt(features.shape[0]))
                        cluster_map = labels.reshape(feat_hw, feat_hw)  # (37, 37)
                        cluster_maps.append(cluster_map)  # Store cluster map
                        seg_onehot = cluster_to_onehot(cluster_map, n_clusters)  # (n_clusters, 37, 37)
                        # cluster_to_onehot returns a Tensor; no further conversion needed
                        if isinstance(seg_onehot, torch.Tensor):
                            seg_onehot_tensor = seg_onehot.unsqueeze(0).to(device)  # (1, n_clusters, 37, 37)
                        else:
                            seg_onehot_tensor = torch.from_numpy(seg_onehot).float().unsqueeze(0).to(device)  # (1, n_clusters, 37, 37)
                        
                        # Pass through reconstruction and discriminator networks
                        seg_recon = model.comp_ae(seg_onehot_tensor).softmax(dim=1)  # (1, n_clusters, 37, 37)
                        recon_maps.append(seg_recon[0].cpu().numpy())  # Store reconstruction result (n_clusters, 37, 37)
                        unet_input = torch.cat([seg_onehot_tensor, seg_recon], dim=1)  # (1, n_clusters*2, 37, 37)
                        pred_mask = model.comp_unet(unet_input).squeeze(1)  # (1, 37, 37)
                        
                        # Store raw discriminator output (37×37, for visualization)
                        disc_outputs.append(pred_mask.detach().cpu())  # Move to CPU immediately
                        
                        # Resize to resize_mask and Gaussian blur (for anomaly detection)
                        pred_mask_resized = F.interpolate(
                            pred_mask.unsqueeze(1), 
                            size=(resize_mask, resize_mask) if resize_mask is not None else img.shape[-1], 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(1)  # (1, H, W)
                        pred_mask_blurred = gaussian_kernel(pred_mask_resized.unsqueeze(1)).squeeze(1)  # (1, H, W)
                        
                        comp_anomaly_maps.append(pred_mask_blurred.detach().cpu())  # Move to CPU immediately
                        
                        # Release intermediate variables immediately
                        del features, distances, labels, cluster_map, seg_onehot, seg_onehot_tensor
                        del seg_recon, unet_input, pred_mask, pred_mask_resized, pred_mask_blurred
                    
                    # Release variables no longer needed
                    del x, en_list
                    del kmeans_centers_np
                    
                    if len(comp_anomaly_maps) > 0:
                        comp_anomaly_map = torch.cat(comp_anomaly_maps, dim=0).unsqueeze(1)  # (B, 1, H, W)
                        comp_anomaly_map = comp_anomaly_map.to(device)  # Same device as anomaly_map_ad for fusion
                        disc_output_batch = torch.cat(disc_outputs, dim=0)  # (B, 37, 37) - raw 37×37 output
                        cluster_map_batch = cluster_maps  # List of (37, 37) arrays
                        recon_map_batch = recon_maps  # List of (n_clusters, 37, 37) arrays
                        # Release temporary lists
                        del comp_anomaly_maps, disc_outputs
                    else:
                        disc_output_batch = None
                        cluster_map_batch = None
                        recon_map_batch = None
            
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
    
    # ===== Visualization (complete all visualization before metrics)
    if vis_path is not None and not skip_visualization and len(img_list) > 0:
        print(f"Saving visualization results to {vis_path}...")
        
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
    
    # ===== Plot test-set pixel-level anomaly score histograms
    # Plot test histograms — only when visualization is enabled
    if not skip_visualization and len(test_ad_pixel_scores) > 0:
        print("Drawing test set histograms...")
        # Determine save directory
        if vis_path is not None:
            # Extract path info from vis_path
            vis_path_parts = vis_path.replace('\\', '/').split('/')
            if len(vis_path_parts) >= 2:
                dataset_name = vis_path_parts[-2] if len(vis_path_parts) >= 2 else 'unknown'
                save_dir = os.path.join('visualize_analysis', vis_path_parts[-1], dataset_name)
            else:
                save_dir = os.path.join('visualize_analysis', 'test_set')
        elif save_name is not None:
            save_dir = os.path.join('visualize_analysis', save_name, 'test_set')
        else:
            save_dir = os.path.join('visualize_analysis', 'test_set')
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"  Merging pixel scores... (AD: {len(test_ad_pixel_scores)} batches, Comp: {len(test_comp_pixel_scores)} batches)")
        # Merge all pixel-level scores (before normalization)
        import sys
        sys.stdout.flush()  # Flush stdout so output appears immediately
        
        # Merge and downsample to avoid OOM (max 10M pixels)
        max_samples = 10_000_000
        test_ad_pixel_all = torch.cat(test_ad_pixel_scores).numpy()
        print(f"  AD branch: {len(test_ad_pixel_all)} pixels (before sampling)")
        sys.stdout.flush()
        if len(test_ad_pixel_all) > max_samples:
            # Random downsampling
            indices = np.random.choice(len(test_ad_pixel_all), size=max_samples, replace=False)
            test_ad_pixel_all = test_ad_pixel_all[indices]
            print(f"  AD branch: {len(test_ad_pixel_all)} pixels (after sampling)")
            sys.stdout.flush()
        
        if len(test_comp_pixel_scores) > 0:
            test_comp_pixel_all = torch.cat(test_comp_pixel_scores).numpy()
            print(f"  Composition branch: {len(test_comp_pixel_all)} pixels (before sampling)")
            sys.stdout.flush()
            if len(test_comp_pixel_all) > max_samples:
                # Random downsampling
                indices = np.random.choice(len(test_comp_pixel_all), size=max_samples, replace=False)
                test_comp_pixel_all = test_comp_pixel_all[indices]
                print(f"  Composition branch: {len(test_comp_pixel_all)} pixels (after sampling)")
                sys.stdout.flush()
        else:
            test_comp_pixel_all = np.array([])
        
        # Plot test AD branch histogram (before normalization)
        print("  Drawing AD branch histogram (before normalization)...")
        plt.figure(figsize=(10, 6))
        plt.hist(test_ad_pixel_all, bins=100, alpha=0.7, edgecolor='black', color='blue')
        plt.xlabel('Pixel-level Anomaly Score (AD Branch, Before Normalization)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Distribution of Pixel-level Anomaly Scores (AD Branch, Test Set, Before Normalization)\nMean={np.mean(test_ad_pixel_all):.4f}, Std={np.std(test_ad_pixel_all):.4f}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'test_ad_branch_pixel_score_histogram_before_norm.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  AD branch histogram saved.")
        
        # Plot test Composition branch histogram (before normalization)
        if len(test_comp_pixel_all) > 0:
            print("  Drawing Composition branch histogram (before normalization)...")
            plt.figure(figsize=(10, 6))
            plt.hist(test_comp_pixel_all, bins=100, alpha=0.7, edgecolor='black', color='orange')
            plt.xlabel('Pixel-level Anomaly Score (Composition Branch, Before Normalization)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'Distribution of Pixel-level Anomaly Scores (Composition Branch, Test Set, Before Normalization)\nMean={np.mean(test_comp_pixel_all):.4f}, Std={np.std(test_comp_pixel_all):.4f}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'test_composition_branch_pixel_score_histogram_before_norm.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("  Composition branch histogram saved.")
        
        # Plot post-normalization histograms
        if len(test_ad_pixel_scores_norm) > 0:
            print("  Drawing AD branch histogram (after normalization)...")
            test_ad_pixel_all_norm = torch.cat(test_ad_pixel_scores_norm).numpy()
            # Downsample to avoid OOM
            if len(test_ad_pixel_all_norm) > max_samples:
                indices = np.random.choice(len(test_ad_pixel_all_norm), size=max_samples, replace=False)
                test_ad_pixel_all_norm = test_ad_pixel_all_norm[indices]
            plt.figure(figsize=(10, 6))
            plt.hist(test_ad_pixel_all_norm, bins=100, alpha=0.7, edgecolor='black', color='green')
            plt.xlabel('Pixel-level Anomaly Score (AD Branch, After Normalization)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'Distribution of Pixel-level Anomaly Scores (AD Branch, Test Set, After Normalization)\nMean={np.mean(test_ad_pixel_all_norm):.4f}, Std={np.std(test_ad_pixel_all_norm):.4f}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'test_ad_branch_pixel_score_histogram_after_norm.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("  AD branch histogram (after normalization) saved.")
        
        if len(test_comp_pixel_scores_norm) > 0:
            print("  Drawing Composition branch histogram (after normalization)...")
            test_comp_pixel_all_norm = torch.cat(test_comp_pixel_scores_norm).numpy()
            # Downsample to avoid OOM
            if len(test_comp_pixel_all_norm) > max_samples:
                indices = np.random.choice(len(test_comp_pixel_all_norm), size=max_samples, replace=False)
                test_comp_pixel_all_norm = test_comp_pixel_all_norm[indices]
            plt.figure(figsize=(10, 6))
            plt.hist(test_comp_pixel_all_norm, bins=100, alpha=0.7, edgecolor='black', color='red')
            plt.xlabel('Pixel-level Anomaly Score (Composition Branch, After Normalization)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'Distribution of Pixel-level Anomaly Scores (Composition Branch, Test Set, After Normalization)\nMean={np.mean(test_comp_pixel_all_norm):.4f}, Std={np.std(test_comp_pixel_all_norm):.4f}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'test_composition_branch_pixel_score_histogram_after_norm.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("  Composition branch histogram (after normalization) saved.")
        
        print(f"Test set histograms saved to {save_dir}")
        # Release histogram data and all intermediates
        del test_ad_pixel_scores, test_comp_pixel_scores, test_ad_pixel_scores_norm, test_comp_pixel_scores_norm
        # Release numpy arrays created during histogram plotting
        if 'test_ad_pixel_all' in locals():
            del test_ad_pixel_all
        if 'test_comp_pixel_all' in locals():
            del test_comp_pixel_all
        if 'test_ad_pixel_all_norm' in locals():
            del test_ad_pixel_all_norm
        if 'test_comp_pixel_all_norm' in locals():
            del test_comp_pixel_all_norm
        # Release histogram plotting intermediates
        if 'max_samples' in locals():
            del max_samples
        if 'indices' in locals():
            del indices
        if 'save_dir' in locals():
            del save_dir
        if 'vis_path_parts' in locals():
            del vis_path_parts
        if 'dataset_name' in locals():
            del dataset_name
        # Close matplotlib figures
        plt.close('all')
        import matplotlib
        matplotlib.pyplot.close('all')
        torch.cuda.empty_cache()
        import gc
        gc.collect()  # Force garbage collection
        print("Histogram data released.")
    
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


def to_device(data, device):  # Move nested structures without .to() to device
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    else:
        return data


def main(args):
    # ===== Set random seed
    setup_seed(42)
    # ===== Data preparation
    # ---------- Define paths
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
    # ---------- Encoder/decoder layers used for feature contrast
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

    # ---------- Define learnable INP prototypes (INP_num, embed_dim)=(6, 768)
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
    # use_bg_feature_aggregation = False if args.phase == "train" else True  # Background feature aggregation enabled only at test time (v0)
    use_bg_feature_aggregation = True  # Background feature aggregation enabled during training too (v1)
    
    # Load k-means cluster centers (test phase)
    kmeans_centers = None
    if args.phase == "test" and args.n_clusters > 0:
        dataset_name = args.dataset_path.split('/')[-1]
        kmeans_save_dir = os.path.join(args.save_dir, args.save_name, dataset_name, args.class_name)
        kmeans_path = os.path.join(kmeans_save_dir, 'kmeans_centers.npy')
        if os.path.exists(kmeans_path):
            kmeans_centers = np.load(kmeans_path)
            print_fn(f"Loaded k-means centers from {kmeans_path}")
        else:
            print_fn(f"Warning: k-means centers file not found at {kmeans_path}, clustering will be disabled")
            args.n_clusters = 0
    
    # Enable clustering collection only in AD training; disabled during seg training for speed
    enable_clustering_collection = False  # Off by default; enabled after segmentation training
    
    # Enable composition branch when clustering is on (train and test)
    use_composition_branch = (args.n_clusters > 0 and not args.disable_segmentation_branch)
    
    model = FBard_logical(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                        target_layers=target_layers, remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                        fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP, device=device, num_classes=2,
                        num_q=2, num_blocks=3, masked_attn_enabled=False,
                        update_bg_prototypes=update_bg_prototype,
                        use_segmentation_branch=use_segmentaion_branch,
                        use_bg_feature_aggregation=use_bg_feature_aggregation,
                        bg_feature_aggregation_binary=args.bg_feature_aggregation_binary,
                        fg_map_dilate_ksize=args.fg_map_dilate_ksize,
                        n_clusters=args.n_clusters,
                        kmeans_centers=kmeans_centers,
                        enable_clustering_collection=enable_clustering_collection,
                        use_composition_branch=use_composition_branch,
                        composition_network_type=args.composition_network_type,
                        post_fusion=args.post_fusion)
    model = model.to(device)
    
    # Set loaded cluster centers on model during test
    if kmeans_centers is not None:
        model.set_kmeans_centers(kmeans_centers)

    # ===== Training
    if args.phase == "train":
        # ---------- Check whether to skip pretraining
        dataset_name = args.dataset_path.split('/')[-1]
        save_dir = os.path.join(args.save_dir, args.save_name, dataset_name, args.class_name)
        weight_path = os.path.join(save_dir, args.weight_file_name)
        kmeans_path = os.path.join(save_dir, 'kmeans_centers.npy')
        
        skip_pretrain = args.skip_pretrain and os.path.exists(weight_path)
        
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
            
            # ----------- Set up optimizers and schedulers
            seg_lr = 1e-5
            seg_base_lr, seg_final_lr = seg_lr, seg_lr / 10
            seg_optimizer = StableAdamW([{'params': model.parameters()}],  # Parameters included in optimization
                                        lr=seg_lr, betas=(0.9, 0.999), weight_decay=0.05, amsgrad=True, eps=1e-10)
            seg_lr_scheduler = WarmCosineScheduler(seg_optimizer, base_value=seg_base_lr, final_value=seg_final_lr,
                                                   total_iters=args.seg_epochs * len(seg_train_dataloader),
                                                   warmup_iters=100)

            ad_trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
            ad_optimizer = StableAdamW([{'params': ad_trainable.parameters()}],  # Parameters included in optimization
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

                    if model.attn_mask_annealing_enabled:  # Compute mask probability if mask annealing is enabled
                        for i in range(model.num_blocks):
                            model.attn_mask_probs[i] = model.mask_annealing(
                                attn_mask_annealing_start_steps[i],
                                global_step,
                                attn_mask_annealing_end_steps[i],
                            )

                    with autocast(enabled=not args.disable_amp):
                        output = model(imgs)  # Forward pass
                        # Handle return values: 5 or 6 tensors when clustering is enabled
                        if len(output) == 6:
                            # Cluster centers returned (training with clustering enabled)
                            _, _, _, mask_logits_per_block, class_logits_per_block, _ = output
                        else:
                            # Clustering disabled; 5 return values
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
                    seg_scaler.scale(loss).backward()  # Backward pass
                    seg_scaler.unscale_(seg_optimizer)
                    nn.utils.clip_grad_norm(model.parameters(), max_norm=0.1)  # Gradient clipping (max norm)
                    seg_scaler.step(seg_optimizer)  # Optimizer step
                    seg_scaler.update()

                    loss_val = loss.item()  # Store current batch loss
                    loss_list.append(loss_val)
                    seg_lr_scheduler.step()  # Learning rate step

                    # Log batch loss with epoch and batch index
                    print_fn(
                        f'epoch [{epoch + 1}/{args.seg_epochs}], batch [{batch_idx + 1}/{len(seg_train_dataloader)}], loss: {loss_val:.4f}')

                    # # <<< GPU memory monitoring >>>
                    # allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
                    # reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
                    # print_fn(
                    #     f'epoch [{epoch + 1}/{args.seg_epochs}], batch [{batch_idx + 1}/{len(seg_train_dataloader)}], loss: {loss_val:.4f}, Allocated Mem: {allocated:.2f} MB, Reserved Mem: {reserved:.2f} MB')

                    global_step += 1

                del seg_optimizer  # Drop segmentation optimizer (no longer needed)
                del seg_lr_scheduler
                torch.cuda.empty_cache()  # Clear compute graph and free unused GPU memory
                if hasattr(model, "set_segmentation_requires_grad"):
                    model.set_segmentation_requires_grad(False)  # Seg branch gradients only needed in seg phase; disable in AD phase to save memory
                print("segmentation branch training completed")
                # After seg training, enable clustering collection for AD training
                if args.n_clusters > 0:
                    model.set_clustering_collection(True)
                    print_fn(f"Clustering collection enabled for anomaly detection training (n_clusters={args.n_clusters})")

            print("training anomaly detection branch...")
            # ---------- Anomaly detection training
            ad_scaler = GradScaler(enabled=not args.disable_amp)
            
            # ---------- Collect k-means cluster centers (if enabled)
            all_cluster_centers = [] if args.n_clusters > 0 else None
            if all_cluster_centers is not None:
                print_fn(f"Collecting cluster centers for k-means (n_clusters={args.n_clusters})...")
            
            for epoch in range(args.ad_epochs):
                model.train()
                for batch_idx, (img, _) in enumerate(tqdm(ad_train_dataloader, ncols=80)):
                    img = img.to(device)

                    with autocast(enabled=not args.disable_amp):
                        if not args.disable_segmentation_branch:  # Forward pass
                            output = model(img)
                            if len(output) == 6 and args.n_clusters > 0:
                                # Output includes cluster centers
                                en, de, g_loss, _, _, cluster_centers_batch = output
                                if all_cluster_centers is not None:
                                    all_cluster_centers.extend(cluster_centers_batch)
                            else:
                                en, de, g_loss = output[0], output[1], output[2]
                        else:
                            en, de, g_loss = model(img)
                        loss = global_cosine_hm_adaptive(en, de, y=3)  # Reconstruction loss
                        loss = loss + 0.2 * g_loss  # Feature contrast loss + λ * INP consistency loss

                    ad_optimizer.zero_grad(set_to_none=True)  # Zero gradients
                    ad_scaler.scale(loss).backward()  # Backward pass

                    ad_scaler.unscale_(ad_optimizer)
                    nn.utils.clip_grad_norm(ad_trainable.parameters(), max_norm=0.1)  # Gradient clipping (max norm)

                    ad_scaler.step(ad_optimizer)  # Optimizer step
                    ad_scaler.update()

                    ad_lr_scheduler.step()  # Learning rate step

                    # Log batch loss with epoch and batch index
                    print_fn(
                        f'epoch [{epoch + 1}/{args.ad_epochs}], batch [{batch_idx + 1}/{len(ad_train_dataloader)}], loss: {loss.item():.4f}')

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
        if args.n_clusters > 0 and not args.disable_segmentation_branch:
            print_fn("=" * 80)
            print_fn("Starting Composition Branch Training...")
            
            # ---------- Create dedicated visualization directory
            dataset_name = args.dataset_path.split('/')[-1]
            comp_vis_base_dir = os.path.join('visualize_composition_branch', args.save_name, dataset_name, args.class_name)
            os.makedirs(comp_vis_base_dir, exist_ok=True)
            
            # ---------- Save composition maps
            print_fn("Step 1: Generating and saving composition maps...")
            model.eval()
            composition_map_dir = os.path.join(comp_vis_base_dir, 'composition_maps')
            os.makedirs(composition_map_dir, exist_ok=True)
            
            composition_maps = []  # Store paths to all composition maps
            img_count = 0
            with torch.no_grad():
                for batch_idx, (img, _) in enumerate(tqdm(ad_train_dataloader, ncols=80, desc="Generating composition maps")):
                    img = img.to(device)
                    output = model(img)
                    if len(output) == 6:
                        cluster_assignments = output[5]  # Get cluster assignment results
                    else:
                        print_fn("Warning: No cluster assignments found, skipping composition branch training")
                        break
                    
                    for i, cluster_assignment in enumerate(cluster_assignments):
                        # Convert to one-hot encoding
                        onehot = cluster_to_onehot(cluster_assignment, args.n_clusters)
                        # Save as numpy file
                        np_path = os.path.join(composition_map_dir, f'composition_map_{img_count:05d}.npy')
                        np.save(np_path, onehot.numpy())
                        composition_maps.append(np_path)
                        
                        # Visualize every 100 maps to save time
                        if img_count % 100 == 0:
                            visualize_clustering_result(
                                cluster_assignment.cpu().numpy(),
                                f'composition_map_{img_count:05d}',
                                composition_map_dir,
                                (518, 518),  # Image size
                                n_clusters=args.n_clusters
                            )
                        img_count += 1
            
            print_fn(f"Saved {len(composition_maps)} composition maps to {composition_map_dir}")
            
            # Determine global background label: sample maps, vote per-image background labels, take majority
            global_background_label = None
            if len(composition_maps) > 0:
                print_fn("Determining global background label...")
                # Randomly sample composition maps (10% or at least 10, max 100)
                sample_size = min(max(10, len(composition_maps) // 10), 100, len(composition_maps))
                sampled_indices = random.sample(range(len(composition_maps)), sample_size)
                
                background_label_votes = {}
                for idx in sampled_indices:
                    seg_sample = np.load(composition_maps[idx])  # (n_clusters, 37, 37)
                    seg_cluster = onehot_to_cluster(seg_sample)
                    bg_label = find_background_label(seg_cluster)
                    background_label_votes[bg_label] = background_label_votes.get(bg_label, 0) + 1
                
                # Use label with most votes as global background
                if background_label_votes:
                    global_background_label = max(background_label_votes.items(), key=lambda x: x[1])[0]
                    print_fn(f"Global background label determined: {global_background_label} (votes: {background_label_votes})")
                else:
                    # Default to label 0 if no votes
                    global_background_label = 0
                    print_fn(f"Warning: No background label votes, using default: {global_background_label}")
            
            # ---------- Train composition branch
            print_fn("Step 2: Training composition branch...")
            
            # Ensure composition branch is initialized
            if model.comp_ae is None or model.comp_unet is None:
                # Reinitialize composition branch if missing from model
                if args.composition_network_type == 'light_v4':
                    model.comp_ae = CompositionAutoEncoder_Light_v4(n_clusters=args.n_clusters).to(device)
                    model.comp_unet = CompositionUNet_Light_v4(n_clusters=args.n_clusters).to(device)
                else:  # 'cnn'
                    model.comp_ae = CompositionAutoEncoder_CNN(n_clusters=args.n_clusters).to(device)
                    model.comp_unet = CompositionUNet_CNN(n_clusters=args.n_clusters).to(device)
                model.use_composition_branch = True
                model.composition_network_type = args.composition_network_type
            
            # Define loss functions
            weights = [1.0] * args.n_clusters  # Weights can be tuned
            dice_loss_f = DiceLoss(weights).to(device)
            multiclass_focal_loss = MultiClassFocalLoss(gamma=2, reduction='mean').to(device)
            focal_loss = sigmoid_focal_loss
            
            # Define optimizer (SALAD-style)
            comp_optimizer = torch.optim.Adam(
                [{"params": list(model.comp_ae.parameters()) + list(model.comp_unet.parameters()), "lr": 1e-5}],
                lr=1e-5, weight_decay=1e-5
            )
            
            # Visualization directory
            comp_vis_dir = os.path.join(comp_vis_base_dir, 'composition_branch_vis')
            os.makedirs(comp_vis_dir, exist_ok=True)
            
            # Train for 3 epochs
            num_epochs = 3
            iteration = 0
            for epoch in range(num_epochs):
                model.train()
                # Shuffle composition maps
                random.shuffle(composition_maps)
                
                for map_idx in tqdm(range(len(composition_maps)), ncols=80, desc=f"Epoch {epoch+1}/{num_epochs}"):
                    # Load composition map
                    seg = torch.from_numpy(np.load(composition_maps[map_idx])).float().to(device)  # (n_clusters, 37, 37)
                    seg = seg.unsqueeze(0)  # (1, n_clusters, 37, 37)
                    
                    # Pick another composition map for anomaly synthesis
                    random_idx = random.randint(0, len(composition_maps) - 1)
                    diff_seg = torch.from_numpy(np.load(composition_maps[random_idx])).float().to(device)
                    diff_seg = diff_seg.unsqueeze(0)
                    
                    # Use global background label (no per-image computation)
                    # Randomly pick anomaly strategy (2, 3, or 4)
                    strategy = random.choice([2, 3, 4])
                    
                    # Synthesize anomaly
                    seg_normal = seg.clone()  # Keep normal sample for batching and reconstruction target
                    anom_seg = seg.clone()  # Start from normal sample to build anomaly
                    
                    if strategy == 2:
                        # Strategy 2: change labels within same image
                        mask, recon_mask, anom_seg = change_label_same_img_feat(anom_seg.squeeze(0), diff_seg.squeeze(0), global_background_label)
                    elif strategy == 3:
                        # Strategy 3: copy label cluster from another image
                        mask, recon_mask, anom_seg = copy_label_from_other_image(anom_seg.squeeze(0), diff_seg.squeeze(0), global_background_label)
                    else:  # strategy == 4
                        # Strategy 4: combine strategies 2 and 3
                        mask, recon_mask, anom_seg = combine_strategy_2_and_3(anom_seg.squeeze(0), diff_seg.squeeze(0), global_background_label)
                    
                    mask = mask.unsqueeze(0).to(device)  # (1, 37, 37) - anomaly region mask
                    anom_seg = anom_seg.unsqueeze(0).to(device)  # (1, n_clusters, 37, 37) - anomalous composition map
                    
                    # Store data for visualization (before batching)
                    seg_normal_vis = seg_normal.squeeze(0).cpu().numpy()  # (n_clusters, 37, 37) - normal composition map
                    anom_seg_vis = anom_seg.squeeze(0).cpu().numpy()  # (n_clusters, 37, 37) - anomalous composition map
                    mask_vis = mask.squeeze(0).cpu().numpy()  # (37, 37) - synthesized mask
                    
                    # ===== Batch size 2: pair anomalous and normal samples =====
                    # batch[0]: anomalous sample (anom_seg) — train detection and reconstruction
                    # batch[1]: normal sample (seg_normal) — train faithful normal reconstruction
                    input_batch = torch.cat([anom_seg, seg_normal], dim=0)  # (2, n_clusters, 37, 37)
                    # mask: batch[0] has anomaly mask; batch[1] is all zeros (normal)
                    mask_batch = torch.cat([mask, torch.zeros_like(mask)], dim=0)  # (2, 37, 37)
                    # Reconstruction target: both samples should be normal (anomaly -> normal, normal stays normal)
                    target_batch = torch.cat([seg_normal, seg_normal], dim=0)  # (2, n_clusters, 37, 37)
                    target_argmax = target_batch.argmax(dim=1)  # (2, 37, 37)
                    
                    # Forward pass
                    # Reconstruction net: input batch (anomaly + normal), output reconstruction
                    seg_recon = model.comp_ae(input_batch).softmax(dim=1)  # (2, n_clusters, 37, 37)
                    # Discriminator: input is concatenation of sample + reconstruction
                    unet_input = torch.cat([input_batch, seg_recon], dim=1)  # (2, n_clusters*2, 37, 37)
                    pred_mask = model.comp_unet(unet_input).squeeze(1)  # (2, 37, 37)
                    
                    # Compute loss
                    # Reconstruction loss: output should match normal target for both samples
                    loss_comp_recon = multiclass_focal_loss(seg_recon, target_argmax) + dice_loss_f(seg_recon, target_argmax)
                    # Discriminator loss: batch[0] predicts anomaly mask; batch[1] predicts zeros
                    loss_comp_mask = 5 * focal_loss(pred_mask, mask_batch, reduction='mean') + F.l1_loss(torch.sigmoid(pred_mask), mask_batch)
                    
                    loss_total = loss_comp_recon + loss_comp_mask
                    
                    # Backward pass
                    comp_optimizer.zero_grad()
                    loss_total.backward()
                    comp_optimizer.step()
                    
                    iteration += 1
                    
                    # Visualize every 200 iterations
                    if iteration % 200 == 0:
                        model.eval()
                        with torch.no_grad():
                            
                            # Visualize original composition map (normal)
                            seg_orig_cluster = onehot_to_cluster(seg_normal_vis)  # (37, 37)
                            visualize_clustering_result(
                                seg_orig_cluster,
                                f'iter_{iteration}_original',
                                comp_vis_dir,
                                (518, 518),
                                n_clusters=args.n_clusters
                            )
                            
                            # Visualize anomalous composition map
                            anom_seg_cluster = onehot_to_cluster(anom_seg_vis)  # (37, 37)
                            visualize_clustering_result(
                                anom_seg_cluster,
                                f'iter_{iteration}_anomaly',
                                comp_vis_dir,
                                (518, 518),
                                n_clusters=args.n_clusters
                            )
                            
                            # Visualize synthesized mask
                            mask_resized = cv2.resize(mask_vis.astype(np.float32), (518, 518), interpolation=cv2.INTER_NEAREST)
                            plt.imsave(
                                os.path.join(comp_vis_dir, f'iter_{iteration}_mask.png'),
                                mask_resized,
                                cmap='gray',
                                vmin=0,
                                vmax=1
                            )
                            plt.close()
                            
                            # Visualize reconstruction
                            seg_recon_vis = seg_recon[0].cpu().numpy()  # (n_clusters, 37, 37)
                            seg_recon_cluster = onehot_to_cluster(seg_recon_vis)  # (37, 37)
                            visualize_clustering_result(
                                seg_recon_cluster,
                                f'iter_{iteration}_recon',
                                comp_vis_dir,
                                (518, 518),
                                n_clusters=args.n_clusters
                            )
                            
                            # Visualize discriminator output
                            pred_mask_vis = pred_mask[0].cpu().numpy()  # (37, 37)
                            pred_mask_resized = cv2.resize(pred_mask_vis.astype(np.float32), (518, 518), interpolation=cv2.INTER_LINEAR)
                            plt.imsave(
                                os.path.join(comp_vis_dir, f'iter_{iteration}_disc.png'),
                                pred_mask_resized,
                                cmap='viridis',
                                vmin=0,
                                vmax=1
                            )
                            plt.close()
                        model.train()
                    
                    # Log losses
                    if iteration % 100 == 0:
                        print_fn(f'[Composition Branch] Epoch [{epoch+1}/{num_epochs}], Iteration [{iteration}], '
                                f'Recon Loss: {loss_comp_recon.item():.4f}, Disc Loss: {loss_comp_mask.item():.4f}')
            
            print_fn("Composition branch training completed!")
            print_fn(f"Visualization results saved to {comp_vis_dir}")
            
            # Save model including composition branch
            torch.save(model.state_dict(), os.path.join(save_dir, args.weight_file_name))
            print_fn("Model with composition branch saved.")


    # ===== Testing
    elif args.phase == "test":
        # ---------- Load model
        if args.weight_path is not None:
            weight_path = args.weight_path
        else:
            dataset_name = args.dataset_path.split('/')[-1]
            weight_path = f"saved_results/{args.save_name}/{dataset_name}/{args.class_name}/{args.weight_file_name}"
        print(f"Loading weights from {weight_path}")
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
            if args.composition_network_type == 'light_v4':
                model.comp_ae = CompositionAutoEncoder_Light_v4(n_clusters=args.n_clusters).to(device)
                model.comp_unet = CompositionUNet_Light_v4(n_clusters=args.n_clusters).to(device)
            else:  # 'cnn'
                model.comp_ae = CompositionAutoEncoder_CNN(n_clusters=args.n_clusters).to(device)
                model.comp_unet = CompositionUNet_CNN(n_clusters=args.n_clusters).to(device)
            model.use_composition_branch = True
            model.composition_network_type = args.composition_network_type
        
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
            compute_metrics=not args.disable_compute_metrics,
            visualize_score_distribution=False,
            compute_per_class_auroc=False,
            visualize_pixel_roc=False,
            save_name=args.save_name,
            n_clusters=args.n_clusters if args.n_clusters > 0 else None,
            normalization_params=normalization_params,
            skip_visualization=args.disable_visualize,
            apply_fg_prob_map=args.post_fusion,
        )

        if results is not None:
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
            print_fn(  # Print metrics
                'I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
        else:
            print_fn("Metric computation skipped")

    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Force synchronous CUDA ops (CPU waits for GPU before continuing)
    parser = argparse.ArgumentParser(description='')  # Create argument parser

    # dataset info
    parser.add_argument('--dataset_path', type=str, default=r'./datasets/MIAD')
    parser.add_argument('--class_name', type=str, default=r'catenary_dropper')

    # save info
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='MIAD_logical')
    parser.add_argument('--visualize_save_dir', type=str, default='visualize_results')
    parser.add_argument('--weight_file_name', type=str, default='model.pth')

    # model info
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--crop_size', type=int, default=518)
    parser.add_argument('--INP_num', type=int, default=6)
    parser.add_argument('--disable_segmentation_branch', action='store_true', default=False)  # Disable segmentation branch
    parser.add_argument('--post_fusion', type=str2bool, default=False)
    parser.add_argument('--fg_map_dilate_ksize', type=int, default=5)
    parser.add_argument('--bg_feature_aggregation_binary', type=str2bool, default=False,
                        help='If True: binarize foreground probability map during background aggregation (foreground keeps encoder features, background replaced by background prototype); otherwise use probability-weighted aggregation')
    parser.add_argument('--n_clusters', type=int, default=4)  # Number of k-means clusters
    parser.add_argument('--composition_network_type', type=str, default='light_v4', choices=['cnn', 'light_v4'],
                        help='Composition branch network type: cnn (v2-1, original), light_v4 (v2-4, ultra-lightweight CNN)')

    # training info
    parser.add_argument('--ad_epochs', type=int, default=2)
    parser.add_argument('--seg_epochs', type=int, default=1)
    parser.add_argument('--ad_batch_size', type=int, default=16)
    parser.add_argument('--seg_batch_size', type=int, default=8)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--disable_amp', action='store_true', default=False)  # Disable automatic mixed precision
    parser.add_argument('--skip_pretrain', action='store_true', default=False)  # Skip pretraining (seg + AD branches); go straight to composition branch training

    # testing_info
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--disable_visualize', action='store_true', default=False)  # Disable visualization
    parser.add_argument('--disable_compute_metrics', action='store_true', default=False)  # Skip metric computation

    args = parser.parse_args()
    
    dataset_name = args.dataset_path.split('/')[-1]
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name, dataset_name, args.class_name))
    print_fn = logger.info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    main(args)


