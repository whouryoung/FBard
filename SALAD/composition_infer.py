from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from SALAD.composition_anomaly import cluster_to_onehot


@dataclass
class CompositionInferResult:
    anomaly_maps: torch.Tensor  # (B, 1, H, W), CPU
    disc_outputs: Optional[torch.Tensor] = None  # (B, 37, 37), CPU
    cluster_maps: Optional[List[np.ndarray]] = None
    recon_maps: Optional[List[np.ndarray]] = None


def composition_available(model, n_clusters) -> bool:
    return (
        n_clusters is not None
        and n_clusters > 0
        and hasattr(model, "comp_ae")
        and hasattr(model, "comp_unet")
        and model.comp_ae is not None
        and getattr(model, "kmeans_centers", None) is not None
    )


def run_composition_branch(
    model,
    img: torch.Tensor,
    n_clusters: int,
    device,
    gaussian_kernel,
    resize_to: Union[int, Sequence[int]],
    return_aux: bool = False,
) -> Optional[CompositionInferResult]:
    """
    Shared composition-branch forward used by validation normalization and test evaluation.

    k-means assign -> one-hot composition map -> AE reconstruct -> UNet discriminate
    -> resize + Gaussian blur.
    """
    if not composition_available(model, n_clusters):
        return None

    en_list, _ = model.extract_features_for_composition_map(img)
    kmeans_centers = model.kmeans_centers
    kmeans_centers_np = (
        kmeans_centers.cpu().numpy() if isinstance(kmeans_centers, torch.Tensor) else kmeans_centers
    )
    x = model.fuse_feature(en_list)  # (B, N, C)

    out_size = (resize_to, resize_to) if isinstance(resize_to, int) else resize_to
    B = img.shape[0]
    comp_anomaly_maps = []
    disc_outputs = []
    cluster_maps = []
    recon_maps = []

    for b in range(B):
        features = x[b].detach().cpu().numpy()  # (N, C)
        distances = np.linalg.norm(features[:, None, :] - kmeans_centers_np[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        feat_hw = int(np.sqrt(features.shape[0]))
        cluster_map = labels.reshape(feat_hw, feat_hw)

        seg_onehot = cluster_to_onehot(cluster_map, n_clusters)
        if isinstance(seg_onehot, torch.Tensor):
            seg_onehot_tensor = seg_onehot.unsqueeze(0).to(device)
        else:
            seg_onehot_tensor = torch.from_numpy(seg_onehot).float().unsqueeze(0).to(device)

        seg_recon = model.comp_ae(seg_onehot_tensor).softmax(dim=1)
        unet_input = torch.cat([seg_onehot_tensor, seg_recon], dim=1)
        pred_mask = model.comp_unet(unet_input).squeeze(1)

        if return_aux:
            cluster_maps.append(cluster_map)
            recon_maps.append(seg_recon[0].detach().cpu().numpy())
            disc_outputs.append(pred_mask.detach().cpu())

        pred_mask_resized = F.interpolate(
            pred_mask.unsqueeze(1),
            size=out_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        pred_mask_blurred = gaussian_kernel(pred_mask_resized.unsqueeze(1)).squeeze(1)
        comp_anomaly_maps.append(pred_mask_blurred.detach().cpu())

        del features, distances, labels, cluster_map, seg_onehot, seg_onehot_tensor
        del seg_recon, unet_input, pred_mask, pred_mask_resized, pred_mask_blurred

    del x, en_list, kmeans_centers_np

    if len(comp_anomaly_maps) == 0:
        return None

    anomaly_maps = torch.cat(comp_anomaly_maps, dim=0).unsqueeze(1)  # (B, 1, H, W)
    return CompositionInferResult(
        anomaly_maps=anomaly_maps,
        disc_outputs=torch.cat(disc_outputs, dim=0) if return_aux and disc_outputs else None,
        cluster_maps=cluster_maps if return_aux else None,
        recon_maps=recon_maps if return_aux else None,
    )
