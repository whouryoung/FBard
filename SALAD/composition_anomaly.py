import numpy as np
import torch
from scipy.ndimage import label, binary_dilation, generate_binary_structure


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
    boundary_pixels = []
    boundary_pixels.extend(cluster_map[0, :].flatten())
    boundary_pixels.extend(cluster_map[H - 1, :].flatten())
    boundary_pixels.extend(cluster_map[:, 0].flatten())
    boundary_pixels.extend(cluster_map[:, W - 1].flatten())
    boundary_pixels = np.array(boundary_pixels)

    unique_labels, counts = np.unique(boundary_pixels, return_counts=True)
    background_label = unique_labels[np.argmax(counts)]
    return int(background_label)


def change_label_same_img_feat(seg_mask, diff_seg_mask, background_label):
    """Strategy 2: change labels within the same image (37×37), excluding background label."""
    if isinstance(seg_mask, torch.Tensor):
        device = seg_mask.device
        seg_mask = seg_mask.clone()
    else:
        device = torch.device('cpu')
        seg_mask = torch.from_numpy(seg_mask).float()

    H, W = seg_mask.shape[1], seg_mask.shape[2]
    mask = torch.zeros((H, W), device=device)
    recon_mask = torch.zeros((H, W), device=device)
    anom_seg = seg_mask.clone()

    original_cluster = onehot_to_cluster(anom_seg.cpu().numpy())

    component_list, ci_list = get_connected_components_37x37(anom_seg)
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

    if isinstance(connected_component, np.ndarray):
        connected_component = torch.from_numpy(connected_component).bool().to(device)
    else:
        connected_component = connected_component.to(device)

    recon_mask[connected_component] = 1

    temp_mask = torch.zeros((H, W), device=device)
    temp_mask[connected_component] = 1
    temp_mask_np = temp_mask.cpu().numpy()
    dilated_mask = binary_dilation(temp_mask_np, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    dilated_mask = torch.from_numpy(dilated_mask).bool().to(device)
    dilated_mask[connected_component] = False

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

    new_cluster = onehot_to_cluster(anom_seg.cpu().numpy())
    mask = torch.from_numpy((original_cluster != new_cluster).astype(np.float32)).to(device)

    return mask, recon_mask, anom_seg


def copy_label_from_other_image(seg_mask, diff_seg_mask, background_label):
    """
    Strategy 3: copy a label cluster from another image to the current image.
    Only copies connected components above the threshold, not all pixels of the label.
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

    original_cluster = onehot_to_cluster(anom_seg.cpu().numpy())

    available_labels = [c for c in range(diff_seg_mask.shape[0]) if c != background_label]
    if len(available_labels) == 0:
        return mask, recon_mask, anom_seg

    ci_copy = available_labels[torch.randint(0, len(available_labels), (1,)).item()]

    label_mask = diff_seg_mask[ci_copy] > 0.5
    if not label_mask.any():
        return mask, recon_mask, anom_seg

    label_mask_np = label_mask.cpu().numpy() if isinstance(label_mask, torch.Tensor) else label_mask
    structure = generate_binary_structure(rank=2, connectivity=2)
    labeled_mask, num_features = label(label_mask_np, structure=structure)

    valid_components = []
    for i in range(1, num_features + 1):
        component = labeled_mask == i
        if component.sum() > 10:
            valid_components.append(component)

    if len(valid_components) == 0:
        return mask, recon_mask, anom_seg

    selected_component = valid_components[torch.randint(0, len(valid_components), (1,)).item()]
    copy_mask = torch.from_numpy(selected_component).bool().to(device)

    anom_seg[:, copy_mask] = 0
    anom_seg[ci_copy, copy_mask] = 1

    new_cluster = onehot_to_cluster(anom_seg.cpu().numpy())
    mask = torch.from_numpy((original_cluster != new_cluster).astype(np.float32)).to(device)
    recon_mask[copy_mask] = 1

    return mask, recon_mask, anom_seg


def combine_strategy_2_and_3(seg_mask, diff_seg_mask, background_label):
    """
    Strategy 4: combine strategies 2 and 3.
    First apply strategy 2 (change labels within the same image), then strategy 3 (copy label from another image).
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

    original_cluster = onehot_to_cluster(anom_seg.cpu().numpy())

    component_list, ci_list = get_connected_components_37x37(anom_seg)
    if len(component_list) > 0:
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

            temp_mask = torch.zeros((H, W), device=device)
            temp_mask[connected_component] = 1
            temp_mask_np = temp_mask.cpu().numpy()
            dilated_mask = binary_dilation(temp_mask_np, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            dilated_mask = torch.from_numpy(dilated_mask).bool().to(device)
            dilated_mask[connected_component] = False

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

    available_labels = [c for c in range(diff_seg_mask.shape[0]) if c != background_label]
    if len(available_labels) > 0:
        ci_copy = available_labels[torch.randint(0, len(available_labels), (1,)).item()]
        label_mask = diff_seg_mask[ci_copy] > 0.5
        if label_mask.any():
            label_mask_np = label_mask.cpu().numpy() if isinstance(label_mask, torch.Tensor) else label_mask
            structure = generate_binary_structure(rank=2, connectivity=2)
            labeled_mask, num_features = label(label_mask_np, structure=structure)

            valid_components = []
            for i in range(1, num_features + 1):
                component = labeled_mask == i
                if component.sum() > 10:
                    valid_components.append(component)

            if len(valid_components) > 0:
                selected_component = valid_components[torch.randint(0, len(valid_components), (1,)).item()]
                copy_mask = torch.from_numpy(selected_component).bool().to(device)
                anom_seg[:, copy_mask] = 0
                anom_seg[ci_copy, copy_mask] = 1

    new_cluster = onehot_to_cluster(anom_seg.cpu().numpy())
    mask = torch.from_numpy((original_cluster != new_cluster).astype(np.float32)).to(device)

    return mask, recon_mask, anom_seg
