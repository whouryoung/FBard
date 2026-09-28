import glob
import numpy as np
import os
import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import wrap, TVTensor, Mask
from torch import nn, Tensor
from typing import Any, Union


class SegTrainDataset(torch.utils.data.Dataset):

    def __init__(self,
                 root,  # dataset_name/class_name
                 img_size=(518, 518),
                 color_jitter_enabled=True,  # Color jitter augmentation
                 scale_range=(0.5, 2.0),
                 ):

        self.img_path = os.path.join(root, 'train/good')
        self.gt_path = os.path.join(root, 'mask')

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

        # load dataset
        # These attributes hold path and gt path for each image
        self.img_paths, self.gt_paths = self.load_dataset()  # Paths of all image and mask files

    def load_dataset(self):

        img_paths = glob.glob(self.img_path + "/*.png") + \
                    glob.glob(self.img_path + "/*.jpg") + \
                    glob.glob(self.img_path + "/*.JPG") + \
                    glob.glob(self.img_path + "/*.jpeg") + \
                    glob.glob(self.img_path + "/*.JPEG") + \
                    glob.glob(self.img_path + "/*.bmp")
        gt_paths = glob.glob(self.gt_path + "/*.png") + \
                   glob.glob(self.gt_path + "/*.jpg") + \
                   glob.glob(self.gt_path + "/*.JPG") + \
                   glob.glob(self.gt_path + "/*.jpeg") + \
                   glob.glob(self.gt_path + "/*.JPEG") + \
                   glob.glob(self.gt_path + "/*.bmp")

        if not img_paths:
            raise AssertionError(
                f"No training images found under {self.img_path} "
                f"(supported: .png/.jpg/.JPG/.jpeg/.bmp)."
            )

        stem_to_gt = {}
        for p in gt_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem in stem_to_gt:
                raise ValueError(
                    f"Duplicate stem in mask directory: stem='{stem}' -> "
                    f"{stem_to_gt[stem]} and {p}"
                )
            stem_to_gt[stem] = p

        img_tot_paths = []
        gt_tot_paths = []
        missing_masks = []
        used_mask_paths = set()

        for ip in sorted(img_paths):
            stem = os.path.splitext(os.path.basename(ip))[0]
            mask_path = None
            # Pair with xxx.jpg: prefer xxx.png, else xxx_mask.png (any supported extension)
            for cand_stem in (stem, stem + "_mask"):
                if cand_stem in stem_to_gt:
                    mask_path = stem_to_gt[cand_stem]
                    break
            if mask_path is None:
                missing_masks.append(ip)
                continue
            img_tot_paths.append(ip)
            gt_tot_paths.append(mask_path)
            used_mask_paths.add(mask_path)

        unused_masks = [p for p in gt_paths if p not in used_mask_paths]

        if missing_masks or unused_masks:
            msg = [
                "train/good and mask must align by filename: image xxx.* -> mask xxx.* or xxx_mask.*.",
                f"Image dir: {self.img_path}, mask dir: {self.gt_path}",
                f"Paired samples: {len(img_tot_paths)}",
            ]
            if missing_masks:
                msg.append(f"Images missing a matching mask (first 5): {missing_masks[:5]}")
            if unused_masks:
                msg.append(f"Masks unused by any training image (first 5): {unused_masks[:5]}")
            raise AssertionError("\n".join(msg))

        return np.array(img_tot_paths), np.array(gt_tot_paths)

    def __len__(self):
        return len(self.img_paths)


    def process_gt(self, gt):

        gt_array = np.array(gt, dtype=np.uint8)  # Convert to NumPy array
        gt_tensor = torch.from_numpy(gt_array)  # Convert to PyTorch tensor

        mask_0 = (gt_tensor < 128)
        mask_255 = (gt_tensor >= 128)

        # Stack into a (2, h, w) boolean mask
        return Mask(torch.stack([mask_0, mask_255], dim=0))


    def __getitem__(self, idx):
        img_path, gt_path = self.img_paths[idx], self.gt_paths[idx]
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')

        to_tensor = transforms.ToTensor()  # Convert to tensor
        img = to_tensor(img)  # 0-255 -> 0-1

        masks = self.process_gt(gt)

        labels = torch.tensor([0, 1])
        is_crowd = torch.tensor([False, False])

        target = {'masks': masks,
                  'labels': labels,
                  'is_crowd': is_crowd}
        img, target = self.transforms(img, target)

        assert img.size()[1:] == target['masks'].size()[1:], "image.size != gt.size"

        return img, target


class SegTestDataset(torch.utils.data.Dataset):

    def __init__(self,
                 root,  # rod
                 img_size=(518, 518),
                 ):

        self.img_size = img_size
        self.img_path = os.path.join(root, 'test')
        self.img_paths = self.load_dataset()  # Paths of all image files

    def load_dataset(self):

        img_total_paths = []

        defect_types = os.listdir(self.img_path)
        for defect_type in defect_types:
            img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                        glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                        glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
            img_total_paths.extend(img_paths)

        return np.array(img_total_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        to_tensor = transforms.ToTensor()  # Convert to tensor
        img = to_tensor(img)  # 0-255 -> 0-1
        img = F.resize(img, self.img_size, interpolation=InterpolationMode.BILINEAR, antialias=True)

        return img, img_path


class Transforms(nn.Module):
    def __init__(
        self,
        img_size,
        color_jitter_enabled: bool,
        scale_range,
        max_brightness_delta: int = 32,
        max_contrast_factor: float = 0.5,
        saturation_factor: float = 0.5,
        max_hue_delta: int = 18,
    ):
        super().__init__()

        self.img_size = img_size
        self.color_jitter_enabled = color_jitter_enabled
        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0

        self.random_horizontal_flip = T.RandomHorizontalFlip()
        self.scale_jitter = T.ScaleJitter(target_size=img_size, scale_range=scale_range)
        self.random_crop = T.RandomCrop(img_size)

    def _random_factor(self, factor: float, center: float = 1.0):
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    def _brightness(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(
                img, self._random_factor(self.max_brightness_factor)
            )

        return img

    def _contrast(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(img, self._random_factor(self.max_contrast_factor))

        return img

    def _saturation_and_hue(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(
                img, self._random_factor(self.max_saturation_factor)
            )

        if torch.rand(()) < 0.5:
            img = F.adjust_hue(img, self._random_factor(self.max_hue_delta, center=0.0))

        return img

    def color_jitter(self, img):
        if not self.color_jitter_enabled:
            return img

        img = self._brightness(img)

        if torch.rand(()) < 0.5:
            img = self._contrast(img)
            img = self._saturation_and_hue(img)
        else:
            img = self._saturation_and_hue(img)
            img = self._contrast(img)

        return img

    def _filter(self, target, keep: Tensor):
        return {k: wrap(v[keep], like=v) for k, v in target.items()}

    def pad(self, img: Tensor, target):
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        img = F.pad(img, padding)
        target["masks"] = F.pad(target["masks"], padding)

        return img, target


    def forward(self, img: Tensor, target):

        img_orig, target_orig = img, target

        target = self._filter(target, ~target["is_crowd"])

        img = self.color_jitter(img)

        img = F.resize(img, self.img_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        target['masks'] = F.resize(target['masks'], self.img_size, interpolation=InterpolationMode.NEAREST)

        img, target = self.random_horizontal_flip(img, target)
        img, target = self.scale_jitter(img, target)
        img, target = self.pad(img, target)
        img, target = self.random_crop(img, target)

        valid = target["masks"].flatten(1).any(1)
        if not valid.any():
            return self(img_orig, target_orig)

        target = self._filter(target, valid)

        return img, target


def train_collate(batch):
    imgs, targets = [], []

    for img, target in batch:
        imgs.append(img)
        targets.append(target)

    return torch.stack(imgs), targets

