import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json
from pathlib import Path

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train  # ImageNet normalization mean
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([  # Mask pipeline: resize, center crop (no normalization or binarization)
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

class ADDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        # These attributes hold path, gt path, label, and defect type for each image
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []  # tot=total; paths of all image files
        gt_tot_paths = []  # Paths of all gt files
        tot_labels = []   # All label ids
        tot_types = []   # All defect type names

        def _collect_files(folder: str, suffixes_lower: set):
            """
            Collect direct child files whose suffix matches `suffixes_lower` (case-insensitive).
            We intentionally do not recurse; expected structure is `test/<type>/*.ext`.
            """
            folder_path = Path(folder)
            if not folder_path.is_dir():
                return []
            out = []
            for p in folder_path.iterdir():
                if p.is_file() and p.suffix.lower() in suffixes_lower:
                    out.append(str(p))
            out.sort()
            return out

        if not os.path.isdir(self.img_path):
            raise FileNotFoundError(f"AD img_path not found: {self.img_path}")

        # Only treat sub-directories as defect types (avoid accidentally including files)
        defect_types = [
            d for d in os.listdir(self.img_path)
            if os.path.isdir(os.path.join(self.img_path, d))
        ]

        img_suffixes = {'.png', '.jpg', '.jpeg', '.bmp'}
        gt_suffixes = {'.png'}

        for defect_type in defect_types:
            if defect_type == 'good':  # Collect all matching image files in this folder
                img_paths = _collect_files(os.path.join(self.img_path, defect_type), img_suffixes)
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = _collect_files(os.path.join(self.img_path, defect_type), img_suffixes)
                gt_paths = _collect_files(os.path.join(self.gt_path, defect_type), gt_suffixes)

                # Keep the same pairing logic: sort both lists and assume index-alignment.
                # Sorting is already applied in `_collect_files`.
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path

