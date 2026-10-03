# FBard

FBard: Feature-Level Background Recalibration for Unsupervised Anomaly Detection in Unconstrained Environments (ACCV 2026)

## Requirements

- Python 3.8+
- PyTorch with CUDA (recommended)

```bash
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install `torch` and `torchvision` from [pytorch.org](https://pytorch.org/get-started/locally/) first, then run the same command.

## Dataset Layout

Download the [MIAD](https://miad-2022.github.io/) dataset and place it under `datasets/MIAD`.

Point `--dataset_path` and `--class_name` at one object category:

```
datasets/MIAD/metal_welding/
├── train/
│   └── good/          # normal training images
├── mask/              # foreground masks (paired with train/good by filename)
└── test/
    ├── good/          # normal test images
    ├── defect_type_a/ # anomalous test images
    └── ...
ground_truth/          # optional; same folder level as test/
    └── defect_type_a/ # pixel-level GT masks for each defect type
```

Mask pairing: for image `train/good/001.jpg`, use `mask/001.png` or `mask/001_mask.png`.

## Training

```bash
python run_FBard.py \
  --phase train \
  --dataset_path ./datasets/MIAD \
  --class_name metal_welding
```

Training runs in two stages:

1. **Segmentation branch** (`--seg_epochs`, default 1) — uses `train/good` + `mask/`
2. **Anomaly detection branch** (`--ad_epochs`, default 2) — uses `train/good`

Checkpoint is saved to:

```
saved_results/{save_name}/{dataset_name}/{class_name}/model.pth
```

## Testing

```bash
python run_FBard.py \
  --phase test \
  --dataset_path ./datasets/MIAD \
  --class_name metal_welding \
```

Loads weights from the path above by default. Override with `--weight_path`.

Visualizations are written to `visualize_results/{save_name}/` unless `--disable_visualize` is set.

## Notes

- For logical anomaly detection (composition branch), use `run_FBard_logical.py` instead.
