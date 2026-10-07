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

1. Download the [MIAD](https://miad-2022.github.io/) images and place them under `datasets/MIAD`.
2. Download our foreground/background masks: **[MIAD_fg_bg_masks.zip](https://github.com/whouryoung/FBard/releases/download/masks-v1/MIAD_fg_bg_masks.zip)**
3. Extract the archive into `datasets/` so that each class has a `mask/` folder next to `train/` and `test/`.

Point `--dataset_path` and `--class_name` at one object category:

```
datasets/MIAD/metal_welding/
├── train/
│   └── good/          # normal training images (official MIAD)
├── mask/              # foreground/background masks (from the zip above)
└── test/
    ├── good/          # normal test images
    ├── defect_type_a/ # anomalous test images
    └── ...
ground_truth/          # optional; same folder level as test/
    └── defect_type_a/ # pixel-level GT masks for each defect type
```

The mask zip covers all 7 MIAD classes: `catenary_dropper`, `electrical_insulator`, `metal_welding`, `nut_and_bolt`, `photovoltaic_module`, `wind_turbine`, `witness_mark`.

Filename pairing: `train/good/000000.jpg` ↔ `mask/000000.png` (or `000000_mask.png`). Training only reads mask files **directly under** `mask/`, not nested folders.

Two classes ship extra variants; copy one variant's pngs into `mask/` before training:

| Class | Folders inside `mask/` | Typical choice |
| --- | --- | --- |
| `nut_and_bolt` | `mask/`, `no_plate/` | `mask/` |
| `photovoltaic_module` | `coarse/`, `fine/` | `fine/` |

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

## Logical Anomaly Detection

Turn on the composition / logical branch with `--use_logical_branch true`. Other defaults stay the same as standard training; a typical MIAD logical-anomaly run looks like:

```bash
python run_FBard.py \
  --use_logical_branch true \
  --phase train \
  --dataset_path ./datasets/MIAD \
  --class_name catenary_dropper \
  --save_name MIAD_logical
```

```bash
python run_FBard.py \
  --use_logical_branch true \
  --phase test \
  --dataset_path ./datasets/MIAD \
  --class_name catenary_dropper \
  --save_name MIAD_logical
```

Logical-only flags:

- `--n_clusters` (default `4`) — k-means clusters for composition maps
- `--skip_pretrain` — skip segmentation + AD pretraining and jump to composition training
- `--disable_compute_metrics` — skip metric computation during logical testing
