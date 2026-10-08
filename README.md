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

### MIAD

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

### Blade30

Blade30 is a drone-based wind turbine blade inspection dataset ([cong-yang/Blade30](https://github.com/cong-yang/Blade30)). Download the official release from that repository, then place the raw archives under `datasets/raw/Blade30` **before** running the preparation script.

Download options (from the upstream README):

- [Baidu Disc — part1 (blade1–15)](https://pan.baidu.com/s/17kv5Xadz1QcSrvoG58WtBw) (code: `1234`)
- [Baidu Disc — part2 (blade16–30)](https://pan.baidu.com/s/1hzcwdc6sBXOeja3nkfartg) (code: `1234`)
- [OneDrive — full dataset](https://1drv.ms/u/s!AoXJBmXKVWu5tmtUzCJULhrtYuIP?e=KYOtlo)
- [Google Drive — blade_1_15_with_annotation](https://drive.google.com/file/d/1HbB4t9xV2oCgSSxR9hMEOU6v9qDfetmR/view?usp=sharing)
- [Google Drive — blade_16_30_with_annotation](https://drive.google.com/file/d/1SwRdMzA7zCkNVlHuWvk8uK6eDToM0mUV/view?usp=sharing)

Expected raw layout after extraction:

```
datasets/raw/Blade30/
├── 3_blade_1_15_with_labeldata/
│   └── <blade_dir>/              # one directory per blade
│       └── <sample_dir>/         # exactly one sample subdirectory (not named mask)
│           ├── *.jpg / *.jpeg    # images
│           ├── *.json            # same stem as a defect image → defect annotation
│           └── mask/
│               └── <stem>.png    # foreground mask for normal images
└── 3_blade_16_30_with_labeldata/
    └── ...                       # same structure
```

Requirements:

- Both group folders `3_blade_1_15_with_labeldata` and `3_blade_16_30_with_labeldata` must exist.
- Under each blade directory there must be **exactly one** sample subdirectory (directories named `mask` are ignored when discovering samples).
- Images with a same-name `.json` are treated as defects; images without `.json` are treated as normal (good).
- Foreground masks used for training live at `<sample_dir>/mask/<image_stem>.png`.
- The preparation script expects **30** sample directories in total.

After the raw data is in place, run:

```bash
python scripts/prepare_blade30_random.py
```

Optional flags: `--raw-root`, `--out-root`, `--seed` (default `42`), `--dry-run`.

The script writes the FBard-ready split to `datasets/Blade30/`:

```
datasets/Blade30/
├── train/
│   └── good/
├── test/
│   ├── good/
│   └── defect/
├── ground_truth/
│   └── defect/
└── mask/                         # foreground masks for train/good
```

Then point training/testing at this prepared root (`dataset_path` / `class_name` → `datasets/Blade30`):

```bash
python run_FBard.py \
  --phase train \
  --dataset_path ./datasets \
  --class_name Blade30
```

### InsPLAD

InsPLAD is a UAV power-line asset inspection dataset ([andreluizbvs/InsPLAD](https://github.com/andreluizbvs/InsPLAD)). Download the official release from that repository (Mendeley Data link in the upstream README), then place the unpacked images under `datasets/raw/InsPLAD` **before** running the preparation scripts.

The upstream archive contains three sub-datasets (`InsPLAD-det`, supervised fault classification, and unsupervised anomaly detection). Extract so that all images for glass / polymer insulators are reachable under the raw root (any nested layout is fine).

Expected raw placement:

```
datasets/raw/InsPLAD/
└── .../                          # official layout preserved; any subdirectory depth is OK
    └── *.jpg / *.png / ...       # images indexed recursively by basename
```

Requirements:

- Default `--raw-root` is `datasets/raw/InsPLAD`. Scripts recursively index all images under this tree and match filenames listed in the manifests.
- Optional: narrow the search with `--glass-subdir` / `--polymer-subdir` (relative to `--raw-root`).
- Before running each prepare script, place the corresponding `annotation/` files under that script’s **output** root (not under `raw/`):

| Subset | Output root | Required under `annotation/` |
| --- | --- | --- |
| Glass insulator | `datasets/InsPLAD/glass-insulator/` | `train_good.txt`, `test_good.txt`, `test_missingcap.txt`, `train_mask.json`, `test_missingcap.json` |
| Polymer insulator | `datasets/InsPLAD/polymer-insulator/` | `train_good.txt`, `test_good.txt`, `test_torned-up.txt`, `train_mask.json`, `test_torned-up.json` |

Manifest `.txt` files list one image filename per line (`#` comments and blank lines ignored). Merged LabelMe bundles must use format `insplad_labelme_bundle` (not generated by the prepare scripts).

After the raw images and annotations are in place, run:

```bash
python scripts/prepare_insplad_glass_insulator.py
python scripts/prepare_insplad_polymer_insulator.py
```

Optional flags (both scripts): `--raw-root`, `--out`, `--dry-run`, `--no-images`, `--no-masks`, `--no-renumber`, plus `--glass-subdir` / `--polymer-subdir`.

Each script copies images, rasterizes masks from the merged JSON, then renumbers files to `000.*`, `001.*`, … Writing to:

```
datasets/InsPLAD/glass-insulator/
├── train/
│   └── good/
├── test/
│   ├── good/
│   └── missingcap/
├── ground_truth/
│   └── missingcap/
├── mask/                         # foreground masks for train/good
└── annotation/

datasets/InsPLAD/polymer-insulator/
├── train/
│   └── good/
├── test/
│   ├── good/
│   └── torned-up/
├── ground_truth/
│   └── torned-up/
├── mask/
└── annotation/
```

Then point training/testing at a prepared class (`dataset_path` / `class_name`):

```bash
python run_FBard.py \
  --phase train \
  --dataset_path ./datasets/InsPLAD \
  --class_name glass-insulator
```

```bash
python run_FBard.py \
  --phase train \
  --dataset_path ./datasets/InsPLAD \
  --class_name polymer-insulator
```

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
