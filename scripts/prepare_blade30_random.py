"""
Prepare Blade30 dataset with per-subdirectory random split.

Rules:
1) Read every blade sample directory under:
   - datasets/raw/Blade30/3_blade_1_15_with_labeldata
   - datasets/raw/Blade30/3_blade_16_30_with_labeldata
2) In each sample directory:
   - Images with same-name .json are defect images:
       copy image -> datasets/Blade30/test/defect
       build binary mask from JSON -> datasets/Blade30/ground_truth/defect (PNG)
  - Images without .json are good images:
      random sample K images to test/good, where
      K = min(defect count, floor(good count / 2))
       copy remaining good images to train/good
       copy corresponding foreground masks from sample_dir/mask/<stem>.png to datasets/Blade30/mask
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG"}


@dataclass
class DirStat:
    sample_dir: Path
    defect_count: int = 0
    good_total: int = 0
    good_test_count: int = 0
    good_train_count: int = 0
    missing_train_mask: int = 0


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_dirs(out_root: Path) -> dict[str, Path]:
    dirs = {
        "test_defect": out_root / "test" / "defect",
        "test_good": out_root / "test" / "good",
        "train_good": out_root / "train" / "good",
        "gt_defect": out_root / "ground_truth" / "defect",
        "train_mask": out_root / "mask",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_name(sample_dir: Path, img_name: str) -> str:
    blade = sample_dir.parent.name
    sample = sample_dir.name
    return f"{blade}_{sample}_{img_name}"


def iter_sample_dirs(raw_root: Path) -> list[Path]:
    group_dirs = [
        raw_root / "3_blade_1_15_with_labeldata",
        raw_root / "3_blade_16_30_with_labeldata",
    ]
    sample_dirs: list[Path] = []
    for g in group_dirs:
        if not g.is_dir():
            raise FileNotFoundError(f"Missing directory: {g}")
        blades = sorted([p for p in g.iterdir() if p.is_dir()], key=lambda p: p.name)
        for blade_dir in blades:
            # Expected exactly one sample subdir like 1_0 / 2_1 ...
            children = sorted([p for p in blade_dir.iterdir() if p.is_dir() and p.name != "mask"], key=lambda p: p.name)
            if not children:
                raise RuntimeError(f"No sample subdirectory found under: {blade_dir}")
            if len(children) > 1:
                raise RuntimeError(f"Multiple sample subdirectories found under {blade_dir}: {[c.name for c in children]}")
            sample_dirs.append(children[0])
    return sample_dirs


def parse_json_mask(json_path: Path) -> np.ndarray:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    h = int(data["imageHeight"])
    w = int(data["imageWidth"])
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data.get("shapes", []):
        pts = shape.get("points") or []
        if not pts:
            continue
        st = shape.get("shape_type", "polygon")
        arr = np.asarray(pts, dtype=np.float64)

        if st in ("polygon", "linestrip"):
            poly = np.round(arr).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [poly], 255)
        elif st == "rectangle":
            xs, ys = arr[:, 0], arr[:, 1]
            x1, x2 = int(np.floor(xs.min())), int(np.ceil(xs.max()))
            y1, y2 = int(np.floor(ys.min())), int(np.ceil(ys.max()))
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
        elif st == "circle":
            if len(arr) >= 2:
                center = np.round(arr[0]).astype(int)
                edge = np.round(arr[1]).astype(int)
                radius = int(np.linalg.norm(edge - center))
                cv2.circle(mask, tuple(center.tolist()), radius, 255, thickness=-1)
        else:
            # Unknown shapes are ignored to keep pipeline robust.
            continue
    return mask


def copy_file(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY] copy {src} -> {dst}")
        return
    shutil.copy2(src, dst)


def write_mask(mask: np.ndarray, out_path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY] write mask {out_path}")
        return
    ok = cv2.imwrite(str(out_path), mask)
    if not ok:
        raise RuntimeError(f"Failed to write mask: {out_path}")


def process_one_sample(
    sample_dir: Path,
    out_dirs: dict[str, Path],
    rng: random.Random,
    dry_run: bool,
) -> DirStat:
    stat = DirStat(sample_dir=sample_dir)
    all_images = sorted([p for p in sample_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS], key=lambda p: p.name)
    mask_dir = sample_dir / "mask"

    defect_images: list[Path] = []
    good_images: list[Path] = []

    for img in all_images:
        if img.with_suffix(".json").is_file():
            defect_images.append(img)
        else:
            good_images.append(img)

    stat.defect_count = len(defect_images)
    stat.good_total = len(good_images)

    # 1) Defect images -> test/defect and generated gt masks.
    for img in defect_images:
        out_name = safe_name(sample_dir, img.name)
        dst_img = out_dirs["test_defect"] / out_name
        copy_file(img, dst_img, dry_run)

        json_path = img.with_suffix(".json")
        mask = parse_json_mask(json_path)
        gt_name = Path(out_name).with_suffix(".png").name
        dst_gt = out_dirs["gt_defect"] / gt_name
        write_mask(mask, dst_gt, dry_run)

    # 2) Randomly sample good images -> test/good, capped at half of good total.
    max_good_for_test = len(good_images) // 2
    k = min(stat.defect_count, max_good_for_test)
    sampled_good = set(rng.sample(good_images, k)) if k > 0 else set()
    stat.good_test_count = len(sampled_good)

    for img in sorted(sampled_good, key=lambda p: p.name):
        out_name = safe_name(sample_dir, img.name)
        dst = out_dirs["test_good"] / out_name
        copy_file(img, dst, dry_run)

    # 3) Remaining good -> train/good; copy corresponding foreground mask to datasets/Blade30/mask.
    train_goods = [p for p in good_images if p not in sampled_good]
    stat.good_train_count = len(train_goods)

    for img in train_goods:
        out_name = safe_name(sample_dir, img.name)
        dst_img = out_dirs["train_good"] / out_name
        copy_file(img, dst_img, dry_run)

        src_mask = mask_dir / f"{img.stem}.png"
        dst_mask = out_dirs["train_mask"] / Path(out_name).with_suffix(".png").name
        if src_mask.is_file():
            copy_file(src_mask, dst_mask, dry_run)
        else:
            stat.missing_train_mask += 1
            print(f"[WARN] Missing train foreground mask: {src_mask}")

    return stat


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Prepare Blade30 dataset with random per-subdir good sampling.")
    parser.add_argument("--raw-root", type=Path, default=root / "datasets" / "raw" / "Blade30")
    parser.add_argument("--out-root", type=Path, default=root / "datasets" / "Blade30")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for per-directory good sampling.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    out_root = args.out_root.resolve()
    out_dirs = ensure_dirs(out_root)
    rng = random.Random(args.seed)

    sample_dirs = iter_sample_dirs(raw_root)
    if len(sample_dirs) != 30:
        print(f"[WARN] Expected 30 sample directories, got {len(sample_dirs)}.")

    stats: list[DirStat] = []
    for sd in sample_dirs:
        stat = process_one_sample(sd, out_dirs, rng, args.dry_run)
        stats.append(stat)
        print(
            f"[INFO] {sd.parent.name}/{sd.name}: defect={stat.defect_count}, "
            f"good_total={stat.good_total}, good_test={stat.good_test_count}, "
            f"good_train={stat.good_train_count}, missing_train_mask={stat.missing_train_mask}"
        )

    sum_defect = sum(s.defect_count for s in stats)
    sum_good_test = sum(s.good_test_count for s in stats)
    sum_good_train = sum(s.good_train_count for s in stats)
    sum_missing_mask = sum(s.missing_train_mask for s in stats)
    print("\n[DONE] Blade30 preparation finished.")
    print(f"  sample_dirs={len(stats)}")
    print(f"  test/defect images={sum_defect}")
    print(f"  ground_truth/defect masks={sum_defect}")
    print(f"  test/good images={sum_good_test}")
    print(f"  train/good images={sum_good_train}")
    print(f"  train foreground masks copied={sum_good_train - sum_missing_mask}")
    print(f"  train foreground masks missing={sum_missing_mask}")


if __name__ == "__main__":
    main()

