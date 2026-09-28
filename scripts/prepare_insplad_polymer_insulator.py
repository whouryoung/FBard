"""
Build InsPLAD polymer-insulator dataset from raw files.

Put raw InsPLAD under: datasets/raw/InsPLAD/
By default this script recursively indexes all images under raw-root.
Use --polymer-subdir to restrict search scope under raw-root.

Required merged LabelMe bundle files under annotation/:
  - train_mask.json
  - test_torned-up.json

Default one-click pipeline:
  1) copy images by annotation txt lists
  2) generate binary masks from merged JSON
  3) sequentially rename files to 000.*, 001.*, ...
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".png", ".bmp", ".jpeg", ".jpg", ".tif", ".tiff", ".webp"}
IMAGE_SUFFIXES |= {s.upper() for s in IMAGE_SUFFIXES}

REN_IMAGE_EXTS = {
    ".png",
    ".bmp",
    ".jpeg",
    ".jpg",
    ".tif",
    ".tiff",
    ".webp",
    ".PNG",
    ".JPEG",
    ".JPG",
    ".BMP",
}

TRAIN_MASK_JSON = "train_mask.json"
TEST_TORNED_JSON = "test_torned-up.json"
BUNDLE_FORMAT = "insplad_labelme_bundle"


def load_merged_bundle(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != BUNDLE_FORMAT:
        print(f"[WARN] {path} missing expected format key '{BUNDLE_FORMAT}'", file=sys.stderr)
    return data


def rasterize_labelme_item(item: dict) -> tuple[np.ndarray, str]:
    h = int(item["imageHeight"])
    w = int(item["imageWidth"])
    mask = np.zeros((h, w), dtype=np.uint8)
    image_path = item.get("imagePath") or ""
    for shape in item.get("shapes", []):
        st = shape.get("shape_type")
        pts = shape.get("points") or []
        if not pts:
            continue
        arr = np.asarray(pts, dtype=np.float64)
        if st == "polygon":
            poly = np.round(arr).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [poly], 255)
        elif st == "rectangle":
            xs, ys = arr[:, 0], arr[:, 1]
            x1 = int(np.floor(xs.min()))
            y1 = int(np.floor(ys.min()))
            x2 = int(np.ceil(xs.max()))
            y2 = int(np.ceil(ys.max()))
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
        else:
            raise ValueError(f"Unsupported shape_type: {st!r} in image {image_path!r}")
    return mask, image_path


def mask_out_path_from_image_path(image_path: str) -> str:
    return f"{Path(image_path).stem}.png"


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_raw_search_root(raw_root: Path, polymer_subdir: str) -> Path:
    base = raw_root.resolve()
    s = (polymer_subdir or "").strip().replace("\\", "/").strip("/")
    if not s or s == ".":
        return base
    return (base / s).resolve()


def _iter_manifest_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _build_basename_index(search_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    dup: list[tuple[str, Path, Path]] = []
    for p in search_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in IMAGE_SUFFIXES:
            continue
        key = p.name.lower()
        if key in index:
            dup.append((key, index[key], p))
        else:
            index[key] = p
    if dup:
        for key, a, b in dup[:10]:
            print(f"[WARN] duplicate basename '{key}': keeping {a}, ignoring {b}", file=sys.stderr)
        if len(dup) > 10:
            print(f"[WARN] ... and {len(dup) - 10} more duplicate basenames", file=sys.stderr)
    return index


def _resolve_source(name: str, index: dict[str, Path]) -> Path | None:
    key = name.lower()
    if key in index:
        return index[key]
    stem = Path(name).stem.lower()
    for k, p in index.items():
        if Path(k).stem.lower() == stem:
            return p
    return None


def _clean_image_dir(d: Path, dry_run: bool) -> None:
    if not d.is_dir():
        return
    for p in d.iterdir():
        if p.is_file() and p.suffix in IMAGE_SUFFIXES:
            if dry_run:
                print(f"[DRY] delete {p}")
            else:
                p.unlink()


def _clean_mask_dir(d: Path, dry_run: bool) -> None:
    if not d.is_dir():
        return
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() == ".png":
            if dry_run:
                print(f"[DRY] delete {p}")
            else:
                p.unlink()


def run_copy_images(args) -> tuple[int, int]:
    raw_search = _resolve_raw_search_root(args.raw_root, args.polymer_subdir)
    if not raw_search.is_dir():
        print(f"[ERROR] Source directory not found: {raw_search}", file=sys.stderr)
        sys.exit(1)

    ann = (args.out / "annotation").resolve()
    specs = [
        (ann / "train_good.txt", args.out / "train" / "good"),
        (ann / "test_good.txt", args.out / "test" / "good"),
        (ann / "test_torned-up.txt", args.out / "test" / "torned-up"),
    ]
    for manifest, _ in specs:
        if not manifest.is_file():
            print(f"[ERROR] Manifest missing: {manifest}", file=sys.stderr)
            sys.exit(1)

    print(f"[INFO] Indexing images under {raw_search} ...")
    index = _build_basename_index(raw_search)
    print(f"[INFO] Found {len(index)} image files.")

    if args.clean:
        for _, dest in specs:
            _clean_image_dir(dest, args.dry_run)

    ok = 0
    miss = 0
    for manifest, dest in specs:
        names = _iter_manifest_lines(manifest)
        print(f"[INFO] {manifest.name}: {len(names)} entries -> {dest}")
        if not args.dry_run:
            dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            src = _resolve_source(name, index)
            if src is None:
                print(f"[MISS] {name}", file=sys.stderr)
                miss += 1
                continue
            dst = dest / name
            if args.dry_run:
                print(f"[DRY] {src} -> {dst}")
            else:
                shutil.copy2(src, dst)
            ok += 1
    print(f"[DONE] copied={ok}, missing={miss}")
    return ok, miss


def run_masks_from_merged(out_root: Path, dry_run: bool, clean_masks: bool) -> tuple[int, int]:
    ann = out_root / "annotation"
    train_json = ann / TRAIN_MASK_JSON
    test_json = ann / TEST_TORNED_JSON
    dst_train = out_root / "mask"
    dst_test = out_root / "ground_truth" / "torned-up"

    if not train_json.is_file() or not test_json.is_file():
        print(
            "[ERROR] Missing merged annotation bundle files:\n"
            f"  {train_json}\n"
            f"  {test_json}",
            file=sys.stderr,
        )
        sys.exit(1)

    if clean_masks:
        _clean_mask_dir(dst_train, dry_run)
        _clean_mask_dir(dst_test, dry_run)

    ok, err = 0, 0
    for merged_path, dest_dir in [(train_json, dst_train), (test_json, dst_test)]:
        bundle = load_merged_bundle(merged_path)
        items = bundle.get("items", [])
        print(f"[INFO] Masks from {merged_path.name}: {len(items)} items -> {dest_dir}")
        for item in items:
            try:
                mask, image_path = rasterize_labelme_item(item)
            except Exception as e:
                print(f"[ERROR] {merged_path.name} imagePath={item.get('imagePath')!r}: {e}", file=sys.stderr)
                err += 1
                continue
            out_name = mask_out_path_from_image_path(image_path)
            dst = dest_dir / out_name
            if dry_run:
                print(f"[DRY] mask -> {dst}")
            else:
                save_mask_png(mask, dst)
            ok += 1
    print(f"[DONE] masks written={ok}, errors={err}")
    return ok, err


def _renumber_pad_width(n: int) -> int:
    if n <= 0:
        return 3
    return max(3, len(str(n - 1)))


def _renumber_sorted_image_paths(d: Path, exts: set[str]) -> list[Path]:
    files = [p for p in d.iterdir() if p.is_file() and p.suffix in exts]
    files.sort(key=lambda p: p.stem.lower())
    return files


def _renumber_sorted_new_names(mapping: dict[str, str]) -> list[str]:
    if not mapping:
        return []
    return sorted(mapping.values(), key=lambda n: int(Path(n).stem))


def _renumber_two_phase_rename(pairs: list[tuple[Path, str]], dry_run: bool) -> None:
    if dry_run:
        for p, new_name in pairs:
            print(f"[DRY] {p.name} -> {new_name}")
        return
    tmps: list[tuple[Path, str]] = []
    for i, (p, new_name) in enumerate(pairs):
        tmp = p.with_name(f".__renseq_{i:05d}{p.suffix}")
        p.rename(tmp)
        tmps.append((tmp, new_name))
    for tmp, new_name in tmps:
        tmp.rename(tmp.parent / new_name)


def run_sequential_renumber(out_root: Path, dry_run: bool) -> None:
    train_good = out_root / "train" / "good"
    test_good = out_root / "test" / "good"
    test_tu = out_root / "test" / "torned-up"
    mask_dir = out_root / "mask"
    gt_tu = out_root / "ground_truth" / "torned-up"
    ann = out_root / "annotation"
    ann_mask_json = ann / "mask_json"
    ann_defect_json = ann / "defect_json"

    def renumber_simple(dir_path: Path, kind: str) -> dict[str, str]:
        if not dir_path.is_dir():
            print(f"[SKIP] {kind}: directory missing {dir_path}", file=sys.stderr)
            return {}
        paths = _renumber_sorted_image_paths(dir_path, REN_IMAGE_EXTS)
        if not paths:
            print(f"[SKIP] {kind}: no files in {dir_path}", file=sys.stderr)
            return {}
        w = _renumber_pad_width(len(paths))
        mapping: dict[str, str] = {}
        pairs: list[tuple[Path, str]] = []
        for i, p in enumerate(paths):
            new_name = f"{i:0{w}d}{p.suffix.lower()}"
            mapping[p.name] = new_name
            pairs.append((p, new_name))
        print(f"[INFO] Renumber {kind}: {len(pairs)} files in {dir_path}")
        _renumber_two_phase_rename(pairs, dry_run)
        return mapping

    def renumber_paired_images_masks(img_dir: Path, mask_d: Path, label: str) -> tuple[dict[str, str], dict[str, str]]:
        img_map: dict[str, str] = {}
        mask_map: dict[str, str] = {}
        if not img_dir.is_dir() or not mask_d.is_dir():
            print(f"[SKIP] paired {label}: missing dir", file=sys.stderr)
            return img_map, mask_map
        imgs = _renumber_sorted_image_paths(img_dir, REN_IMAGE_EXTS)
        masks = sorted([p for p in mask_d.iterdir() if p.is_file() and p.suffix.lower() == ".png"], key=lambda p: p.stem.lower())
        n = min(len(imgs), len(masks))
        if len(imgs) != len(masks):
            print(f"[WARN] {label}: image count {len(imgs)} != mask count {len(masks)}; pairing first {n}", file=sys.stderr)
        if n == 0:
            return img_map, mask_map
        w = _renumber_pad_width(n)
        pairs: list[tuple[Path, str]] = []
        for i in range(n):
            img, m = imgs[i], masks[i]
            new_img = f"{i:0{w}d}{img.suffix.lower()}"
            new_mask = f"{i:0{w}d}.png"
            img_map[img.name] = new_img
            mask_map[m.name] = new_mask
            pairs.append((img, new_img))
            pairs.append((m, new_mask))
        print(f"[INFO] Renumber paired {label}: {n} image+mask pairs")
        _renumber_two_phase_rename(pairs, dry_run)
        return img_map, mask_map

    def renumber_labelme_json_dir(json_dir: Path, img_mapping: dict[str, str], label: str) -> None:
        if not json_dir.is_dir() or not img_mapping:
            return
        json_paths = sorted([p for p in json_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"], key=lambda p: p.stem.lower())
        old_img_names = sorted(img_mapping.keys(), key=lambda n: Path(n).stem.lower())
        n = min(len(json_paths), len(old_img_names))
        if n == 0:
            return
        if len(json_paths) != len(old_img_names):
            print(f"[WARN] {label}: json count {len(json_paths)} != image mapping {len(old_img_names)}", file=sys.stderr)
        wj = _renumber_pad_width(n)
        pairs: list[tuple[Path, str]] = []
        for i in range(n):
            jp = json_paths[i]
            old_img = old_img_names[i]
            new_img = img_mapping.get(old_img)
            if not new_img:
                continue
            data = json.loads(jp.read_text(encoding="utf-8"))
            data["imagePath"] = new_img
            new_json = f"{i:0{wj}d}.json"
            if not dry_run:
                jp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            pairs.append((jp, new_json))
        print(f"[INFO] Renumber {label}: {len(pairs)} json files")
        _renumber_two_phase_rename(pairs, dry_run)

    img_map_train, _ = renumber_paired_images_masks(train_good, mask_dir, "train+mask")
    map_test_good = renumber_simple(test_good, "test/good")
    img_map_tu, _ = renumber_paired_images_masks(test_tu, gt_tu, "test_torned-up+gt")

    if img_map_train:
        renumber_labelme_json_dir(ann_mask_json, img_map_train, "annotation/mask_json")
    if img_map_tu:
        renumber_labelme_json_dir(ann_defect_json, img_map_tu, "annotation/defect_json")

    def update_merged(path: Path, img_mapping: dict[str, str]) -> None:
        if not path.is_file() or not img_mapping:
            return
        bundle = json.loads(path.read_text(encoding="utf-8"))
        if bundle.get("format") != BUNDLE_FORMAT:
            print(f"[WARN] {path} unexpected format", file=sys.stderr)
        changed = 0
        for item in bundle.get("items", []):
            ip = item.get("imagePath")
            if not ip:
                continue
            bn = Path(ip).name
            if bn in img_mapping:
                item["imagePath"] = img_mapping[bn]
                changed += 1
        if dry_run:
            print(f"[DRY] update {path}: would rewrite {changed} imagePath entries")
        else:
            path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[INFO] Updated {path} ({changed} imagePath)")

    update_merged(ann / TRAIN_MASK_JSON, img_map_train)
    update_merged(ann / TEST_TORNED_JSON, img_map_tu)

    def write_manifest(rel_name: str, mapping: dict[str, str], dir_path: Path) -> None:
        mpath = ann / rel_name
        if not mapping:
            return
        lines = _renumber_sorted_new_names(mapping)
        if dry_run:
            print(f"[DRY] rewrite {mpath} ({len(lines)} lines)")
            return
        if not dir_path.is_dir():
            return
        mpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[INFO] Wrote {mpath}")

    write_manifest("train_good.txt", img_map_train, train_good)
    write_manifest("test_good.txt", map_test_good, test_good)
    write_manifest("test_torned-up.txt", img_map_tu, test_tu)

    print("[DONE] sequential renumber finished")


def main() -> None:
    root = _project_root()
    default_raw = root / "datasets" / "raw" / "InsPLAD"
    default_out = root / "datasets" / "InsPLAD" / "polymer-insulator"

    parser = argparse.ArgumentParser(
        description="Build InsPLAD polymer-insulator subset: copy images, masks from merged JSON, renumber (all on by default).",
    )
    parser.add_argument("--raw-root", type=Path, default=default_raw, help=f"Path to unpacked InsPLAD root (default: {default_raw})")
    parser.add_argument(
        "--polymer-subdir",
        type=str,
        default="",
        help="Optional: only index images under raw-root/<this> (recursive). Default empty = index entire raw-root tree.",
    )
    parser.add_argument("--out", type=Path, default=default_out, help=f"Output polymer-insulator root (default: {default_out})")
    parser.add_argument("--clean", action="store_true", help="Remove existing images in train/test folders before copying")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without writing files")
    parser.add_argument("--no-images", action="store_true", help="Skip copying train/test images from raw InsPLAD")
    parser.add_argument("--no-masks", action="store_true", help="Skip generating mask and ground_truth PNGs")
    parser.add_argument("--clean-masks", action="store_true", help="Before writing masks, remove existing PNG files in mask and ground_truth")
    parser.add_argument("--no-renumber", action="store_true", help="Skip renaming files to 000.*, 001.* and updating manifests/JSON")
    args = parser.parse_args()

    exit_code = 0
    out = args.out.resolve()

    if not args.no_images:
        _, total_miss = run_copy_images(args)
        if total_miss:
            exit_code = 2

    if not args.no_masks:
        _, m_err = run_masks_from_merged(out, args.dry_run, args.clean_masks)
        if m_err:
            exit_code = max(exit_code, 3)

    if not args.no_renumber:
        run_sequential_renumber(out, args.dry_run)

    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

