"""
Renumber image files under datasets/Real/Blade30 subdirectories.

For each directory that directly contains image files, rename files in sorted
name order to:
  000.jpg / 001.jpg / ...
or
  000.png / 001.png / ...
keeping each file's original extension family.
"""

from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def iter_dirs_with_images(root: Path) -> list[Path]:
    out: list[Path] = []
    for d in sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda p: str(p).lower()):
        files = [f for f in d.iterdir() if f.is_file() and f.suffix in IMAGE_EXTS]
        if files:
            out.append(d)
    return out


def pad_width(n: int) -> int:
    if n <= 0:
        return 3
    return max(3, len(str(n - 1)))


def renumber_dir(dir_path: Path, dry_run: bool) -> int:
    files = sorted(
        [p for p in dir_path.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )
    if not files:
        return 0

    w = pad_width(len(files))
    pairs: list[tuple[Path, str]] = []
    for i, p in enumerate(files):
        new_name = f"{i:0{w}d}{p.suffix.lower()}"
        pairs.append((p, new_name))

    if dry_run:
        for p, new_name in pairs:
            print(f"[DRY] {p} -> {p.parent / new_name}")
        return len(pairs)

    # Two-phase rename to avoid overwrite collisions.
    temp_pairs: list[tuple[Path, str]] = []
    for i, (src, final_name) in enumerate(pairs):
        tmp = src.with_name(f".__tmp_ren_{i:06d}{src.suffix.lower()}")
        src.rename(tmp)
        temp_pairs.append((tmp, final_name))

    for tmp, final_name in temp_pairs:
        tmp.rename(tmp.parent / final_name)

    return len(pairs)


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Renumber images in all subdirectories of datasets/Real/Blade30.")
    parser.add_argument("--root", type=Path, default=root / "datasets" / "Real" / "Blade30")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = args.root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Directory not found: {data_root}")

    dirs = iter_dirs_with_images(data_root)
    total_files = 0
    for d in dirs:
        n = renumber_dir(d, args.dry_run)
        total_files += n
        print(f"[INFO] {d}: renamed={n}")

    print("[DONE] Renumber finished.")
    print(f"  directories={len(dirs)}")
    print(f"  files={total_files}")


if __name__ == "__main__":
    main()

