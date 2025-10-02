
"""
EuroSAT (RGB) → 4-class dataset builder
---------------------------------------
Remaps EuroSAT classes into:
  forest ← ["Forest"]
  water  ← ["River", "SeaLake"]
  urban  ← ["Residential", "Industrial", "Highway"]
  barren ← ["AnnualCrop", "PermanentCrop", "Pasture", "HerbaceousVegetation"]

It then splits into train/val/test folders with your ratios.

USAGE:
  python eurosat_remap_split.py --src "path/to/EuroSAT_RGB" --dst "dataset" \
      --train 0.7 --val 0.15 --test 0.15 --max_per_class 500 --seed 42
"""
import os, argparse, random, shutil, math
from pathlib import Path

DEFAULT_MAPPING = {
    "forest": ["Forest"],
    "water": ["River", "SeaLake"],
    "urban": ["Residential", "Industrial", "Highway"],
    "barren": ["AnnualCrop", "PermanentCrop", "Pasture", "HerbaceousVegetation"]
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def list_images(folder: Path):
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]

def ensure_dirs(dst_root: Path, classes):
    for split in ["train","val","test"]:
        for c in classes:
            (dst_root / split / c).mkdir(parents=True, exist_ok=True)

def copy_subset(files, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = out_dir / src.name
        i = 1
        while dst.exists():
            dst = out_dir / f"{src.stem}_{i}{src.suffix}"
            i += 1
        shutil.copy2(src, dst)

def build_dataset(src_root: Path, dst_root: Path, mapping, ratios, max_per_class, seed):
    random.seed(seed)
    classes = list(mapping.keys())
    ensure_dirs(dst_root, classes)

    collected = {c: [] for c in classes}
    for tgt, sources in mapping.items():
        for src_name in sources:
            folder = src_root / src_name
            if not folder.exists():
                print(f"[WARN] Missing EuroSAT class folder: {folder}")
                continue
            imgs = list_images(folder)
            collected[tgt].extend(imgs)

    for c in classes:
        random.shuffle(collected[c])
        if max_per_class and max_per_class > 0:
            collected[c] = collected[c][:max_per_class]
        print(f"[INFO] {c}: {len(collected[c])} images")

    train_r, val_r, test_r = ratios
    for c in classes:
        files = collected[c]
        n = len(files)
        n_train = int(math.floor(n * train_r))
        n_val = int(math.floor(n * val_r))
        n_test = n - n_train - n_val

        train_files = files[:n_train]
        val_files   = files[n_train:n_train+n_val]
        test_files  = files[n_train+n_val:]

        print(f"[SPLIT] {c}: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

        copy_subset(train_files, dst_root / "train" / c)
        copy_subset(val_files,   dst_root / "val" / c)
        copy_subset(test_files,  dst_root / "test" / c)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to EuroSAT RGB root directory")
    ap.add_argument("--dst", required=True, help="Output dataset root")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--max_per_class", type=int, default=0, help="0 means no cap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mapping_json", type=str, default="", help="Optional custom mapping JSON path")
    args = ap.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"Splits must sum to 1.0. Got {total}")

    if args.mapping_json:
        import json
        with open(args.mapping_json, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    else:
        mapping = DEFAULT_MAPPING

    return args, mapping

def main():
    args, mapping = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        raise SystemExit(f"Source path not found: {src_root}")

    print("[INFO] Using mapping:")
    for k,v in mapping.items():
        print(f"  {k:<7} <- {v}")

    build_dataset(src_root, dst_root, mapping, (args.train, args.val, args.test),
                  args.max_per_class, args.seed)
    print(f"[DONE] Wrote dataset to: {dst_root}")

if __name__ == "__main__":
    main()
