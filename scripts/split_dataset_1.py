#!/usr/bin/env python3

from __future__ import annotations

import random
import shutil
from pathlib import Path


# Hardcoded settings (edit these if needed)
TRAIN_FRACTION = 0.7
SEED = 0


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "1"
    dst = root / "dataset"
    train_dir = dst / "train"
    test_dir = dst / "test"

    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")

    # IDs are inferred from files like: s000004_experimental_curve.dat
    exp_ids = []
    for p in src.glob("*_experimental_curve.dat"):
        exp_ids.append(p.name.removesuffix("_experimental_curve.dat"))

    exp_ids = sorted(set(exp_ids))
    if not exp_ids:
        raise SystemExit(f"No *_experimental_curve.dat files found in {src}")

    # Keep only IDs that have all three required files
    kept: list[str] = []
    skipped = 0
    for exp_id in exp_ids:
        e = src / f"{exp_id}_experimental_curve.dat"
        t = src / f"{exp_id}_theoretical_curve.dat"
        m = src / f"{exp_id}_model.txt"
        if e.exists() and t.exists() and m.exists():
            kept.append(exp_id)
        else:
            skipped += 1

    if not kept:
        raise SystemExit("No complete (experimental+theoretical+model) trios found")
    if skipped:
        print(f"Warning: skipped {skipped} incomplete experiment IDs")

    rng = random.Random(SEED)
    rng.shuffle(kept)

    n_total = len(kept)
    n_train = int(n_total * TRAIN_FRACTION)
    train_ids = sorted(kept[:n_train])
    test_ids = sorted(kept[n_train:])

    print(f"Found {n_total} complete experiment IDs in {src}")
    print(f"Split: train={len(train_ids)} ({len(train_ids)/n_total:.1%}), test={len(test_ids)} ({len(test_ids)/n_total:.1%})")
    print(f"Writing to: {dst}")

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    def copy_trio(exp_id: str, out_dir: Path) -> None:
        shutil.copy2(src / f"{exp_id}_experimental_curve.dat", out_dir / f"{exp_id}_experimental_curve.dat")
        shutil.copy2(src / f"{exp_id}_theoretical_curve.dat", out_dir / f"{exp_id}_theoretical_curve.dat")
        shutil.copy2(src / f"{exp_id}_model.txt", out_dir / f"{exp_id}_model.txt")

    for exp_id in train_ids:
        copy_trio(exp_id, train_dir)
    for exp_id in test_ids:
        copy_trio(exp_id, test_dir)


if __name__ == "__main__":
    main()
