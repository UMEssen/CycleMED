#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create a 16-bit PNG CycleGAN dataset from registered source/target NIfTIs."""

from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk

MODALITY_CHOICES = ("mri", "ct")
NORMALIZATION_CHOICES = ("fixed", "max", "percentile")
PERCENTILE_REFERENCE_CHOICES = ("independent", "follow-a", "follow-b")


def find_single_nii(base_folder: Path, patient_id: str, subfolder: str, pattern: str) -> Path:
    search_path = str(base_folder / patient_id / subfolder / pattern)
    matches = sorted(glob.glob(search_path))
    if not matches:
        raise FileNotFoundError(f"No file found for {patient_id} in {subfolder} with pattern {pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"More than one file found for {patient_id} in {subfolder}: {matches}")
    return Path(matches[0])


def preprocess_volume(
    nii_path: Path,
    modality: str,
    source_range: tuple[float, float] | None,
    target_range: tuple[float, float] | None,
    normalization: str,
    percentile_range: tuple[float, float] | None,
    target_size: tuple[int, int],
    keep_every_nth_slice: int | None = None,
) -> np.ndarray:
    """Read, clip, normalize, and resize one NIfTI volume to slices in [0, 1]."""
    data = read_volume(nii_path)

    vmin, vmax = get_normalization_bounds(
        data=data,
        modality=modality,
        normalization=normalization,
        source_range=source_range,
        target_range=target_range,
        percentile_range=percentile_range,
    )

    return normalize_and_resize_volume(
        data=data,
        bounds=(vmin, vmax),
        target_size=target_size,
        keep_every_nth_slice=keep_every_nth_slice,
        source_path=nii_path,
        modality=modality,
    )


def read_volume(nii_path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(nii_path))
    return sitk.GetArrayFromImage(img).astype(np.float32)


def normalize_and_resize_volume(
    data: np.ndarray,
    bounds: tuple[float, float],
    target_size: tuple[int, int],
    keep_every_nth_slice: int | None = None,
    source_path: Path | None = None,
    modality: str = "volume",
) -> np.ndarray:
    """Clip to bounds, scale to [0, 1], and resize each slice."""
    vmin, vmax = bounds
    processed_slices = []
    for i in range(data.shape[0]):
        sl = np.clip(data[i], vmin, vmax)
        sl = (sl - vmin) / (vmax - vmin + 1e-8)
        processed_slices.append(cv2.resize(sl, target_size, interpolation=cv2.INTER_LINEAR))

    if not processed_slices:
        nii_path = source_path if source_path is not None else "<array>"
        raise RuntimeError(f"No slices found in {nii_path} for modality {modality}.")

    out = np.array(processed_slices, dtype=np.float32)
    if keep_every_nth_slice is not None and keep_every_nth_slice > 1:
        out = out[::keep_every_nth_slice]
    return out


def get_normalization_bounds(
    data: np.ndarray,
    modality: str,
    normalization: str,
    source_range: tuple[float, float] | None,
    target_range: tuple[float, float] | None,
    percentile_range: tuple[float, float] | None,
) -> tuple[float, float]:
    """Return intensity bounds for fixed HU/window, MRI max, or MRI percentile normalization."""
    if normalization not in NORMALIZATION_CHOICES:
        raise ValueError(f"Unknown normalization mode: {normalization}")

    if normalization == "fixed":
        if modality == "source":
            if source_range is None:
                raise ValueError("Fixed normalization requires --a-min and --a-max.")
            return source_range
        if modality == "target":
            if target_range is None:
                raise ValueError("Fixed normalization requires --b-min and --b-max.")
            return target_range
        raise ValueError(f"Unknown modality: {modality}")

    values = data[np.isfinite(data)]
    nonzero_values = values[values != 0]
    if nonzero_values.size > 0:
        values = nonzero_values
    if values.size == 0:
        return 0.0, 1.0

    if normalization == "max":
        vmax = float(np.max(values))
        return 0.0, vmax if vmax > 0 else 1.0

    if percentile_range is None:
        raise ValueError("Percentile normalization requires --percentile or --percentile-range LOW HIGH.")

    lower, upper = percentile_range
    lower, upper = normalize_percentile_value(lower), normalize_percentile_value(upper)
    if lower < 0 or upper > 100 or lower >= upper:
        raise ValueError("Percentile bounds must satisfy 0 <= LOW < HIGH <= 100.")

    vmin = float(np.percentile(values, lower))
    vmax = float(np.percentile(values, upper))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def normalize_percentile_value(value: float) -> float:
    """Accept either fraction-style percentiles such as 0.99 or percent-style values such as 99."""
    if 0.0 < value <= 1.0:
        return value * 100.0
    return value


def percentile_range_from_upper(upper: float, lower: float = 0.0) -> tuple[float, float]:
    return lower, upper


def resolve_percentile_range(
    percentile_range: tuple[float, float] | None,
    percentile: float | None,
) -> tuple[float, float] | None:
    if percentile is not None:
        return percentile_range_from_upper(percentile)
    return percentile_range


def validate_modality_normalization(data_modality: str, normalization: str) -> None:
    if data_modality not in MODALITY_CHOICES:
        raise ValueError(f"Unknown data modality: {data_modality}")
    if normalization not in NORMALIZATION_CHOICES:
        raise ValueError(f"Unknown normalization mode: {normalization}")
    if data_modality == "ct" and normalization != "fixed":
        raise ValueError("CT datasets must use --normalization fixed with uniform HU boundaries.")


def paired_drop_black_slices(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop slice i in both domains if either corresponding slice is completely black."""
    if a.shape[0] != b.shape[0]:
        raise RuntimeError(f"Slice count mismatch: A has {a.shape[0]} slices, B has {b.shape[0]} slices.")

    keep_idx = []
    dropped = 0
    for i in range(a.shape[0]):
        if np.all(a[i] == 0) or np.all(b[i] == 0):
            dropped += 1
        else:
            keep_idx.append(i)

    if not keep_idx:
        return a[:0], b[:0], dropped
    return a[keep_idx], b[keep_idx], dropped


def save_to_png(data: np.ndarray, output_dir: Path, modality_dir: str, patient_id: str) -> None:
    output_path = output_dir / modality_dir
    output_path.mkdir(parents=True, exist_ok=True)

    for i, slice_data in enumerate(data):
        png_filename = output_path / f"{patient_id}_slice_{i}.png"
        img_16bit = np.clip(slice_data * 65535.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(png_filename), img_16bit)

    print(f"Saved {len(data)} slices for {patient_id} in {output_path}")


def split_patients(
    patient_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1:
        raise ValueError("train_ratio and val_ratio must be non-negative and sum to <= 1.")

    patient_ids = list(patient_ids)
    random.seed(seed)
    random.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    datasets = {
        "train": patient_ids[:n_train],
        "val": patient_ids[n_train:n_train + n_val],
        "test": patient_ids[n_train + n_val:],
    }
    print(f"Total patients: {n}")
    print(f"Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")
    return datasets


def create_cyclegan_dataset(
    input_folder: Path,
    output_folder: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    data_modality: str,
    source_range: tuple[float, float] | None,
    target_range: tuple[float, float] | None,
    normalization: str,
    percentile_range: tuple[float, float] | None,
    percentile_reference: str | None,
    target_size: tuple[int, int],
    source_folder: str,
    source_image: str,
    target_folder: str,
    target_image: str,
    pairwise_drop_black_slices: bool,
    keep_every_nth_slice: int | None = None,
) -> None:
    """Convert registered NIfTIs into CycleGAN domain folders."""
    validate_modality_normalization(data_modality, normalization)
    if percentile_reference is not None and percentile_reference not in PERCENTILE_REFERENCE_CHOICES:
        raise ValueError(f"Unknown percentile reference: {percentile_reference}")
    if normalization == "fixed" and (source_range is None or target_range is None):
        raise ValueError("Fixed normalization requires --a-min, --a-max, --b-min, and --b-max.")
    if normalization == "percentile" and percentile_range is None:
        raise ValueError("Percentile normalization requires --percentile or --percentile-range LOW HIGH.")
    if normalization == "percentile" and percentile_reference is None:
        raise ValueError("Percentile normalization requires --percentile-reference.")

    patient_ids = sorted([f for f in os.listdir(input_folder) if (input_folder / f).is_dir()])
    datasets = split_patients(patient_ids, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    print(f"Data modality: {data_modality}")
    print(f"Normalization: {normalization}")
    if normalization == "percentile":
        print(f"Percentile reference: {percentile_reference}")

    split_dirs = {
        "train": ("trainA", "trainB"),
        "val": ("valA", "valB"),
        "test": ("testA", "testB"),
    }

    for split_name, ids in datasets.items():
        for patient_id in ids:
            print(f"\nProcessing {patient_id} ({split_name})")
            source_path = find_single_nii(input_folder, patient_id, source_folder, source_image)
            target_path = find_single_nii(input_folder, patient_id, target_folder, target_image)

            if normalization == "percentile" and percentile_reference != "independent":
                source_data = read_volume(source_path)
                target_data = read_volume(target_path)
                reference_data = source_data if percentile_reference == "follow-a" else target_data
                reference_modality = "source" if percentile_reference == "follow-a" else "target"
                bounds = get_normalization_bounds(
                    data=reference_data,
                    modality=reference_modality,
                    normalization=normalization,
                    source_range=source_range,
                    target_range=target_range,
                    percentile_range=percentile_range,
                )
                source_slices = normalize_and_resize_volume(
                    data=source_data,
                    bounds=bounds,
                    target_size=target_size,
                    keep_every_nth_slice=keep_every_nth_slice,
                    source_path=source_path,
                    modality="source",
                )
                target_slices = normalize_and_resize_volume(
                    data=target_data,
                    bounds=bounds,
                    target_size=target_size,
                    keep_every_nth_slice=keep_every_nth_slice,
                    source_path=target_path,
                    modality="target",
                )
            else:
                source_slices = preprocess_volume(
                    source_path,
                    modality="source",
                    source_range=source_range,
                    target_range=target_range,
                    normalization=normalization,
                    percentile_range=percentile_range,
                    target_size=target_size,
                    keep_every_nth_slice=keep_every_nth_slice,
                )
                target_slices = preprocess_volume(
                    target_path,
                    modality="target",
                    source_range=source_range,
                    target_range=target_range,
                    normalization=normalization,
                    percentile_range=percentile_range,
                    target_size=target_size,
                    keep_every_nth_slice=keep_every_nth_slice,
                )

            if pairwise_drop_black_slices:
                source_slices, target_slices, dropped = paired_drop_black_slices(source_slices, target_slices)
                print(f"  Paired black-slice filter: dropped {dropped} slices, kept {source_slices.shape[0]}")

            if source_slices.shape[0] == 0:
                print(f"  WARNING: no paired slices left after filtering for {patient_id} -> skipping patient")
                continue

            dir_a, dir_b = split_dirs[split_name]
            save_to_png(source_slices, output_folder, dir_a, patient_id)
            save_to_png(target_slices, output_folder, dir_b, patient_id)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-root", type=Path, required=True, help="Input registered NIfTI root.")
    parser.add_argument("--dataroot", type=Path, required=True, help="Output CycleGAN dataset root.")
    parser.add_argument("--source-folder", required=True, help="Folder name for source-domain registered volumes.")
    parser.add_argument("--source-image", required=True, help="Filename of each source-domain registered volume.")
    parser.add_argument("--target-folder", required=True, help="Folder name for target-domain registered volumes.")
    parser.add_argument("--target-image", required=True, help="Filename of each target-domain registered volume.")
    parser.add_argument("--train-ratio", type=float, required=True)
    parser.add_argument("--val-ratio", type=float, required=True)
    parser.add_argument("--split-seed", "--seed", dest="seed", type=int, required=True, help="Seed for the random train/validation/test split.")
    parser.add_argument("--keep-every-nth-slice", type=int, default=None)
    black_slice_group = parser.add_mutually_exclusive_group(required=True)
    black_slice_group.add_argument("--drop-black-slices", dest="drop_black_slices", action="store_true")
    black_slice_group.add_argument("--keep-black-slices", dest="drop_black_slices", action="store_false")
    parser.add_argument("--modality", choices=MODALITY_CHOICES, required=True, help="Dataset modality tag.")
    parser.add_argument("--normalization", choices=NORMALIZATION_CHOICES, required=True)
    parser.add_argument("--percentile-range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None)
    parser.add_argument("--percentile", type=float, default=None, help="Shortcut for --percentile-range 0 UPPER.")
    parser.add_argument("--percentile-reference", choices=PERCENTILE_REFERENCE_CHOICES, default=None)
    parser.add_argument("--a-min", dest="source_min", metavar="A_MIN", type=float, default=None)
    parser.add_argument("--a-max", dest="source_max", metavar="A_MAX", type=float, default=None)
    parser.add_argument("--b-min", dest="target_min", metavar="B_MIN", type=float, default=None)
    parser.add_argument("--b-max", dest="target_max", metavar="B_MAX", type=float, default=None)
    parser.add_argument("--target-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), required=True)
    return parser


def make_range(lower: float | None, upper: float | None, domain_name: str) -> tuple[float, float] | None:
    if lower is None and upper is None:
        return None
    if lower is None or upper is None:
        raise ValueError(f"{domain_name} range requires both minimum and maximum values.")
    return lower, upper


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    percentile_range_arg = tuple(args.percentile_range) if args.percentile_range is not None else None
    percentile_range = resolve_percentile_range(percentile_range_arg, args.percentile)
    source_range = make_range(args.source_min, args.source_max, "A")
    target_range = make_range(args.target_min, args.target_max, "B")
    create_cyclegan_dataset(
        input_folder=args.registered_root,
        output_folder=args.dataroot,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        keep_every_nth_slice=args.keep_every_nth_slice,
        pairwise_drop_black_slices=args.drop_black_slices,
        data_modality=args.modality,
        source_range=source_range,
        target_range=target_range,
        normalization=args.normalization,
        percentile_range=percentile_range,
        percentile_reference=args.percentile_reference,
        target_size=tuple(args.target_size),
        source_folder=args.source_folder,
        source_image=args.source_image,
        target_folder=args.target_folder,
        target_image=args.target_image,
    )


if __name__ == "__main__":
    main()
