#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reconstruct CycleGAN PNG outputs into NIfTI volumes with reference geometry."""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk


def get_reference_nifti(
    reference_root: Path,
    patient_id: str,
    tag: str,
    source_reference_folder: str,
    source_reference_image: str,
    target_reference_folder: str,
    target_reference_image: str,
) -> Path:
    """Locate the reference NIfTI for a generated output tag."""
    if tag == "real_A":
        subfolder = source_reference_folder
        pattern = source_reference_image
    else:
        subfolder = target_reference_folder
        pattern = target_reference_image

    search_path = str(reference_root / patient_id / subfolder / pattern)
    matches = glob.glob(search_path)
    if not matches:
        raise FileNotFoundError(f"No reference NIfTI found for {patient_id}, tag={tag}, pattern={search_path}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple reference NIfTIs found for {patient_id}, tag={tag}: {matches}")
    return Path(matches[0])


def load_png_volume(png_paths: list[tuple[int, Path]]) -> np.ndarray:
    """Load sorted 16-bit PNG slices and stack to a 3D float32 volume in [0, 1]."""
    slices = []
    for _, path in png_paths:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read PNG: {path}")
        slices.append(img.astype(np.float32) / 65535.0)
    return np.stack(slices, axis=0)


def maybe_restore_range(
    volume: np.ndarray,
    tag: str,
    restore_original_range: bool,
    source_range: tuple[float, float] | None,
    target_range: tuple[float, float] | None,
) -> np.ndarray:
    """Optionally map normalized [0, 1] volumes back to approximate modality intensity ranges."""
    if not restore_original_range:
        return volume

    value_range = source_range if tag == "real_A" else target_range
    if value_range is None:
        raise ValueError("Restoring original range requires explicit A and B intensity bounds.")

    vmin, vmax = value_range
    return volume * (vmax - vmin) + vmin


def save_volume_as_nifti(volume_np: np.ndarray, reference_path: Path, out_path: Path) -> None:
    """Save a numpy volume as NIfTI while copying geometry from a reference image."""
    ref_img = sitk.ReadImage(str(reference_path))
    out_img = sitk.GetImageFromArray(volume_np)
    out_img.SetSpacing(ref_img.GetSpacing())
    out_img.SetOrigin(ref_img.GetOrigin())
    out_img.SetDirection(ref_img.GetDirection())
    sitk.WriteImage(out_img, str(out_path))
    print(f"Saved NIfTI: {out_path}")


def collect_slices(images_dir: Path) -> dict[tuple[str, str], list[tuple[int, Path]]]:
    """Collect CycleGAN result PNGs grouped by patient ID and image tag."""
    regex = re.compile(r"^(?P<pid>.+)_slice_(?P<idx>\d+)_(?P<tag>fake_B|real_A|real_B)\.png$")
    groups: dict[tuple[str, str], list[tuple[int, Path]]] = {}

    for fname in os.listdir(images_dir):
        if not fname.endswith(".png"):
            continue
        match = regex.match(fname)
        if not match:
            continue

        key = (match.group("pid"), match.group("tag"))
        groups.setdefault(key, []).append((int(match.group("idx")), images_dir / fname))

    for key in groups:
        groups[key].sort(key=lambda x: x[0])
    return groups


def reconstruct_niftis(
    images_dir: Path,
    reference_root: Path,
    output_dir: Path,
    restore_original_range: bool,
    source_range: tuple[float, float] | None,
    target_range: tuple[float, float] | None,
    source_reference_folder: str,
    source_reference_image: str,
    target_reference_folder: str,
    target_reference_image: str,
) -> None:
    """Reconstruct all supported CycleGAN PNG output groups into NIfTI volumes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = collect_slices(images_dir)

    if not groups:
        print("No matching PNGs found. Check --images-dir and filename pattern.")
        return

    print(f"Found {len(groups)} patient/tag groups to reconstruct.")
    for (patient_id, tag), slice_list in groups.items():
        print(f"\nReconstructing patient {patient_id}, tag={tag}, num_slices={len(slice_list)}")
        volume = load_png_volume(slice_list)
        volume = maybe_restore_range(
            volume,
            tag,
            restore_original_range=restore_original_range,
            source_range=source_range,
            target_range=target_range,
        )
        ref_path = get_reference_nifti(
            reference_root,
            patient_id,
            tag,
            source_reference_folder=source_reference_folder,
            source_reference_image=source_reference_image,
            target_reference_folder=target_reference_folder,
            target_reference_image=target_reference_image,
        )
        save_volume_as_nifti(volume, ref_path, output_dir / f"{patient_id}_{tag}.nii.gz")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True, help="CycleGAN results image folder.")
    parser.add_argument("--reference-root", type=Path, required=True, help="Registered NIfTI root for geometry.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output reconstructed NIfTI folder.")
    parser.add_argument("--source-reference-folder", required=True, help="Reference folder for source-domain geometry.")
    parser.add_argument("--source-reference-image", required=True, help="Reference filename for source-domain geometry.")
    parser.add_argument("--target-reference-folder", required=True, help="Reference folder for target-domain geometry.")
    parser.add_argument("--target-reference-image", required=True, help="Reference filename for target-domain geometry.")
    parser.add_argument("--no-restore-original-range", action="store_true")
    parser.add_argument("--a-min", dest="source_min", metavar="A_MIN", type=float, default=None)
    parser.add_argument("--a-max", dest="source_max", metavar="A_MAX", type=float, default=None)
    parser.add_argument("--b-min", dest="target_min", metavar="B_MIN", type=float, default=None)
    parser.add_argument("--b-max", dest="target_max", metavar="B_MAX", type=float, default=None)
    return parser


def make_range(lower: float | None, upper: float | None, domain_name: str) -> tuple[float, float] | None:
    if lower is None and upper is None:
        return None
    if lower is None or upper is None:
        raise ValueError(f"{domain_name} range requires both minimum and maximum values.")
    return lower, upper


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    source_range = make_range(args.source_min, args.source_max, "A")
    target_range = make_range(args.target_min, args.target_max, "B")
    if not args.no_restore_original_range and (source_range is None or target_range is None):
        raise ValueError("Restoring original range requires --a-min, --a-max, --b-min, and --b-max.")

    reconstruct_niftis(
        images_dir=args.images_dir,
        reference_root=args.reference_root,
        output_dir=args.output_dir,
        restore_original_range=not args.no_restore_original_range,
        source_range=source_range,
        target_range=target_range,
        source_reference_folder=args.source_reference_folder,
        source_reference_image=args.source_reference_image,
        target_reference_folder=args.target_reference_folder,
        target_reference_image=args.target_reference_image,
    )


if __name__ == "__main__":
    main()
