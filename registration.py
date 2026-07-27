#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Register paired source/target NIfTI volumes with ANTs SyN.

Input layout:
  curated_root/
    <case_id>/
      <source-folder>/<source-image>
      <target-folder>/<target-image>

Output layout:
  registered_root/
    <case_id>/
      <registered-source-folder>/<registered-source-image>
      <registered-target-folder>/<registered-target-image>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Literal, TypeAlias

import numpy as np

RegistrationType: TypeAlias = Literal["SyN", "SyNRA"]


def run_registration(
    fixed_path: Path,
    moving_path: Path,
    fixed_output_path: Path,
    registered_output_path: Path,
    registration_type: RegistrationType,
    overwrite: bool,
    intensity_threshold: float,
) -> None:
    """Register the source image to the target image and save informative slices."""
    try:
        import ants
    except ImportError as exc:
        raise ImportError("ANTs registration requires antspyx. Install it with `pip install antspyx`.") from exc

    fixed_output_path.parent.mkdir(parents=True, exist_ok=True)
    registered_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and registered_output_path.is_file() and fixed_output_path.is_file():
        print("  Outputs exist -> skipping (overwrite=False).")
        return

    fixed = ants.image_read(str(fixed_path))
    moving = ants.image_read(str(moving_path))

    fixed_rs = ants.resample_image(fixed, fixed.shape, use_voxels=True)
    moving_rs = ants.resample_image(moving, fixed_rs.shape, use_voxels=True)

    transforms = ants.registration(fixed=fixed_rs, moving=moving_rs, type_of_transform=registration_type)
    registered_img = ants.apply_transforms(
        fixed=fixed_rs,
        moving=moving_rs,
        transformlist=transforms["fwdtransforms"],
    )

    fixed_np = fixed_rs.numpy()
    registered_np = registered_img.numpy()

    informative_slices: List[int] = []
    for i in range(fixed_np.shape[2]):
        fixed_informative = np.mean(fixed_np[:, :, i]) > intensity_threshold
        registered_informative = np.mean(registered_np[:, :, i]) > intensity_threshold
        if fixed_informative and registered_informative:
            informative_slices.append(i)

    if not informative_slices:
        informative_slices = list(range(fixed_np.shape[2]))

    fixed_inf_img = ants.from_numpy(
        fixed_np[:, :, informative_slices],
        origin=fixed_rs.origin,
        spacing=fixed_rs.spacing,
        direction=fixed_rs.direction,
    )
    reg_inf_img = ants.from_numpy(
        registered_np[:, :, informative_slices],
        origin=registered_img.origin,
        spacing=registered_img.spacing,
        direction=registered_img.direction,
    )

    ants.image_write(fixed_inf_img, str(fixed_output_path))
    ants.image_write(reg_inf_img, str(registered_output_path))


def process_all_cases(
    curated_base_dir: Path,
    output_base_dir: Path,
    registration_type: RegistrationType,
    overwrite: bool,
    intensity_threshold: float,
    source_folder: str,
    source_image: str,
    target_folder: str,
    target_image: str,
    registered_source_folder: str,
    registered_source_image: str,
    registered_target_folder: str,
    registered_target_image: str,
) -> None:
    """Register all complete case folders in a curated source/target dataset."""
    output_base_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = sorted([p for p in curated_base_dir.iterdir() if p.is_dir()])

    print(f"Curated base: {curated_base_dir}")
    print(f"Output base:  {output_base_dir}")
    print(f"Transform:    {registration_type}")
    print(f"Overwrite:    {overwrite}")
    print(f"Threshold:    {intensity_threshold}")
    print(f"Cases:        {len(case_dirs)}")
    print("Direction:    source domain (moving) -> target domain (fixed)")

    for case_dir in case_dirs:
        case_id = case_dir.name
        fixed_path = case_dir / target_folder / target_image
        moving_path = case_dir / source_folder / source_image

        if not fixed_path.exists():
            print(f"[{case_id}] missing target image: {fixed_path} -> skip")
            continue
        if not moving_path.exists():
            print(f"[{case_id}] missing source image: {moving_path} -> skip")
            continue

        fixed_out = output_base_dir / case_id / registered_target_folder / registered_target_image
        reg_out = output_base_dir / case_id / registered_source_folder / registered_source_image

        print(f"[{case_id}] registering source -> target")
        run_registration(
            fixed_path=fixed_path,
            moving_path=moving_path,
            fixed_output_path=fixed_out,
            registered_output_path=reg_out,
            registration_type=registration_type,
            overwrite=overwrite,
            intensity_threshold=intensity_threshold,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-root", type=Path, required=True, help="Input curated NIfTI root.")
    parser.add_argument("--registered-root", type=Path, required=True, help="Output registered NIfTI root.")
    parser.add_argument("--registration-type", choices=["SyN", "SyNRA"], required=True)
    parser.add_argument("--intensity-threshold", type=float, required=True)
    parser.add_argument("--source-folder", required=True, help="Folder name for source-domain input volumes.")
    parser.add_argument("--source-image", required=True, help="Filename of each source-domain input volume.")
    parser.add_argument("--target-folder", required=True, help="Folder name for target-domain input volumes.")
    parser.add_argument("--target-image", required=True, help="Filename of each target-domain input volume.")
    parser.add_argument("--registered-source-folder", required=True, help="Output folder name for registered source volumes.")
    parser.add_argument("--registered-source-image", required=True, help="Output filename for registered source volumes.")
    parser.add_argument("--registered-target-folder", required=True, help="Output folder name for target reference volumes.")
    parser.add_argument("--registered-target-image", required=True, help="Output filename for target reference volumes.")
    overwrite_group = parser.add_mutually_exclusive_group(required=True)
    overwrite_group.add_argument("--overwrite", dest="overwrite", action="store_true")
    overwrite_group.add_argument("--skip-existing", dest="overwrite", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    process_all_cases(
        curated_base_dir=args.curated_root,
        output_base_dir=args.registered_root,
        registration_type=args.registration_type,
        overwrite=args.overwrite,
        intensity_threshold=args.intensity_threshold,
        source_folder=args.source_folder,
        source_image=args.source_image,
        target_folder=args.target_folder,
        target_image=args.target_image,
        registered_source_folder=args.registered_source_folder,
        registered_source_image=args.registered_source_image,
        registered_target_folder=args.registered_target_folder,
        registered_target_image=args.registered_target_image,
    )


if __name__ == "__main__":
    main()
