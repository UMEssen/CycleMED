#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified CLI for CycleMED medical image-translation workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from registration import process_all_cases
from postprocess import reconstruct_niftis
from preprocess import (
    MODALITY_CHOICES,
    NORMALIZATION_CHOICES,
    PERCENTILE_REFERENCE_CHOICES,
    create_cyclegan_dataset,
    resolve_percentile_range,
)


def add_registration_args(parser: argparse.ArgumentParser) -> None:
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


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--modality", choices=MODALITY_CHOICES, required=True)
    parser.add_argument("--normalization", choices=NORMALIZATION_CHOICES, required=True)
    parser.add_argument("--percentile-range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None)
    parser.add_argument("--percentile", type=float, default=None, help="Shortcut for --percentile-range 0 UPPER.")
    parser.add_argument("--percentile-reference", choices=PERCENTILE_REFERENCE_CHOICES, default=None)
    parser.add_argument("--a-min", dest="source_min", metavar="A_MIN", type=float, default=None)
    parser.add_argument("--a-max", dest="source_max", metavar="A_MAX", type=float, default=None)
    parser.add_argument("--b-min", dest="target_min", metavar="B_MIN", type=float, default=None)
    parser.add_argument("--b-max", dest="target_max", metavar="B_MAX", type=float, default=None)
    parser.add_argument("--target-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), required=True)


def add_reconstruction_args(parser: argparse.ArgumentParser) -> None:
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


def make_range(lower: float | None, upper: float | None, domain_name: str) -> tuple[float, float] | None:
    if lower is None and upper is None:
        return None
    if lower is None or upper is None:
        raise ValueError(f"{domain_name} range requires both minimum and maximum values.")
    return lower, upper


def run_register(args: argparse.Namespace) -> None:
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


def run_make_dataset(args: argparse.Namespace) -> None:
    percentile_range_arg = tuple(args.percentile_range) if args.percentile_range is not None else None
    percentile_range = resolve_percentile_range(percentile_range_arg, args.percentile)
    source_range = make_range(args.source_min, args.source_max, "A")
    target_range = make_range(args.target_min, args.target_max, "B")
    source_folder = (
        getattr(args, "dataset_source_folder", None)
        or getattr(args, "registered_source_folder", None)
        or args.source_folder
    )
    source_image = (
        getattr(args, "dataset_source_image", None)
        or getattr(args, "registered_source_image", None)
        or args.source_image
    )
    target_folder = (
        getattr(args, "dataset_target_folder", None)
        or getattr(args, "registered_target_folder", None)
        or args.target_folder
    )
    target_image = (
        getattr(args, "dataset_target_image", None)
        or getattr(args, "registered_target_image", None)
        or args.target_image
    )
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
        source_folder=source_folder,
        source_image=source_image,
        target_folder=target_folder,
        target_image=target_image,
    )


def run_prepare_dataset(args: argparse.Namespace) -> None:
    run_register(args)
    run_make_dataset(args)


def run_reconstruct_nifti(args: argparse.Namespace) -> None:
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register", help="Register source-domain volumes to target-domain space.")
    add_registration_args(register_parser)
    register_parser.set_defaults(func=run_register)

    make_dataset_parser = subparsers.add_parser("make-dataset", help="Create CycleGAN PNG folders.")
    add_dataset_args(make_dataset_parser)
    make_dataset_parser.set_defaults(func=run_make_dataset)

    prepare_parser = subparsers.add_parser("prepare-dataset", help="Register and create CycleGAN PNG folders.")
    add_registration_args(prepare_parser)
    prepare_parser.add_argument("--dataroot", type=Path, required=True, help="Output CycleGAN dataset root.")
    prepare_parser.add_argument("--dataset-source-folder", dest="dataset_source_folder", help=argparse.SUPPRESS)
    prepare_parser.add_argument("--dataset-source-image", dest="dataset_source_image", help=argparse.SUPPRESS)
    prepare_parser.add_argument("--dataset-target-folder", dest="dataset_target_folder", help=argparse.SUPPRESS)
    prepare_parser.add_argument("--dataset-target-image", dest="dataset_target_image", help=argparse.SUPPRESS)
    prepare_parser.add_argument("--train-ratio", type=float, required=True)
    prepare_parser.add_argument("--val-ratio", type=float, required=True)
    prepare_parser.add_argument("--split-seed", "--seed", dest="seed", type=int, required=True, help="Seed for the random train/validation/test split.")
    prepare_parser.add_argument("--keep-every-nth-slice", type=int, default=None)
    black_slice_group = prepare_parser.add_mutually_exclusive_group(required=True)
    black_slice_group.add_argument("--drop-black-slices", dest="drop_black_slices", action="store_true")
    black_slice_group.add_argument("--keep-black-slices", dest="drop_black_slices", action="store_false")
    prepare_parser.add_argument("--modality", choices=MODALITY_CHOICES, required=True)
    prepare_parser.add_argument("--normalization", choices=NORMALIZATION_CHOICES, required=True)
    prepare_parser.add_argument("--percentile-range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None)
    prepare_parser.add_argument("--percentile", type=float, default=None, help="Shortcut for --percentile-range 0 UPPER.")
    prepare_parser.add_argument("--percentile-reference", choices=PERCENTILE_REFERENCE_CHOICES, default=None)
    prepare_parser.add_argument("--a-min", dest="source_min", metavar="A_MIN", type=float, default=None)
    prepare_parser.add_argument("--a-max", dest="source_max", metavar="A_MAX", type=float, default=None)
    prepare_parser.add_argument("--b-min", dest="target_min", metavar="B_MIN", type=float, default=None)
    prepare_parser.add_argument("--b-max", dest="target_max", metavar="B_MAX", type=float, default=None)
    prepare_parser.add_argument("--target-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), required=True)
    prepare_parser.set_defaults(func=run_prepare_dataset)

    reconstruct_parser = subparsers.add_parser("reconstruct-nifti", help="Rebuild NIfTI volumes from result PNGs.")
    add_reconstruction_args(reconstruct_parser)
    reconstruct_parser.set_defaults(func=run_reconstruct_nifti)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
