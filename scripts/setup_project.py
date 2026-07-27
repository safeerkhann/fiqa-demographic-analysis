#!/usr/bin/env python3
"""
Create and validate a runtime configuration for the FIQA demographic analysis project.

This script does not assume:
- a specific operating system,
- Google Colab,
- Google Drive,
- a fixed project directory,
- a fixed dataset location,
- a fixed annotation filename,
- or a fixed CR-FIQA checkpoint filename.

Example
-------
python scripts/setup_project.py \
    --project-dir /path/to/fiqa-project \
    --images /path/to/diveface/images \
    --annotations /path/to/annotations.pkl \
    --checkpoint /path/to/backbone.pth \
    --output-config config/runtime_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANNOTATION_EXTENSIONS = {".pkl", ".pickle", ".csv", ".parquet"}
CHECKPOINT_EXTENSIONS = {".pth", ".pt", ".ckpt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate project inputs and create a reusable runtime "
            "configuration file."
        )
    )

    parser.add_argument(
        "--project-dir",
        type=Path,
        required=True,
        help="Root directory for project outputs and configuration.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing the DiveFace images.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to the annotation file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to any compatible CR-FIQA checkpoint.",
    )
    parser.add_argument(
        "--cr-fiqa-dir",
        type=Path,
        default=None,
        help="Optional path to an existing CR-FIQA source directory.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("config/runtime_config.json"),
        help=(
            "Path of the JSON configuration file to create. "
            "Default: config/runtime_config.json"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Optional results directory. "
            "Default: <project-dir>/results"
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help=(
            "Optional models directory. "
            "Default: <project-dir>/models"
        ),
    )
    parser.add_argument(
        "--create-output-dirs",
        action="store_true",
        help="Create the results and models directories if they do not exist.",
    )
    parser.add_argument(
        "--skip-image-count",
        action="store_true",
        help="Skip recursive image counting for faster validation.",
    )

    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    """Expand '~' and convert a path to an absolute resolved path."""
    return path.expanduser().resolve()


def validate_directory(
    path: Path,
    label: str,
    *,
    create: bool = False,
) -> Path:
    path = normalize_path(path)

    if create:
        path.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")

    return path


def validate_file(
    path: Path,
    label: str,
    allowed_extensions: set[str] | None = None,
) -> Path:
    path = normalize_path(path)

    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")

    if allowed_extensions and path.suffix.lower() not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise ValueError(
            f"{label} must use one of these extensions: {allowed}. "
            f"Received: {path.suffix or '<no extension>'}"
        )

    return path


def count_files_by_extension(
    directory: Path,
    extensions: Iterable[str],
) -> int:
    normalized_extensions = {extension.lower() for extension in extensions}

    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_extensions
    )


def resolve_output_path(path: Path) -> Path:
    """Resolve the configuration output path.

    Relative paths are resolved from the directory in which the command is
    executed. This makes the default ``config/runtime_config.json`` land in the
    cloned repository when the script is run from the repository root.
    """
    return path.expanduser().resolve()


def build_configuration(args: argparse.Namespace) -> dict:
    project_dir = validate_directory(
        args.project_dir,
        "Project directory",
    )
    image_dir = validate_directory(
        args.images,
        "Image directory",
    )
    annotation_file = validate_file(
        args.annotations,
        "Annotation file",
        ANNOTATION_EXTENSIONS,
    )
    checkpoint = validate_file(
        args.checkpoint,
        "Checkpoint",
        CHECKPOINT_EXTENSIONS,
    )

    cr_fiqa_dir = None
    if args.cr_fiqa_dir is not None:
        cr_fiqa_dir = validate_directory(
            args.cr_fiqa_dir,
            "CR-FIQA directory",
        )

    results_dir = (
        normalize_path(args.results_dir)
        if args.results_dir is not None
        else project_dir / "results"
    )
    models_dir = (
        normalize_path(args.models_dir)
        if args.models_dir is not None
        else project_dir / "models"
    )

    if args.create_output_dirs:
        results_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

    image_count = None
    if not args.skip_image_count:
        image_count = count_files_by_extension(
            image_dir,
            IMAGE_EXTENSIONS,
        )
        if image_count == 0:
            raise FileNotFoundError(
                "No supported image files were found recursively in: "
                f"{image_dir}"
            )

    return {
        "project_dir": str(project_dir),
        "image_dir": str(image_dir),
        "annotation_file": str(annotation_file),
        "checkpoint": str(checkpoint),
        "cr_fiqa_dir": (
            str(cr_fiqa_dir)
            if cr_fiqa_dir is not None
            else None
        ),
        "results_dir": str(results_dir.resolve()),
        "models_dir": str(models_dir.resolve()),
        "image_count": image_count,
    }


def save_configuration(
    configuration: dict,
    output_path: Path,
) -> Path:
    output_path = resolve_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    return output_path


def main() -> int:
    args = parse_args()

    try:
        configuration = build_configuration(args)

        output_path = save_configuration(
            configuration,
            args.output_config,
        )

    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"File-system error: {error}", file=sys.stderr)
        return 1

    print("Setup completed successfully.")
    print(f"Configuration saved to: {output_path}")
    print(f"Project directory:       {configuration['project_dir']}")
    print(f"Image directory:         {configuration['image_dir']}")
    print(f"Annotation file:         {configuration['annotation_file']}")
    print(f"Checkpoint:              {configuration['checkpoint']}")
    print(f"Results directory:       {configuration['results_dir']}")
    print(f"Models directory:        {configuration['models_dir']}")

    if configuration["image_count"] is not None:
        print(f"Images found:            {configuration['image_count']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
