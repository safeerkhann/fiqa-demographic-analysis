#!/usr/bin/env python3
"""
Extract CR-FIQA scores and merge them with the annotation table.

The script reads paths created by ``scripts/setup_project.py`` from a JSON
configuration file. Paths and model choices can also be overridden through
command-line arguments.

Nothing in this script assumes:
- Google Colab or Google Drive,
- a fixed project directory,
- a fixed checkpoint filename,
- a fixed CR-FIQA repository location,
- or one specific iResNet backbone.

Example
-------
python scripts/extract_scores.py \
    --config config/runtime_config.json \
    --architecture iresnet100 \
    --batch-size 32
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHECKPOINT_KEYS = (
    "state_dict",
    "model_state_dict",
    "model",
    "backbone",
    "network",
    "net",
)
COMMON_STATE_PREFIXES = (
    "module.",
    "model.",
    "backbone.",
    "network.",
    "net.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CR-FIQA scores for images and merge the scores "
            "with an annotation table."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/runtime_config.json"),
        help=(
            "Runtime JSON created by setup_project.py. "
            "Default: config/runtime_config.json"
        ),
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Optional override for the image directory.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Optional override for the annotation file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional override for the CR-FIQA checkpoint.",
    )
    parser.add_argument(
        "--cr-fiqa-dir",
        type=Path,
        default=None,
        help="Optional override for the CR-FIQA source directory.",
    )
    parser.add_argument(
        "--architecture",
        default="iresnet100",
        help=(
            "Backbone factory name exposed by the selected model module, "
            "for example iresnet18, iresnet34, iresnet50, or iresnet100. "
            "Default: iresnet100"
        ),
    )
    parser.add_argument(
        "--model-module",
        default="backbones.iresnet",
        help=(
            "Python module containing the backbone factory. "
            "Default: backbones.iresnet"
        ),
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=512,
        help="Embedding dimension passed to the backbone. Default: 512",
    )
    parser.add_argument(
        "--quality-heads",
        type=int,
        default=1,
        help=(
            "Value passed as 'qs' to CR-FIQA backbones that support it. "
            "Default: 1"
        ),
    )
    parser.add_argument(
        "--use-se",
        action="store_true",
        help="Enable squeeze-and-excitation blocks when supported.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size. Default: 32",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. Default: auto",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=112,
        help="Square model input size. Default: 112",
    )
    parser.add_argument(
        "--index-column",
        default="index",
        help="Join-key column in the annotation table. Default: index",
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=None,
        help=(
            "Optional output CSV for generated scores. "
            "Default: <results-dir>/01_score_extraction/"
            "diveface_cr_fiqa_scores.csv"
        ),
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=None,
        help=(
            "Optional output CSV for merged data. "
            "Default: <results-dir>/01_score_extraction/"
            "diveface_fiqa_merged.csv"
        ),
    )
    parser.add_argument(
        "--unreadable-output",
        type=Path,
        default=None,
        help=(
            "Optional text file for unreadable images. "
            "Default: <results-dir>/01_score_extraction/"
            "unreadable_images.txt"
        ),
    )
    parser.add_argument(
        "--checkpoint-key",
        default=None,
        help=(
            "Optional explicit key containing the state dictionary "
            "inside a wrapped checkpoint."
        ),
    )
    parser.add_argument(
        "--strict-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require an exact checkpoint/model match. "
            "Use --no-strict-checkpoint only for deliberate partial loading."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of images to process for a smoke test.",
    )

    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    return path.expanduser().resolve()


def load_config(path: Path) -> dict[str, Any]:
    path = normalize_path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Runtime configuration not found: {path}. "
            "Run scripts/setup_project.py first or pass another --config."
        )

    if not path.is_file():
        raise ValueError(f"Configuration path is not a file: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON configuration: {path}") from exc

    if not isinstance(data, dict):
        raise ValueError("Runtime configuration must contain a JSON object.")

    return data


def path_from_config_or_override(
    override: Path | None,
    config: dict[str, Any],
    key: str,
    label: str,
    *,
    required: bool = True,
) -> Path | None:
    raw_value: str | Path | None = override

    if raw_value is None:
        raw_value = config.get(key)

    if raw_value in (None, ""):
        if required:
            raise ValueError(
                f"{label} is missing. Add '{key}' to the configuration "
                f"or pass the corresponding command-line argument."
            )
        return None

    return normalize_path(Path(raw_value))


def validate_directory(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return path


def validate_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def resolve_output_path(
    override: Path | None,
    default_path: Path,
) -> Path:
    path = normalize_path(override) if override is not None else default_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )

    return torch.device(requested)


def import_backbone_factory(
    cr_fiqa_dir: Path,
    module_name: str,
    architecture: str,
):
    root = str(cr_fiqa_dir)
    if root not in sys.path:
        sys.path.insert(0, root)

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Could not import '{module_name}' from {cr_fiqa_dir}. "
            "Check --cr-fiqa-dir and --model-module."
        ) from exc

    factory = getattr(module, architecture, None)

    if factory is None or not callable(factory):
        available = sorted(
            name
            for name in dir(module)
            if name.startswith("iresnet") and callable(getattr(module, name))
        )
        raise ValueError(
            f"Backbone factory '{architecture}' was not found in "
            f"'{module_name}'. Available iResNet factories: "
            f"{available or 'none detected'}"
        )

    return factory


def instantiate_model(
    factory,
    *,
    num_features: int,
    quality_heads: int,
    use_se: bool,
) -> torch.nn.Module:
    """
    Instantiate a CR-FIQA backbone.

    CR-FIQA's modified iResNet accepts ``qs``. A clear error is raised when
    a standard recognition-only backbone is selected because it cannot emit
    the quality score required by this script.
    """
    kwargs = {
        "num_features": num_features,
        "qs": quality_heads,
        "use_se": use_se,
    }

    try:
        model = factory(**kwargs)
    except TypeError as exc:
        raise TypeError(
            "The selected backbone factory does not accept the CR-FIQA "
            "arguments 'num_features', 'qs', and 'use_se'. Use a CR-FIQA "
            "compatible backbone implementation or specify another "
            "--model-module/--architecture."
        ) from exc

    return model


def extract_state_dict(
    checkpoint: Any,
    explicit_key: str | None,
) -> dict[str, torch.Tensor]:
    if explicit_key is not None:
        if not isinstance(checkpoint, dict) or explicit_key not in checkpoint:
            raise KeyError(
                f"Checkpoint key '{explicit_key}' was not found."
            )
        checkpoint = checkpoint[explicit_key]

    elif isinstance(checkpoint, dict):
        for key in CHECKPOINT_KEYS:
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The checkpoint does not contain a recognizable state dictionary."
        )

    tensor_items = {
        str(key): value
        for key, value in checkpoint.items()
        if torch.is_tensor(value)
    }

    if not tensor_items:
        raise ValueError(
            "No tensor parameters were found in the checkpoint state dictionary."
        )

    return tensor_items


def strip_common_prefixes(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cleaned = dict(state_dict)

    for prefix in COMMON_STATE_PREFIXES:
        if cleaned and all(key.startswith(prefix) for key in cleaned):
            cleaned = {
                key[len(prefix):]: value
                for key, value in cleaned.items()
            }

    return cleaned


def load_model(
    *,
    factory,
    checkpoint_path: Path,
    device: torch.device,
    num_features: int,
    quality_heads: int,
    use_se: bool,
    checkpoint_key: str | None,
    strict: bool,
) -> torch.nn.Module:
    model = instantiate_model(
        factory,
        num_features=num_features,
        quality_heads=quality_heads,
        use_se=use_se,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = strip_common_prefixes(
        extract_state_dict(checkpoint, checkpoint_key)
    )

    incompatible = model.load_state_dict(state_dict, strict=strict)

    if not strict:
        if incompatible.missing_keys:
            print(
                "Warning: missing checkpoint keys:",
                incompatible.missing_keys,
                file=sys.stderr,
            )
        if incompatible.unexpected_keys:
            print(
                "Warning: unexpected checkpoint keys:",
                incompatible.unexpected_keys,
                file=sys.stderr,
            )

    model.to(device)
    model.eval()
    return model


def collect_image_paths(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def preprocess_image(
    image_path: Path,
    image_size: int,
) -> np.ndarray | None:
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )

    # CR-FIQA's reference pipeline uses OpenCV BGR channel ordering.
    image = image.transpose(2, 0, 1)
    return image.astype(np.float32)


def extract_quality_tensor(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, (tuple, list)) and len(model_output) >= 2:
        quality = model_output[1]
    elif isinstance(model_output, dict):
        quality = next(
            (
                model_output[key]
                for key in ("quality", "quality_score", "qs", "score")
                if key in model_output
            ),
            None,
        )
        if quality is None:
            raise ValueError(
                "The model output dictionary contains no recognized "
                "quality-score key."
            )
    else:
        raise ValueError(
            "The selected model did not return a CR-FIQA quality score. "
            "Expected a tuple/list '(embedding, quality_score)' or a "
            "dictionary containing a quality-score tensor."
        )

    if not torch.is_tensor(quality):
        raise TypeError("The extracted quality score is not a tensor.")

    return quality


def generate_scores(
    *,
    image_paths: list[Path],
    image_root: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    unreadable: list[str] = []

    with torch.inference_mode():
        for start in tqdm(
            range(0, len(image_paths), batch_size),
            desc="Generating CR-FIQA scores",
        ):
            current_paths = image_paths[start:start + batch_size]
            images: list[np.ndarray] = []
            valid_paths: list[Path] = []

            for image_path in current_paths:
                image = preprocess_image(image_path, image_size)

                if image is None:
                    unreadable.append(str(image_path))
                    continue

                images.append(image)
                valid_paths.append(image_path)

            if not images:
                continue

            batch = torch.from_numpy(np.stack(images)).to(
                device=device,
                dtype=torch.float32,
            )
            batch = batch.div(255.0).sub(0.5).div(0.5)

            quality = extract_quality_tensor(model(batch))
            values = quality.detach().cpu().numpy().reshape(-1)

            if len(values) != len(valid_paths):
                raise RuntimeError(
                    "The number of returned quality scores does not match "
                    "the inference batch size."
                )

            for image_path, score in zip(valid_paths, values):
                rows.append(
                    {
                        "index": image_path.relative_to(
                            image_root
                        ).as_posix(),
                        "file_name": image_path.name,
                        "cr_fiqa_score": float(score),
                    }
                )

    return pd.DataFrame(rows), unreadable


def read_annotations(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(
        "Unsupported annotation format. Use .pkl, .pickle, .csv, or .parquet."
    )


def normalize_index_series(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype(str)
        .str.replace("\\", "/", regex=False)
        .str.strip()
        .str.lstrip("./")
    )

    # Remove common top-level dataset folder names without requiring one.
    known_prefixes = (
        "DiveFace_subset/",
        "DiveFace_subsampled/",
    )
    for prefix in known_prefixes:
        normalized = normalized.str.removeprefix(prefix)

    return normalized


def merge_scores(
    annotations: pd.DataFrame,
    scores: pd.DataFrame,
    index_column: str,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    if index_column not in annotations.columns:
        raise KeyError(
            f"Annotation table does not contain '{index_column}'. "
            f"Available columns: {annotations.columns.tolist()}"
        )

    annotations = annotations.copy()
    scores = scores.copy()

    annotations[index_column] = normalize_index_series(
        annotations[index_column]
    )
    scores["index"] = normalize_index_series(scores["index"])

    annotation_duplicates = int(
        annotations[index_column].duplicated().sum()
    )
    score_duplicates = int(scores["index"].duplicated().sum())

    if annotation_duplicates:
        raise ValueError(
            f"Annotation table contains {annotation_duplicates} duplicate "
            f"'{index_column}' values."
        )
    if score_duplicates:
        raise ValueError(
            f"Score table contains {score_duplicates} duplicate image indices."
        )

    if index_column != "index":
        scores = scores.rename(columns={"index": index_column})

    merged = annotations.merge(
        scores[[index_column, "cr_fiqa_score"]],
        on=index_column,
        how="inner",
        validate="one_to_one",
    )

    matched = len(merged)
    statistics: dict[str, int | float] = {
        "annotation_rows": len(annotations),
        "score_rows": len(scores),
        "matched_rows": matched,
        "unmatched_annotations": len(annotations) - matched,
        "unmatched_scores": len(scores) - matched,
        "score_match_rate": (
            matched / len(scores) if len(scores) else 0.0
        ),
    }

    return merged, statistics


def main() -> int:
    args = parse_args()

    try:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be greater than zero.")
        if args.image_size <= 0:
            raise ValueError("--image-size must be greater than zero.")
        if args.num_features <= 0:
            raise ValueError("--num-features must be greater than zero.")
        if args.quality_heads <= 0:
            raise ValueError("--quality-heads must be greater than zero.")
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be greater than zero.")

        config = load_config(args.config)

        image_dir = validate_directory(
            path_from_config_or_override(
                args.images,
                config,
                "image_dir",
                "Image directory",
            ),
            "Image directory",
        )
        annotation_file = validate_file(
            path_from_config_or_override(
                args.annotations,
                config,
                "annotation_file",
                "Annotation file",
            ),
            "Annotation file",
        )
        checkpoint = validate_file(
            path_from_config_or_override(
                args.checkpoint,
                config,
                "checkpoint",
                "Checkpoint",
            ),
            "Checkpoint",
        )
        cr_fiqa_dir = validate_directory(
            path_from_config_or_override(
                args.cr_fiqa_dir,
                config,
                "cr_fiqa_dir",
                "CR-FIQA directory",
            ),
            "CR-FIQA directory",
        )

        results_dir = path_from_config_or_override(
            None,
            config,
            "results_dir",
            "Results directory",
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        stage_dir = results_dir / "01_score_extraction"
        stage_dir.mkdir(parents=True, exist_ok=True)

        scores_output = resolve_output_path(
            args.scores_output,
            stage_dir / "diveface_cr_fiqa_scores.csv",
        )
        merged_output = resolve_output_path(
            args.merged_output,
            stage_dir / "diveface_fiqa_merged.csv",
        )
        unreadable_output = resolve_output_path(
            args.unreadable_output,
            stage_dir / "unreadable_images.txt",
        )

        device = select_device(args.device)
        print(f"Device: {device}")
        print(f"Architecture: {args.architecture}")
        print(f"Checkpoint: {checkpoint}")

        factory = import_backbone_factory(
            cr_fiqa_dir,
            args.model_module,
            args.architecture,
        )
        model = load_model(
            factory=factory,
            checkpoint_path=checkpoint,
            device=device,
            num_features=args.num_features,
            quality_heads=args.quality_heads,
            use_se=args.use_se,
            checkpoint_key=args.checkpoint_key,
            strict=args.strict_checkpoint,
        )
        print("CR-FIQA model loaded successfully.")

        image_paths = collect_image_paths(image_dir)
        if args.limit is not None:
            image_paths = image_paths[:args.limit]

        if not image_paths:
            raise FileNotFoundError(
                f"No supported images were found in: {image_dir}"
            )

        print(f"Images to process: {len(image_paths)}")

        scores, unreadable = generate_scores(
            image_paths=image_paths,
            image_root=image_dir,
            model=model,
            device=device,
            batch_size=args.batch_size,
            image_size=args.image_size,
        )

        if scores.empty:
            raise RuntimeError("No CR-FIQA scores were generated.")

        scores.to_csv(scores_output, index=False)

        unreadable_output.write_text(
            "\n".join(unreadable),
            encoding="utf-8",
        )

        annotations = read_annotations(annotation_file)
        merged, statistics = merge_scores(
            annotations,
            scores,
            args.index_column,
        )

        if merged.empty:
            raise RuntimeError(
                "The merge produced no rows. Check the annotation join key "
                "and the relative image paths."
            )

        merged.to_csv(merged_output, index=False)

        print("\nExtraction summary")
        print("------------------")
        print(f"Generated scores:       {len(scores)}")
        print(f"Unreadable images:      {len(unreadable)}")
        print(f"Matched rows:           {statistics['matched_rows']}")
        print(
            "Score match rate:       "
            f"{statistics['score_match_rate']:.2%}"
        )
        print(f"Scores saved to:        {scores_output}")
        print(f"Merged data saved to:   {merged_output}")
        print(f"Unreadable image list:  {unreadable_output}")

        if statistics["score_match_rate"] < 0.95:
            print(
                "Warning: fewer than 95% of generated scores matched the "
                "annotation table.",
                file=sys.stderr,
            )

        return 0

    except (
        FileNotFoundError,
        NotADirectoryError,
        ImportError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        print(f"Score extraction failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
