#!/usr/bin/env python3
"""Run the standalone FIQA analysis stages in their required order.

Score extraction is intentionally not repeated here because it can take a long
time and depends on the user's selected CR-FIQA backbone. Run extract_scores.py
once before this orchestrator.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGES = [
    "run_eda.py",
    "train_models.py",
    "optimize_models.py",
    "correlation_analysis.py",
    "regression_analysis.py",
    "explainability.py",
    "demographic_consistency.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FIQA analysis scripts sequentially.")
    parser.add_argument("--config", type=Path, default=Path("config/runtime_config.json"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--optimization-mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument(
        "--skip",
        nargs="*",
        choices=[Path(stage).stem for stage in STAGES],
        default=[],
        help="Stage names to skip, for example --skip run_eda explainability.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scripts_dir = Path(__file__).resolve().parent
    config = args.config.expanduser().resolve()
    data_arguments = ["--data", str(args.data.expanduser().resolve())] if args.data else []

    for stage in STAGES:
        stage_name = Path(stage).stem
        if stage_name in args.skip:
            print(f"Skipping: {stage_name}")
            continue

        command = [
            sys.executable,
            str(scripts_dir / stage),
            "--config",
            str(config),
            *data_arguments,
        ]

        if stage == "train_models.py":
            command.extend(["--n-jobs", str(args.n_jobs)])
            if args.skip_xgboost:
                command.append("--skip-xgboost")
        elif stage == "optimize_models.py":
            command.extend(
                [
                    "--mode",
                    args.optimization_mode,
                    "--n-jobs",
                    str(args.n_jobs),
                ]
            )
            if args.skip_xgboost:
                command.extend(["--models", "random_forest", "gradient_boosting"])
        elif stage == "explainability.py":
            command.extend(["--n-jobs", str(args.n_jobs)])
            if args.skip_shap:
                command.append("--skip-shap")

        print("\n" + "=" * 88)
        print(f"Running stage: {stage_name}")
        print("Command:", " ".join(command))
        print("=" * 88)
        completed = subprocess.run(command, cwd=scripts_dir.parent)
        if completed.returncode != 0:
            print(f"Pipeline stopped because {stage_name} failed.", file=sys.stderr)
            return completed.returncode

    print("\nAll requested FIQA analysis stages completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
