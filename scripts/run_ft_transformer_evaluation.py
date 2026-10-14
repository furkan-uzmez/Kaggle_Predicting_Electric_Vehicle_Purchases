"""Run the fail-closed FT-Transformer evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ev_purchase.ft_transformer_evaluation import (
    CANONICAL_DIR,
    DEFAULT_OUTPUT_DIR,
    FROZEN_SLICES_PATH,
    FT_ARTIFACT_DIR,
    SPLIT_PATH,
    run_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ft-dir", type=Path, default=FT_ARTIFACT_DIR)
    parser.add_argument("--canonical-dir", type=Path, default=CANONICAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-path", type=Path, default=SPLIT_PATH)
    parser.add_argument("--frozen-slices-path", type=Path, default=FROZEN_SLICES_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_evaluation(
        ft_dir=args.ft_dir,
        canonical_dir=args.canonical_dir,
        output_dir=args.output_dir,
        split_path=args.split_path,
        frozen_slices_path=args.frozen_slices_path,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
