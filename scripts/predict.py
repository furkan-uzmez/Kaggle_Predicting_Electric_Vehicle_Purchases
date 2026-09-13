#!/usr/bin/env python3
"""CLI script for final model retraining, prediction generation, and submission validation."""

import argparse
import logging
import sys
from pathlib import Path

from ev_purchase.submission import generate_final_model_and_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ev_purchase.predict")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain locked final policy on full dataset and generate validated submission."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/final.yaml",
        help="Path to final model YAML configuration file (default: configs/final.yaml)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional custom output root directory path",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable full traceback logging on error",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        logger.info(f"Starting final retraining and prediction with config: {config_path}")
        result = generate_final_model_and_submission(
            config_path=config_path,
            output_root=args.output_root,
        )
        logger.info(f"Successfully generated submission: {result['submission_path']}")
        logger.info(f"Validation report saved: {result['validation_report_path']}")
        logger.info(f"Final model directory: {result['final_model_dir']}")
        logger.info(f"Submission SHA-256: {result['sha256']}")
        logger.info(f"Target stats: {result['target_stats']}")
        logger.info(f"Completed in {result['runtime_seconds']} seconds.")
    except Exception as exc:
        if args.debug:
            logger.error(f"Final prediction failed: {exc}", exc_info=True)
        else:
            logger.error(f"Final prediction failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
