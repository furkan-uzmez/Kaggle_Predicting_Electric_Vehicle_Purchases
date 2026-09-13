import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from ev_purchase.config import load_config
from ev_purchase.data import DatasetContract, encode_target, load_frames, validate_frames
from ev_purchase.splits import build_split_manifest, validate_split_manifest


def write_split_artifacts_atomically(
    parquet_path: Path,
    json_path: Path,
    manifest: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """Write parquet and json artifacts transactionally, preserving backups on rollback failure."""
    parquet_tmp = parquet_path.with_suffix(".parquet.tmp")
    json_tmp = json_path.with_suffix(".json.tmp")
    parquet_bak = parquet_path.with_suffix(".parquet.bak")
    json_bak = json_path.with_suffix(".json.bak")

    parquet_had_dest = parquet_path.exists()
    json_had_dest = json_path.exists()

    parquet_backed_up = False
    json_backed_up = False
    parquet_replaced = False
    json_replaced = False

    try:
        manifest.to_parquet(parquet_tmp, index=False)
        with json_tmp.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        if parquet_had_dest:
            os.replace(parquet_path, parquet_bak)
            parquet_backed_up = True
        if json_had_dest:
            os.replace(json_path, json_bak)
            json_backed_up = True

        os.replace(parquet_tmp, parquet_path)
        parquet_replaced = True

        os.replace(json_tmp, json_path)
        json_replaced = True

        for p in (parquet_tmp, json_tmp, parquet_bak, json_bak):
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    except Exception as orig_err:
        parquet_restored = False
        json_restored = False
        rollback_failures: list[str] = []

        if parquet_replaced or parquet_backed_up:
            if parquet_path.exists():
                try:
                    parquet_path.unlink()
                except Exception as e:
                    rollback_failures.append(f"Failed to remove new parquet destination: {e}")

            if parquet_backed_up and parquet_bak.exists():
                try:
                    os.replace(parquet_bak, parquet_path)
                    parquet_restored = True
                except Exception as e:
                    rollback_failures.append(f"Failed to restore parquet from backup: {e}")
            elif not parquet_had_dest and not parquet_path.exists():
                parquet_restored = True

        if json_replaced or json_backed_up:
            if json_path.exists():
                try:
                    json_path.unlink()
                except Exception as e:
                    rollback_failures.append(f"Failed to remove new json destination: {e}")

            if json_backed_up and json_bak.exists():
                try:
                    os.replace(json_bak, json_path)
                    json_restored = True
                except Exception as e:
                    rollback_failures.append(f"Failed to restore json from backup: {e}")
            elif not json_had_dest and not json_path.exists():
                json_restored = True

        for tmp_file in (parquet_tmp, json_tmp):
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass

        if parquet_backed_up and parquet_restored and parquet_bak.exists():
            try:
                parquet_bak.unlink()
            except Exception:
                pass
        if json_backed_up and json_restored and json_bak.exists():
            try:
                json_bak.unlink()
            except Exception:
                pass

        if rollback_failures:
            fail_msg = "; ".join(rollback_failures)
            raise RuntimeError(
                f"Atomic write failed ({orig_err}); rollback encountered failure(s): {fail_msg}"
            ) from orig_err

        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset and generate split manifest.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Path to project root directory (defaults to current working directory)",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root else Path.cwd().resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    config = load_config(config_path, project_root=project_root)

    train, test = load_frames(config.data.train, config.data.test)
    contract = DatasetContract(
        id_column=config.data.id_column, target_column=config.data.target_column
    )
    validate_frames(train, test, contract)

    y = encode_target(train[config.data.target_column], contract)
    n_rows = len(y)
    n_splits = config.validation.n_splits
    n_repeats = config.validation.n_repeats
    random_state = config.validation.random_state

    output_dir = config.outputs.root / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_clean = config.validation.strategy.removesuffix("_kfold")
    base_name = f"{strategy_clean}_{n_splits}x{n_repeats}"
    parquet_path = output_dir / f"{base_name}.parquet"
    json_path = output_dir / f"{base_name}.json"

    parquet_exists = parquet_path.exists()
    json_exists = json_path.exists()

    expected_meta = {
        "strategy": config.validation.strategy,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "random_state": random_state,
        "n_rows": n_rows,
        "total_assignments": n_rows * n_repeats,
        "target_column": config.data.target_column,
        "target_mapping": {
            contract.negative_label: 0,
            contract.positive_label: 1,
        },
        "protocol_version": 1,
    }

    if parquet_exists and json_exists:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                existing_meta = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse existing split metadata JSON: {e}") from e

        for k, v in expected_meta.items():
            if existing_meta.get(k) != v:
                raise ValueError(
                    f"Existing split metadata is incompatible or stale for '{k}': "
                    f"expected {v}, got {existing_meta.get(k)}. "
                    f"Remove existing files under outputs/splits/ to regenerate."
                )

        try:
            existing_manifest = pd.read_parquet(parquet_path)
        except Exception as e:
            raise ValueError(f"Failed to read existing split parquet: {e}") from e

        validate_split_manifest(
            existing_manifest, n_rows=n_rows, n_splits=n_splits, n_repeats=n_repeats, y=y
        )

        expected_manifest = build_split_manifest(
            y, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        if not existing_manifest.equals(expected_manifest):
            raise ValueError(
                "Existing split manifest content does not match expected protocol split. "
                "Remove existing files under outputs/splits/ to regenerate."
            )

        rel_parquet = parquet_path.relative_to(project_root)
        rel_json = json_path.relative_to(project_root)
        total_folds = n_splits * n_repeats
        print(f"Reused existing valid split manifest at: {rel_parquet}")
        print(f"Reused existing valid split metadata at: {rel_json}")
        print(
            f"Summary: {total_folds}-fold CV ({n_splits} splits x {n_repeats} repeats), "
            f"{n_repeats} validation assignments per row for {n_rows} rows."
        )
        return

    if parquet_exists or json_exists:
        raise ValueError(
            "Incomplete split manifest artifacts found: missing parquet or json metadata file. "
            "Remove existing files under outputs/splits/ to regenerate."
        )

    manifest = build_split_manifest(
        y=y,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    validate_split_manifest(manifest, n_rows=n_rows, n_splits=n_splits, n_repeats=n_repeats, y=y)

    write_split_artifacts_atomically(parquet_path, json_path, manifest, expected_meta)

    rel_parquet = parquet_path.relative_to(project_root)
    rel_json = json_path.relative_to(project_root)

    total_folds = n_splits * n_repeats
    print(f"Wrote split manifest to: {rel_parquet}")
    print(f"Wrote split metadata to: {rel_json}")
    print(
        f"Summary: {total_folds}-fold CV ({n_splits} splits x {n_repeats} repeats), "
        f"{n_repeats} validation assignments per row for {n_rows} rows."
    )


if __name__ == "__main__":
    main()
