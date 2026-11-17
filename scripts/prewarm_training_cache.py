#!/usr/bin/env python3
"""Prewarm and integrity-check the train/validation cache; never execute a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_organ_schema, load_preprocessing_spec, load_validated_manifest
from src.training import load_baseline_config
from src.training.data import prewarm_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prewarm the fingerprinted cache without model, optimizer, or training execution."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline_unet_v1.yaml",
    )
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Exact split member to prewarm; repeat for an ordered subset.",
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--skip-source-hash-verification",
        action="store_true",
        help="Skip expensive comparison of actual NIfTI file hashes to the manifest.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "cache" / "prewarm_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema = load_organ_schema(PROJECT_ROOT / "configs" / "organs_v1.yaml")
    manifest = load_validated_manifest(PROJECT_ROOT / "data" / "splits" / "dataset_manifest.json", schema)
    preprocessing = load_preprocessing_spec(PROJECT_ROOT / "configs" / "dataset.yaml")
    config = load_baseline_config(args.config)
    preprocessing_sha256 = preprocessing.fingerprint(
        manifest_sha256=manifest["manifest_sha256"],
        schema_sha256=schema.fingerprint(),
    )
    locks = {
        "manifest": manifest["manifest_sha256"] == config.manifest_sha256,
        "organ_schema": schema.fingerprint() == config.organ_schema_sha256,
        "preprocessing": preprocessing_sha256 == config.preprocessing_sha256,
    }
    failed = [name for name, passed in locks.items() if not passed]
    if failed:
        raise ValueError(f"Cache prewarm contract locks failed: {failed}")

    def progress(record: dict[str, object]) -> None:
        print(
            f"[{record['position']}] {record['case_id']}: "
            f"cache_hit={record['cache_hit_before_prewarm']} "
            f"seconds={float(record['elapsed_seconds']):.2f}",
            flush=True,
        )

    report = prewarm_cache(
        project_root=PROJECT_ROOT,
        dataset_root=args.dataset_root.resolve(),
        manifest=manifest,
        schema=schema,
        preprocessing=preprocessing,
        preprocessing_sha256=preprocessing_sha256,
        config=config,
        split=args.split,
        case_ids=args.case_ids,
        max_cases=args.max_cases,
        verify_source_hashes=not args.skip_source_hash_verification,
        progress=progress,
    )
    report["contract_locks"] = locks
    report["config_sha256"] = config.config_sha256
    report_path = args.report.resolve()
    try:
        report_path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("Prewarm report must remain inside the project directory") from exc
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(f"Report: {report_path}")
    print(
        f"Completed {report['cases_completed']} case(s); "
        f"hits={report['cache_hits']}, writes={report['cache_misses_written']}, "
        f"elapsed={report['elapsed_seconds']:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
