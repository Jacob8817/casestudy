#!/usr/bin/env python3
"""Fully audit the selected 10-organ mask subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.label_mapping import load_organ_schema  # noqa: E402
from src.dataset.selected_mask_audit import (  # noqa: E402
    run_selected_audit,
    write_selected_audit_outputs,
    write_selected_audit_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-root", "--dataset_root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "splits" / "dataset_manifest.json")
    parser.add_argument("--organ-config", type=Path, default=PROJECT_ROOT / "configs" / "organs_v1.yaml")
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "outputs" / "selected_organs_audit.csv")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "outputs" / "organ_volume_statistics.json")
    parser.add_argument("--output-report", type=Path, default=PROJECT_ROOT / "outputs" / "selected_organs_report.md")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-voxels", type=int, default=1_048_576)
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_voxels < 1:
        parser.error("--workers and --chunk-voxels must be positive")

    try:
        schema = load_organ_schema(args.organ_config)
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest.get("organ_schema", {}).get("canonical_sha256") != schema.fingerprint():
            raise ValueError("Manifest organ-schema fingerprint does not match the selected config")
        results = run_selected_audit(
            args.dataset_root,
            manifest,
            schema,
            workers=args.workers,
            chunk_voxels=args.chunk_voxels,
        )
        summary = write_selected_audit_outputs(
            results,
            schema,
            manifest,
            args.output_csv.resolve(),
            args.output_json.resolve(),
        )
        write_selected_audit_report(summary, args.output_report.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    console = {
        "case_count": summary["case_count"],
        "source_file_count": summary["source_file_count"],
        "source_crc_checked_count": summary["source_crc_checked_count"],
        "source_binary_valid_count": summary["source_binary_valid_count"],
        "cases_with_valid_merged_metrics": summary["cases_with_valid_merged_metrics"],
        "severity_counts": summary["severity_counts"],
        "test_metrics_redacted": summary["test_policy"]["per_organ_voxel_counts_and_volumes_redacted"],
        "output_csv": str(args.output_csv.resolve()),
        "output_json": str(args.output_json.resolve()),
        "output_report": str(args.output_report.resolve()),
    }
    print(json.dumps(console, indent=2))
    return 1 if summary["severity_counts"].get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
