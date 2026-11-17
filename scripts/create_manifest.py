#!/usr/bin/env python3
"""Create the official-split selected-organ dataset manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.label_mapping import load_organ_schema  # noqa: E402
from src.dataset.manifest import build_manifest, write_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-root", "--dataset_root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--organ-config", type=Path, default=PROJECT_ROOT / "configs" / "organs_v1.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "splits" / "dataset_manifest.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-file-hashes", action="store_true", help="Faster development manifest without content locking")
    parser.add_argument("--overwrite", action="store_true", help="Deliberately replace an existing manifest")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    try:
        schema = load_organ_schema(args.organ_config)
        manifest = build_manifest(
            args.dataset_root,
            schema,
            args.organ_config.resolve().relative_to(PROJECT_ROOT.resolve()),
            workers=args.workers,
            hash_files=not args.skip_file_hashes,
        )
        write_manifest(args.output, manifest, overwrite=args.overwrite)
    except (OSError, ValueError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "case_count": manifest["dataset"]["case_count"],
                "split_counts": manifest["split_policy"]["counts"],
                "foreground_classes": manifest["organ_schema"]["foreground_classes"],
                "source_masks_per_case": manifest["organ_schema"]["source_mask_count_per_case"],
                "test_locked": manifest["split_policy"]["test_lock"]["locked"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
