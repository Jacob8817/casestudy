"""Correct the empty-target handling in the completed Phase 6 bootstrap summary.

This never opens dataset files or reruns inference. It derives the corrected
patient-level confidence interval solely from the already committed per-case
test JSON records and records both report hashes for auditability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.pipeline import _bootstrap_macro_dice, atomic_json, canonical_sha256, utc_now


TEST_ROOT = PROJECT_ROOT / "outputs" / "phase6" / "locked_test"
REPORT_PATH = TEST_ROOT / "test_report.json"
CORRECTION_PATH = TEST_ROOT / "bootstrap_empty_target_correction.json"


def main() -> int:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if report.get("status") != "complete_one_shot" or report.get("cases") != 89:
        raise RuntimeError("A complete 89-case one-shot report is required")
    case_paths = sorted((TEST_ROOT / "cases").glob("*.json"))
    if len(case_paths) != 89:
        raise RuntimeError(f"Expected 89 committed case records, found {len(case_paths)}")
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in case_paths]
    corrected = _bootstrap_macro_dice(cases, seed=42, replicates=2000)
    existing = report.get("patient_level_macro_dice_ci95", {})
    if existing == corrected:
        print("Bootstrap summary is already corrected.")
        return 0
    if existing.get("point_estimate") is not None:
        raise RuntimeError("Refusing to replace an already-defined bootstrap estimate")
    before_sha256 = canonical_sha256(report)
    report["patient_level_macro_dice_ci95"] = corrected
    report["reporting_corrections"] = [
        {
            "id": "bootstrap_empty_target_nan_exclusion_v1",
            "corrected_at_utc": utc_now(),
            "scope": "patient-level bootstrap summary only; no inference, prediction, or test-time selection changed",
            "reason": "10 partial-field cases contain none of the selected ground-truth organs and have undefined patient macro Dice",
        }
    ]
    atomic_json(REPORT_PATH, report)
    after = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    correction = {
        "schema_version": 1,
        "status": "applied",
        "applied_at_utc": utc_now(),
        "report_sha256_before": before_sha256,
        "report_sha256_after": canonical_sha256(after),
        "changed_field": "patient_level_macro_dice_ci95",
        "inference_rerun": False,
        "test_data_reopened": False,
        "threshold_or_postprocessing_changed": False,
        "corrected_summary": corrected,
    }
    atomic_json(CORRECTION_PATH, correction)
    print(json.dumps(corrected, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
