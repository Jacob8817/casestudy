# TotalSegmentator CT Dataset Audit

Generated: `2026-08-18T07:50:27.844800+00:00`

## Audit scope

- Search root: `F:\csstdy\Totalsegmentator_dataset_v201\s0000`
- Detected dataset root: `F:\csstdy\Totalsegmentator_dataset_v201`
- Integrity mode: `header`
- CT integrity coverage: Every CT payload streamed to EOF with gzip CRC validation while calculating scaled intensity statistics
- Mask integrity coverage: Every mask NIfTI header, gzip signature/trailer, advertised uncompressed size, datatype, and CT-grid geometry; payload CRC not checked

## Inventory

| Metric | Result |
|---|---:|
| Subjects | 1 |
| CT files | 1 |
| Valid CT headers | 1 |
| CT payloads with HU statistics | 1 |
| Mask files | 117 |
| Unique mask labels | 117 |
| Metadata rows | 1,228 |
| Compressed NIfTI size | 16.11 MiB |
| Distinct CT matrix shapes | 1 |
| CT datatype codes | 4: 1 |
| CT orientations | RAS: 1 |

## Findings

- Errors: **0**
- Warnings: **1,227**

| Category | Count |
|---|---:|
| `metadata_subject_missing_on_disk` | 1,227 |

## Source metadata splits

These are split labels supplied by `meta.csv`; the audit did not create or modify a research split.

| Split | Subjects |
|---|---:|
| `test` | 89 |
| `train` | 1,082 |
| `val` | 57 |

## Native CT spacing ranges

| Axis | Minimum (mm) | Maximum (mm) |
|---|---:|---:|
| X | 1.5 | 1.5 |
| Y | 1.5 | 1.5 |
| Z | 1.5 | 1.5 |

## CT intensity statistics

Values include NIfTI `scl_slope` / `scl_inter` scaling and are aggregated over finite voxels.

| Metric | Value |
|---|---:|
| Global minimum HU | -1051 |
| Global maximum HU | 1712 |
| Voxel-weighted mean HU | -515.386 |
| Voxel-weighted standard deviation HU | 503.878 |
| Non-finite voxels | 0 |

## Metadata missingness

These are missing descriptive values in `meta.csv`, not missing CT or mask files.

| Column | Missing values |
|---|---:|
| `age` | 1 |
| `gender` | 2 |
| `institute` | 5 |
| `study_type` | 5 |
| `manufacturer` | 16 |
| `scanner_model` | 16 |
| `kvp` | 16 |
| `pathology` | 24 |
| `pathology_location` | 24 |

## Mask coverage

All 117 discovered labels are listed in `mask_inventory.csv`.

## Generated files

- `dataset_summary.json`: machine-readable dataset-level summary
- `dataset_audit.csv`: one row per case with geometry, HU statistics, masks, metadata, and issue count
- `mask_inventory.csv`: per-label subject coverage
- `issues.csv`: actionable error/warning list with exact paths

## Integrity note

The default run reads every CT payload through gzip EOF, verifies its CRC, and computes HU statistics. Mask files receive header/trailer and CT-grid checks by default. Use `--integrity-mode full` for exhaustive mask decompression/CRC checking; the 143,676 masks expand to roughly 2.06 TiB and this operation can take many hours.

Audit runtime: 0.56 seconds using 1 workers.
