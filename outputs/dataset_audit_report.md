# TotalSegmentator CT Dataset Audit

Generated: `2026-08-18T07:53:30.088300+00:00`

## Audit scope

- Search root: `F:\csstdy\Totalsegmentator_dataset_v201`
- Detected dataset root: `F:\csstdy\Totalsegmentator_dataset_v201`
- Integrity mode: `header`
- CT integrity coverage: Every CT payload streamed to EOF with gzip CRC validation while calculating scaled intensity statistics
- Mask integrity coverage: Every mask NIfTI header, gzip signature/trailer, advertised uncompressed size, datatype, and CT-grid geometry; payload CRC not checked

## Inventory

| Metric | Result |
|---|---:|
| Subjects | 1,228 |
| CT files | 1,228 |
| Valid CT headers | 1,228 |
| CT payloads with HU statistics | 1,228 |
| Mask files | 143,676 |
| Unique mask labels | 117 |
| Metadata rows | 1,228 |
| Compressed NIfTI size | 30.10 GiB |
| Distinct CT matrix shapes | 1,207 |
| CT datatype codes | 4: 1,228 |
| CT orientations | RAS: 1,228 |

## Findings

- Errors: **0**
- Warnings: **0**
- No file, header, grid-geometry, or metadata-ID inconsistencies were detected.

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
| Global minimum HU | -28526 |
| Global maximum HU | 30571 |
| Voxel-weighted mean HU | -593.47 |
| Voxel-weighted standard deviation HU | 657.592 |
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

The default run reads every CT payload through gzip EOF, verifies its CRC, and computes HU statistics. Mask files receive header/trailer and CT-grid checks by default. Use `--integrity-mode full` for exhaustive mask decompression/CRC checking; these masks advertise 2.06 TiB uncompressed and this operation can take many hours.

Audit runtime: 301.57 seconds using 8 workers.
