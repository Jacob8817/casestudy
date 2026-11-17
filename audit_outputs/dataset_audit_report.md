# TotalSegmentator CT Dataset Audit

Generated: `2026-08-18T07:30:33.998389+00:00`

## Audit scope

- Search root: `F:\csstdy`
- Detected dataset root: `F:\csstdy\Totalsegmentator_dataset_v201`
- Integrity mode: `header`
- Integrity coverage: Every NIfTI header plus gzip magic/trailer and advertised uncompressed size; CRC/payload not fully decompressed

## Inventory

| Metric | Result |
|---|---:|
| Subjects | 1,228 |
| CT files | 1,228 |
| Valid CT headers | 1,228 |
| Mask files | 143,676 |
| Unique mask labels | 117 |
| Metadata rows | 1,228 |
| Compressed NIfTI size | 30.10 GiB |
| Distinct CT matrix shapes | 1,207 |
| CT datatype codes | 4: 1,228 |

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

- `summary.json`: concise machine-readable summary
- `subjects.csv`: per-subject CT geometry, mask count, metadata, and issue count
- `mask_inventory.csv`: per-label subject coverage
- `issues.csv`: actionable error/warning list with exact paths

## Integrity note

Header mode checks every file's readable NIfTI header, gzip signature/trailer, advertised uncompressed size, dimensions, spacing, datatype, and CT-mask geometry. It does not prove the CRC of every compressed payload. Run the same command with `--integrity-mode full` for exhaustive decompression/CRC checking; that operation can take many hours and read terabytes of uncompressed data.

Audit runtime: 169.12 seconds using 8 workers.
