# TotalSegmentator CT Dataset Audit

Generated: `2026-08-18T07:26:38.431908+00:00`

## Audit scope

- Search root: `F:\csstdy\Totalsegmentator_dataset_v201\s0000`
- Detected dataset root: `F:\csstdy\Totalsegmentator_dataset_v201`
- Integrity mode: `header`
- Integrity coverage: Every NIfTI header plus gzip magic/trailer and advertised uncompressed size; CRC/payload not fully decompressed

## Inventory

| Metric | Result |
|---|---:|
| Subjects | 1 |
| CT files | 1 |
| Valid CT headers | 1 |
| Mask files | 117 |
| Unique mask labels | 117 |
| Metadata rows | 1,228 |
| Compressed NIfTI size | 16.11 MiB |

## Findings

- Errors: **0**
- Warnings: **1,227**

| Category | Count |
|---|---:|
| `metadata_subject_missing_on_disk` | 1,227 |

## Mask coverage

All 117 discovered labels are listed in `mask_inventory.csv`.

## Generated files

- `summary.json`: concise machine-readable summary
- `subjects.csv`: per-subject CT geometry, mask count, metadata, and issue count
- `mask_inventory.csv`: per-label subject coverage
- `issues.csv`: actionable error/warning list with exact paths

## Integrity note

Header mode checks every file's readable NIfTI header, gzip signature/trailer, advertised uncompressed size, dimensions, spacing, datatype, and CT-mask geometry. It does not prove the CRC of every compressed payload. Run the same command with `--integrity-mode full` for exhaustive decompression/CRC checking; that operation can take many hours and read terabytes of uncompressed data.

Audit runtime: 0.06 seconds using 1 workers.
