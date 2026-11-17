# Selected-Organ Dataset Audit

Generated: `2026-08-18T08:31:07.821566+00:00`

## Integrity

- Cases: **1,228**
- Selected source-mask files: **15,964**
- Header-valid: **15,964**
- Geometry-valid: **15,964**
- Binary-valid: **15,964**
- Payload/CRC-valid: **15,964**
- Errors: **0**
- Warnings: **664**

## Coverage

- Empty selected source masks: **3,962**
- Train/validation cases with no selected foreground: **78**
- Empty masks are retained as legitimate partial-field coverage, not classified as corruption.

## Overlap policy

- Method: `official_totalsegmentator_v2_last_label_wins`
- Priority, low to high: `spleen, kidney_right, kidney_left, liver, stomach, pancreas, lung_left, lung_right, aorta, inferior_vena_cava`
- Reference: https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/libs.py
- Raw mask volumes and exclusive post-resolution training volumes are both reported.

## Development-set organ statistics

Test-set voxel counts and volumes are deliberately excluded.

| Organ | Present / 1,139 | Exclusive median mL | P05 mL | P95 mL |
|---|---:|---:|---:|---:|
| `liver` | 869 | 1351.026 | 116.836 | 2286.637 |
| `spleen` | 821 | 175.244 | 17.196 | 437.451 |
| `kidney_left` | 734 | 125.535 | 3.200 | 225.600 |
| `kidney_right` | 718 | 125.540 | 2.752 | 222.151 |
| `stomach` | 839 | 228.771 | 23.220 | 695.082 |
| `pancreas` | 746 | 59.113 | 3.367 | 107.863 |
| `aorta` | 1,027 | 180.482 | 10.985 | 405.068 |
| `inferior_vena_cava` | 905 | 57.004 | 7.661 | 110.118 |
| `lung_left` | 977 | 1315.572 | 76.845 | 2988.002 |
| `lung_right` | 981 | 1485.331 | 77.777 | 3577.915 |

## Test lock

All 89 test cases received integrity checks, but per-organ voxel counts, overlap counts, and volumes are redacted from the CSV and excluded from statistics.

## Spatial-unit note

The NIfTI spatial-units flag is unset in 1,228 cases. For volume calculations, affine units are explicitly assumed to be millimetres, consistent with the dataset-wide 1.5 mm spacing audit.
