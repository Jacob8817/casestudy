# CS16 Medical Digital Twin

## Patient skull verification

The CASE_001 patient reconstruction and the historical s0346 confidence display belong to different patients. The corrected workflow preserves the native anatomical surface and attaches patient-matched model confidence without deleting low-confidence anatomy.

- [Verification report](outputs/patient_skull_verification/verification_report.md)
- [Machine-readable checks](outputs/patient_skull_verification/verification_report.json)
- [Interactive CASE_001 skull with partial mandible and upper teeth](outputs/patient_skull_verification/skull_plus_mandible_confidence.html)
- [CT evidence through the cranial opening](outputs/patient_skull_verification/figures/opening_raw_vs_overlay.png)

Run `.venv/Scripts/python.exe scripts/verify_patient_skull.py` from this directory. This uses existing labels and a frozen checkpoint/temperature; it does not retrain, recalibrate or evaluate the official test set. Confidence is the MONAI skull probability sampled on TotalSegmentator geometry, not TotalSegmentator confidence. The CT-supported opening remains; incomplete anatomy outside the CT field of view is not reconstructed. Technical geometry verification does not certify anatomical accuracy.

Research objective:

> Lightweight, uncertainty-aware multi-organ 3D CT segmentation and reconstruction for patient-specific digital-twin generation.

Completed stages:

1. full dataset inspection and CT intensity audit;
2. versioned 10-organ schema;
3. official-split, SHA-256-locked manifest;
4. full selected-mask payload/content audit with leakage-safe volume statistics;
5. deterministic Phase 3 preprocessing, exclusive-label construction, patch sampling, and visual QA notebook;
6. pinned CUDA PyTorch/MONAI environment validated on the RTX 4060 with a real Phase 3 patch;
7. Phase 4 baseline preflight with exact-split datasets, paired augmentation QA, model/metric contracts,
   and an explicit no-training acceptance record.

No baseline training, model-quality evaluation, test-set evaluation, or original-dataset modification has
been performed. Phase 4 applies one deterministic augmentation for visual QA and one frozen inference-only
forward pass; it creates no optimizer, gradient, epoch, checkpoint, or research result. The earlier GPU gate
used a disposable optimizer smoke test only and did not retain a model.

## Environment

Python 3.10 or newer is required. The current audit tools use NumPy and PyYAML:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### GPU dependency set

The core GPU stack is exact-pinned separately so its versions remain auditable:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements-gpu.txt
.\.venv\Scripts\python.exe .\scripts\verify_gpu_stack.py
```

Validated stack: PyTorch `2.13.0+cu130`, CUDA runtime `13.0`, cuDNN `9.20.0`, MONAI `1.6.0`,
and NiBabel `5.4.0` on an NVIDIA GeForce RTX 4060 Laptop GPU (compute capability 8.9,
8,188 MiB VRAM; driver 592.82). The PyTorch wheel includes the required CUDA runtime and
cuDNN, so a standalone CUDA Toolkit/`nvcc` installation is not required unless custom CUDA
extensions are compiled later.

The fail-closed verifier performs CUDA allocation/reduction, CUDA matrix-multiplication parity, Conv3d
forward/backward, float16 AMP with gradient scaling, a paired MONAI CUDA transform, and a
MONAI 3D U-Net optimizer step on the exact locked Phase 3 training patch. It refuses CPU
fallback and confirms that the test split remains inaccessible. The passed machine-readable
record is `outputs/environment/gpu_verification.json`. Each successful report includes UTC timestamps,
the invocation, environment/config hashes, and data-contract fingerprints. A previous pass is archived
before a new attempt, so a failed rerun cannot leave the canonical PASS report looking current.

Reproducibility note: this PyTorch build does not mark 5D CUDA CrossEntropy as strictly
deterministic. The verifier enables deterministic algorithms in warning mode and records the
limitation; future experiments must report it and must not claim bitwise-identical CUDA runs.

## Run the complete dataset audit

Pass either the extracted TotalSegmentator directory or a parent directory. Patient folders are discovered programmatically.

```powershell
.\.venv\Scripts\python.exe .\scripts\inspect_dataset.py `
    --dataset_root .\Totalsegmentator_dataset_v201 `
    --output-dir .\outputs `
    --workers 8
```

This audit discovers all cases, validates every CT/mask header and grid, streams every CT payload through gzip EOF, applies NIfTI scaling, calculates HU statistics, and verifies CT CRCs. Original dataset files are opened read-only.

Primary outputs:

- `outputs/dataset_audit.csv` - per-case geometry, HU statistics, mask inventory, and metadata.
- `outputs/dataset_summary.json` - dataset-level counts, distributions, integrity scope, and findings.
- `outputs/dataset_audit_report.md` - human-readable summary.
- `outputs/mask_inventory.csv` - coverage of all 117 source labels.
- `outputs/issues.csv` - actionable structural audit findings.

## Create the selected-organ manifest

The initial task has 10 foreground classes plus background. Left and right lungs are unions of their respective lobe masks. The manifest preserves the official `meta.csv` split and stores only dataset-root-relative paths.

```powershell
.\.venv\Scripts\python.exe .\scripts\create_manifest.py `
    --dataset_root .\Totalsegmentator_dataset_v201 `
    --organ-config .\configs\organs_v1.yaml `
    --output .\data\splits\dataset_manifest.json `
    --workers 8
```

The manifest contains SHA-256 hashes for every CT and all 13 selected source masks per case, split/test-lock fingerprints, CT geometry, and the complete output-channel mapping. It refuses to overwrite an existing manifest unless `--overwrite` is passed deliberately.

## Audit the selected masks

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_selected_labels.py `
    --dataset_root .\Totalsegmentator_dataset_v201 `
    --manifest .\data\splits\dataset_manifest.json `
    --organ-config .\configs\organs_v1.yaml `
    --workers 4
```

This streams all 15,964 selected source-mask payloads and validates gzip CRC, raw binary values, identity scaling, CT geometry, nonempty status, merged lungs, and cross-class overlaps. Peak memory is bounded by `--chunk-voxels`.

Outputs:

- `data/splits/dataset_manifest.json` - immutable official-split data contract.
- `outputs/selected_organs_audit.csv` - one row per case and target organ.
- `outputs/organ_volume_statistics.json` - train/validation-only raw and exclusive volume statistics.
- `outputs/selected_organs_report.md` - concise findings and presence table.

All test masks receive integrity checks, but test voxel counts, volumes, and overlap counts are redacted and excluded from statistics.

### Overlap policy

Individual TotalSegmentator masks overlap in some cases. The project does not silently assign priority. `configs/organs_v1.yaml` explicitly reproduces the official TotalSegmentator v2 binary-to-multilabel behavior: labels are processed in official class-map order and later labels overwrite earlier labels. Both raw mask volume and exclusive post-resolution training volume remain auditable.

Empty masks are retained because many scans cover only part of the body. They are coverage findings, not corruption. Future training samplers must handle background-only cases deliberately rather than altering the official split.

## Optional exhaustive 117-mask CRC audit

The selected 13 source masks are fully CRC-checked. Auditing every payload for all 117 labels remains available, but the decoded masks total roughly 2.06 TiB:

```powershell
.\.venv\Scripts\python.exe .\scripts\inspect_dataset.py `
    --dataset_root .\Totalsegmentator_dataset_v201 `
    --output-dir .\outputs_full_crc `
    --integrity-mode full
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover little- and big-endian NIfTI files, intensity scaling, orientation/canonicalization, CRC corruption,
invalid offsets, organ-schema validation, merged-lung union, nonbinary masks, overlap resolution, locked-test
rejection/redaction, affine-aware resampling, fixed-HU normalization, CT-derived cropping, deterministic sampling,
background-only padding, strict baseline configuration, paired-transform integrity, model/loss construction,
pre-I/O test rejection, and double training authorization.

## Show or rerun the Phase 3 notebook

The executed presentation notebook is `code.ipynb`. It contains saved tables and figures, so it can be opened directly in VS Code or Jupyter without rerunning it first.

To rebuild the notebook source and execute every cell from a clean state:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_phase3_notebook.py
.\.venv\Scripts\python.exe .\scripts\execute_notebook.py .\code.ipynb --timeout 600
```

The notebook:

- validates the manifest, schema, and preprocessing fingerprints;
- blocks the locked test split before resolving any test path;
- streams 13 binary masks into one exclusive 11-class label volume using the declared official overlap priority;
- verifies exact agreement with the prior full selected-mask audit;
- verifies RAS orientation and 1.5 mm spacing as exact no-ops for the audited release;
- clips `[-1000, 1000]` HU and maps it to fixed `[0, 1]` float32 values;
- produces a deterministic class-balanced `96 x 96 x 96` foreground patch;
- proves that a background-only partial-field scan is retained and padded safely;
- previews CT-only foreground cropping while keeping it globally disabled pending dataset-wide retention QA.

Phase 3 outputs:

- `outputs/phase3/qa_summary.json` - machine-readable acceptance matrix and spatial trace;
- `outputs/phase3/s0014_multiplanar_overlay.png` - three-plane CT/organ overlay;
- `outputs/phase3/s0014_class_balanced_patch.png` - pancreas-centered deterministic patch;
- `outputs/phase3/s0099_background_fallback_patch.png` - padded background-only fallback.

All 17 notebook acceptance checks pass. The preprocessing fingerprint is `92ec417172ae9a564135fed9bb0ae173de5305651f96220c57b29655e92b8c71`.

## Show or rerun the Phase 4 baseline preflight

The executed presentation notebook is `04_baseline_training.ipynb`. It implements and validates the baseline
contract while keeping training hard-disabled.

```powershell
.\.venv\Scripts\python.exe .\scripts\build_phase4_notebook.py
.\.venv\Scripts\python.exe .\scripts\execute_notebook.py .\04_baseline_training.ipynb --timeout 600
```

The preflight:

- validates the locked manifest, schema, preprocessing, GPU environment, and official split counts;
- constructs exact train-patch and deterministic full-volume validation interfaces;
- rejects the test split before resolving its deliberately invalid path;
- exactly reproduces the Phase 3 `s0014` sample-14 patch;
- applies paired affine augmentation without laterality-unsafe flips;
- constructs the 4,817,434-parameter MONAI 3D U-Net and DiceCE objective;
- performs one frozen `eval()`/`inference_mode()` shape check without evaluating loss or quality metrics;
- proves parameter/buffer hashes are unchanged and all Phase 4 training counters remain zero;
- creates no cache, optimizer, checkpoint, export, TensorBoard event, or training-run directory.

Phase 4 setup artifacts:

- `outputs/phase4/preflight_summary.json` - machine-readable no-training acceptance record;
- `outputs/phase4/augmentation_qa.png` - original versus paired-augmentation multiplanar QA.

All 22 preflight acceptance checks and all 23 project unit tests pass. The experiment fingerprint is
`a0de203536b032ca4f033f04b887a546f08c1b14eeb7bbbf1449b52b863af8fb`.

## Run the staged Phase 5 training notebook

`05_staged_training.ipynb` is the authorized training notebook. The staged run is complete: all three gates passed,
the full baseline completed 50 epochs, and the best checkpoint is
`outputs/training/gate3_full_baseline/baseline_v1/checkpoints/best.pt`.

The notebook executes a fail-closed sequence:

1. revalidates the immutable dataset, preprocessing, GPU, disk-space, and locked-test contracts;
2. verifies source SHA-256 values and writes content-hashed preprocessing cache entries;
3. requires a 50-epoch train-only, fixed-patch one-case overfit diagnostic gate to pass;
4. deliberately pauses and resumes a deterministic 16-train/4-validation pilot;
5. trains the complete 1,082/57 development baseline for at most **50 epochs**, with validation every five
   epochs, early stopping, atomic `last.pt`/`best.pt`, and exact checkpoint resume.

The complete train/validation cache is expected to use about 82.3 GiB. The notebook uses zero DataLoader workers
for Windows/Jupyter reliability. Re-running **Run All** verifies and skips passed stages and resumes incomplete
stages from the latest complete-epoch checkpoint. The 89-case test split remains inaccessible until final frozen
evaluation.

All three training stages explicitly lock the AMP initial scale to `1024`. This value was verified on the real
32-patch Gate-1 bank and avoids the first-epoch FP16 gradient overflow observed with PyTorch's default scale of
`65536`. During the full run, isolated infinite FP16 gradients use GradScaler's standard skipped-update/backoff
path; NaN gradients and more than eight consecutive overflows still fail closed. The full-run cosine schedule
retains its original 100-epoch horizon and is deliberately truncated at the new 50-epoch cap so the validated
epoch-16 checkpoint can continue without changing its learning-rate trajectory.

To regenerate the unexecuted notebook source without starting training:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_staged_training_notebook.py
```

## Phase 6 uncertainty, locked test, and 3D digital twin

The fully executed presentation notebook is `06_uncertainty_locked_test_and_3d_digital_twin.ipynb`. It contains
saved calibration tables, uncertainty–error analysis, qualitative overlays, the frozen pre-test protocol, the
one-shot 89-case test report, and an embedded interactive 3D viewer. It never trains or changes model weights.

The validation-only decisions were frozen under protocol SHA-256
`b0a5f3239922d99741569a25d3e202b7f7ac1287cb823c4d9e37f9696ddb0759` before test access. Temperature scaling
selected `T=1.035455`, predictive-entropy error detection reached validation AUROC `0.9844`, and validation Dice
selected no connected-component postprocessing. The completed one-shot test reached macro foreground Dice
`0.8726`, uncertainty–error AUROC `0.9792`, and ECE `0.000693`. The patient-level macro-Dice bootstrap uses 79
defined cases and excludes 10 partial-field cases containing none of the selected ground-truth organs.

Final 3D outputs for the geometry-predeclared case `s0244`:

- `outputs/phase6/digital_twin/s0244_digital_twin.glb` — combined portable 10-organ digital twin;
- `outputs/phase6/digital_twin/s0244_digital_twin.html` — self-contained interactive viewer;
- `outputs/phase6/digital_twin/s0244_digital_twin_preview.png` — static preview;
- `outputs/phase6/digital_twin/individual_organs/` — full-resolution PLY and STL per organ;
- `outputs/phase6/locked_test/test_report.json` — final metrics;
- `outputs/phase6/frozen_protocol.json` — immutable pre-test selections and hashes.

Re-running the notebook verifies and reuses completed validation, test, and mesh artifacts. It will not repeat the
one-shot test or overwrite the frozen protocol with different choices.

## Brain-only 3D reconstruction notebook

`06_brain_only_3d_reconstruction.ipynb` is a standalone, executed alternative to the multi-organ Phase 6
notebook. It loads one CT plus its matching binary brain mask, validates their geometry, shows three-plane overlay
QA, and exports a patient-space brain surface as PLY, STL, GLB, interactive HTML, PNG, and JSON. The default
demonstration uses validation case `s0346` and writes to `outputs/brain_3d/s0346/`.

The current ten-organ checkpoint does not predict brain. For a new CT, a matching mask from a separately trained
and validated brain segmentation model must be supplied through `CT_PATH` and `BRAIN_MASK_PATH` in the input cell.

```powershell
.\.venv\Scripts\python.exe .\scripts\build_brain_3d_notebook.py
.\.venv\Scripts\python.exe .\scripts\execute_notebook.py .\06_brain_only_3d_reconstruction.ipynb --timeout 600
```

## Notebook 15: brain and skull benchmark

`15_brain_skull_benchmark.ipynb` is a separate, development-only CT segmentation experiment. It selects
complete-head cases from the official train/validation splits (98 train, 5 validation in the current inventory),
uses the existing `brain.nii.gz` and `skull.nii.gz` masks with an explicit skull-wins overlap policy, and compares
a >300 HU skull threshold with newly trained 3D U-Net and SegResNet models. The official test split stays locked.

The notebook saves resumable `last.pt`, best-validation `best.pt`, per-model reports, a comparison CSV, and
learning curves under `outputs/brain_skull_benchmark/`. Its default 30 epochs are a development benchmark; increasing
`MAX_EPOCHS` resumes the same runs. Inputs are two CT windows (brain and bone) on a mask-derived development crop.
This crop is not suitable for independent CT-only deployment or external validation without replacing it with a
CT-derived head localizer. The original ten-organ checkpoint is not reused.

```powershell
.\.venv\Scripts\python.exe .\scripts\build_brain_skull_benchmark_notebook.py
.\.venv\Scripts\python.exe .\scripts\execute_notebook.py .\15_brain_skull_benchmark.ipynb --timeout 7200
```

## Notebook 16: original MONAI U-Net head-task ablation

`16_original_monai_unet_brain_skull_ablation.ipynb` retrains the original project's five-level MONAI 3D U-Net
architecture from scratch for three separate targets: brain-only, skull-only, and joint brain+skull. It uses the
same frozen 98-train/5-validation complete-head cohort for all three, the original single CT window
`[-1000,1000] HU → [0,1]`, 96³ patches, and 50 epochs. The old ten-organ checkpoint cannot be reused because its
output channels contain neither brain nor skull. All three use matched sampled patch centers. Each task writes
resumable `last.pt` and validation-selected `best.pt` under
`outputs/original_monai_head_ablation_matched/`, plus a comparison table and visual QA. Earlier task-specific
sampling pilot artifacts remain in `outputs/original_monai_head_ablation/` but are not used for the matched comparison.

This is a task-formulation ablation, not an exact rerun of the old 10-organ training recipe. Like Notebook 15, its
development crops use reference masks and its official test split is untouched. A fair comparison to a fresh
nnU-Net requires training nnU-Net on the same cohort, target rules, and evaluation cases. Comparing directly to
released TotalSegmentator weights on this source dataset would not establish independent external performance.

```powershell
.\.venv\Scripts\python.exe .\scripts\build_original_monai_ablation_notebook.py
.\.venv\Scripts\python.exe .\scripts\execute_notebook.py .\16_original_monai_unet_brain_skull_ablation.ipynb --timeout 10800
```
