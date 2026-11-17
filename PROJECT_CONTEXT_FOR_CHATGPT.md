# Complete project context for ChatGPT

> Copy this document into a new ChatGPT/Codex conversation when continuing the project. It records what was
> implemented and executed from dataset extraction through the final 3D digital twin. The dataset is
> **TotalSegmentator CT v2.0.1, not MRI**.

## 1. Project identity and research objective

- Degree context: M.Tech healthcare computer-vision project.
- Problem statement: **CS16 — AI-Driven 3D Reconstruction and Segmentation of MRI/CT Images for Precision Medical Planning**.
- This implementation selected the CT branch of the problem statement.
- End-to-end research goal:

  `3D CT volume → multi-organ segmentation → uncertainty estimation → 3D anatomical reconstruction → patient-specific digital twin`

- Project root on the original machine: `F:\csstdy`.
- Current state: the dataset audit, preprocessing pipeline, GPU environment, baseline model, staged training,
  validation calibration, uncertainty analysis, one-shot locked-test evaluation, and 3D reconstruction are complete.
- This is a research pipeline with immutable fingerprints, patient-level split controls, test-access auditing,
  resumable checkpoints, and machine-readable reports. It is not merely a demonstration notebook.

## 2. Dataset actually used

- Dataset: **TotalSegmentator CT v2.0.1**.
- It is **not** the TotalSegmentator MRI dataset.
- Source archive: `Totalsegmentator_dataset_v201.zip`.
- Extracted directory: `Totalsegmentator_dataset_v201/`.
- Original download source:
  `https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip?download=1`
- Modality evidence:
  - every subject uses `ct.nii.gz` or `ct.nii`;
  - preprocessing operates in Hounsfield units (HU);
  - the dataset metadata identifies CT study types;
  - all configuration and reports use `totalsegmentator_ct_v2.0.1`.

Typical native case layout:

```text
Totalsegmentator_dataset_v201/
├── meta.csv
├── s0001/
│   ├── ct.nii.gz
│   └── segmentations/
│       ├── liver.nii.gz
│       ├── spleen.nii.gz
│       └── ... 117 binary masks total
└── ...
```

The code discovers and validates the actual hierarchy rather than relying blindly on the example layout.

## 3. Dataset audit completed

The first implementation stage inspected the extracted dataset programmatically and created a durable audit.

Verified inventory:

- subjects: **1,228**;
- CT files: **1,228**;
- structurally valid CT files: **1,228**;
- mask files: **143,676** (`1,228 × 117`);
- distinct mask labels: **117**;
- every subject has exactly 117 masks;
- metadata rows: **1,228**;
- official split counts: train `1,082`, validation `57`, test `89`;
- missing/corrupt structural records: none in the audit report;
- audit severity counts and issue-category counts: empty.

Integrity scope:

- every NIfTI header was parsed;
- gzip magic/trailer and advertised uncompressed size were checked for the dataset-wide header audit;
- the selected 13 binary source masks used by the 10-organ task were later decoded/CRC-checked during selected-label auditing and preprocessing;
- original NIfTI files were opened read-only and were not modified.

Primary audit artifacts:

```text
audit_outputs/summary.json
audit_outputs/dataset_audit_report.md
audit_outputs/subjects.csv
audit_outputs/mask_inventory.csv
audit_outputs/issues.csv
outputs/selected_organs_audit.csv
outputs/selected_organs_report.md
```

Important nuance: many CTs cover only part of the body, so some selected organs are legitimately absent. Empty
masks were retained; they were not treated as corrupt data or deleted.

## 4. Selected segmentation task

Training all 117 structures was deliberately not attempted on the 8 GB laptop GPU. A locked, clinically relevant
10-organ foreground task plus background was defined in `configs/organs_v1.yaml`.

| Class ID | Target | TotalSegmentator source mask(s) |
|---:|---|---|
| 0 | background | none |
| 1 | liver | `liver` |
| 2 | spleen | `spleen` |
| 3 | kidney_left | `kidney_left` |
| 4 | kidney_right | `kidney_right` |
| 5 | stomach | `stomach` |
| 6 | pancreas | `pancreas` |
| 7 | aorta | `aorta` |
| 8 | inferior_vena_cava | `inferior_vena_cava` |
| 9 | lung_left | union of left upper and lower lobes |
| 10 | lung_right | union of right upper, middle, and lower lobes |

The 13 selected binary source masks are streamed and merged into one exclusive `uint8` label map. Overlaps follow
the official TotalSegmentator v2 last-label-wins order, explicitly recorded rather than handled silently. The
low-to-high overwrite priority is:

```text
spleen → kidney_right → kidney_left → liver → stomach → pancreas
→ lung_left → lung_right → aorta → inferior_vena_cava
```

## 5. Immutable dataset and preprocessing locks

The official `meta.csv` split was retained at patient level. No custom random resplit was introduced.

```text
train = 1,082
validation = 57
test = 89
```

The test split was rejected before path resolution in all audit, preprocessing, preflight, and training code. It was
opened only after the final validation choices were frozen in Phase 6.

Critical fingerprints:

```text
dataset manifest SHA-256:
4c94df936370ee913b0202fc5a34cba9843727d8dabb0552a3b8160dea223d8d

organ schema SHA-256:
082fa87535d1a4c91d0eaf4aa074ebf297cd0ec386183643038bf2c583989c05

preprocessing SHA-256:
92ec417172ae9a564135fed9bb0ae173de5305651f96220c57b29655e92b8c71
```

Manifest: `data/splits/dataset_manifest.json`.

## 6. Preprocessing implementation

Configuration: `configs/dataset.yaml`.

Implemented deterministic preprocessing:

1. Decode calibrated CT values as finite `float32` HU.
2. Decode each selected binary mask and validate CT/mask geometry.
3. Merge the source masks into an exclusive 11-class `uint8` label map.
4. Canonicalize to RAS orientation by exact axis permutation/flipping when needed.
5. Resample to `1.5 × 1.5 × 1.5 mm` using linear CT interpolation and nearest-neighbour label interpolation.
6. Clip CT intensities to `[-1000, 1000] HU`.
7. Map the fixed HU interval linearly to `[0, 1]` without per-volume normalization.
8. Use deterministic class-balanced `96 × 96 × 96` training patches.
9. Sample a foreground class with probability `0.9`; otherwise sample valid body/background.
10. Retain and safely pad partial-field/background-only scans.

Global CT-derived foreground cropping is implemented but remains disabled because it was not audited for retention
across the entire dataset. A Phase 3 preview retained 100% of selected labels on the demonstration subject and
reduced voxels by about 29%, but this was not enough evidence to enable it globally.

The fingerprinted lazy disk cache is under:

```text
data/cache/baseline_v1/92ec417172ae9a56/
```

Each cached case contains immutable `ct_hu.npy`, `label.npy`, `affine.npy`, hashes, source fingerprint, and atomic
generation metadata. The complete train/validation cache was estimated at roughly 82.3 GiB.

Phase 3 notebook and artifacts:

```text
code.ipynb
outputs/phase3/qa_summary.json
outputs/phase3/s0014_multiplanar_overlay.png
outputs/phase3/s0014_class_balanced_patch.png
outputs/phase3/s0099_background_fallback_patch.png
```

All 17 Phase 3 acceptance checks passed, and no test case was accessed.

## 7. GPU/CUDA environment

A dedicated virtual environment at `.venv/` was configured and cross-checked using real 3D tensors and a real
preprocessed CT patch.

Verified stack:

```text
OS: Windows 11
Python: 3.13.12
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
VRAM: approximately 8,188 MiB
NVIDIA driver: 592.82
PyTorch: 2.13.0+cu130
PyTorch CUDA runtime: 13.0
cuDNN: 9.20.0
MONAI: 1.6.0
NiBabel: 5.4.0
NumPy: 2.5.2
```

GPU verification report: `outputs/environment/gpu_verification.json`.

Passed checks included CUDA allocation/reduction, matrix multiplication parity, cuDNN Conv3D, AMP + GradScaler,
the real Phase 3 patch on GPU, paired MONAI transforms, and an AMP training smoke step. `nvcc` was not installed and
was not required because prebuilt PyTorch wheels were used.

The known determinism limitation is documented: this PyTorch build does not mark CUDA 5D cross-entropy as strictly
deterministic. Deterministic controls were enabled in warn-only mode.

## 8. Baseline model and training configuration

The fixed baseline is a conventional MONAI 3D U-Net, chosen as a defensible reference rather than claimed as a new
architecture.

Model:

```text
implementation: monai.networks.nets.UNet
spatial dimensions: 3
input channels: 1
output channels: 11
channels: [16, 32, 64, 128, 256]
strides: [2, 2, 2, 2]
residual units per stage: 2
normalization: instance normalization
activation: PReLU
dropout: 0.0
parameters: 4,817,434
```

Objective and optimization:

```text
loss: MONAI DiceCELoss
Dice background: excluded
target conversion: one-hot
activation in loss: softmax
Dice weight: 1.0
cross-entropy weight: 1.0
optimizer: AdamW
learning rate: 1e-4
weight decay: 1e-5
scheduler: CosineAnnealingLR
scheduler T_max: 100 epochs
gradient accumulation: 2
gradient clipping norm: 12.0
AMP: enabled
AMP initial scale: 1024
full training cap: 50 epochs
validation interval: every 5 epochs
```

Training augmentation:

- paired affine transform probability `0.50`;
- rotation range `0.10` radians per axis;
- translation range `5` voxels per axis;
- scale range `0.10`;
- bilinear image interpolation and nearest label interpolation;
- Gaussian noise probability `0.15`, standard deviation `0.01`;
- contrast adjustment probability `0.15`, gamma range `[0.90, 1.10]`;
- no naive spatial flips because left/right organs are distinct classes;
- final values clamped to `[0, 1]`.

Full-volume validation used deterministic MONAI sliding-window inference:

```text
ROI: 96 × 96 × 96
overlap: 0.50
sliding-window batch size: 1
blend mode: Gaussian
metrics: foreground Dice and HD95
empty-target policy: NaN excluded from class means
```

Baseline preflight notebook: `04_baseline_training.ipynb`.

It constructed the model and validated the complete contract but performed no training. All 22 preflight acceptance
checks passed, the test remained locked, and the random-weight model state was unchanged by its inference-only shape
check.

## 9. Staged training and failure recovery

Training notebook: `05_staged_training.ipynb`.

Training was organized into three gates so that data, gradients, checkpointing, resume behavior, and validation
semantics were proven before the expensive full run.

### Gate 1 — one-case overfit diagnostic

- Case: train subject `s0014` only.
- Fixed deterministic patch bank: 32 patches representing all 10 foreground classes.
- Final configured length: 50 epochs.
- Optimizer steps: 1,600.
- Initial bank loss: `3.82825`.
- Final bank loss: `0.59306`.
- Loss reduction: `84.51%`.
- Initial represented-class mean Dice: `0.03683`.
- Final represented-class mean Dice: `0.94633`.
- Checkpoint reload parity: passed.
- Validation and test splits were not accessed.

### Gate 2 — deterministic pilot

- Train subjects: 16 predeclared cases.
- Validation subjects: 4 predeclared cases.
- Epochs: 3.
- A deliberate pause and exact resume were required and passed.
- Optimizer steps: 48.
- Validation runs: 3.
- Resume events: 1.
- This gate verified the end-to-end training/checkpoint/validation machinery; its small validation score was not used
  as the final model-quality result.

### Gate 3 — full baseline

- Train cases: all 1,082 official training subjects.
- Validation cases: all 57 official validation subjects.
- Completed epochs: 50.
- Training batches consumed: 54,100.
- Backward calls: 54,100.
- Optimizer steps: 27,044.
- AMP overflow skips: 6.
- Scheduler steps: 50.
- Full-volume validation runs: 10.
- Resume events: 1.
- Best validation mean foreground Dice: **0.8915950242**.
- Test split accessed during training: false.

Best checkpoint:

```text
outputs/training/gate3_full_baseline/baseline_v1/checkpoints/best.pt
```

Best-checkpoint SHA-256:

```text
81632bd721ffd43b10ad341ab0b2e906dab96d38a322e7f97cb8752a17fb3ba1
```

Final validation Dice from the trained baseline:

| Organ | Dice |
|---|---:|
| liver | 0.954721 |
| spleen | 0.937831 |
| kidney_left | 0.875632 |
| kidney_right | 0.903878 |
| stomach | 0.865275 |
| pancreas | 0.740566 |
| aorta | 0.890687 |
| inferior_vena_cava | 0.802468 |
| lung_left | 0.969513 |
| lung_right | 0.975380 |

### Training failures encountered and how they were handled

These events are preserved in `outputs/training/recovery_archive/` rather than deleted.

1. **Initial Gate-1 non-finite gradient:** training failed around batch 14 with the default AMP scale. The root cause
   was FP16 overflow rather than corrupt data. The staged configs were locked to AMP initial scale `1024`.
2. **Initial one-case overfit acceptance failure:** the shorter 15-epoch diagnostic did not meet the required loss
   reduction and represented-class Dice. The diagnostic was strengthened to a deterministic 32-patch bank and 50
   epochs; it then passed strongly.
3. **Full-run non-finite gradient around epoch 17:** the engine initially treated every infinite AMP gradient as a
   fatal error. It was corrected to match standard GradScaler semantics: an infinite-gradient optimizer update is
   skipped, the scale backs off, and training continues; NaN gradients still fail immediately, and more than eight
   consecutive overflows fail closed.
4. **Trajectory-preserving resume:** the valid epoch-16 state was migrated under a recorded provenance-compatibility
   report, the cosine horizon remained 100 epochs, the total cap became 50 epochs, and training resumed rather than
   restarting. Six isolated overflows were safely skipped in the complete run.

The final maximum was **50 epochs**, not 100 or 300. No distributed/multi-GPU training was introduced. The production
loader supports limited workers, but the notebook used zero DataLoader workers for Windows/Jupyter reliability and
preserved exact resume behavior.

## 10. Phase 6 uncertainty and validation-only calibration

Notebook: `06_uncertainty_locked_test_and_3d_digital_twin.ipynb`.

This notebook is fully executed and saved. It does not train or change model weights.

Because the trained U-Net has dropout `0.0`, MC-dropout uncertainty would be invalid. The implemented uncertainty is
therefore explicitly called **post-hoc predictive uncertainty**, not epistemic uncertainty.

Method:

1. Run the fixed best checkpoint on all 57 validation cases.
2. Uniformly sample 20,000 logits/labels per validation case: 1,140,000 voxels total.
3. Fit one positive scalar temperature by minimizing validation multiclass negative log likelihood.
4. Run validation inference again to compute calibrated normalized predictive entropy.
5. Compare entropy with voxelwise segmentation error.
6. Select an error-flag threshold by maximum validation voxel-error F1.
7. Compare three predeclared class-wise postprocessing policies on validation Dice.
8. Generate qualitative overlays.
9. Freeze all values and hashes before opening test.

Calibration results:

```text
temperature T: 1.0354552767
NLL before: 0.02259128
NLL after: 0.02256477
15-bin ECE before: 0.00062512
15-bin ECE after: 0.00029349
multiclass Brier after: 0.01198109
voxel accuracy: 0.9919333
```

Validation uncertainty–error results:

```text
sampled voxels: 1,140,000
error prevalence: 0.008007
error-detection AUROC: 0.984430
error-detection AUPRC: 0.351050
area under risk–coverage curve: 0.00017026
mean entropy on correct voxels: 0.007712
mean entropy on erroneous voxels: 0.227056
```

Frozen uncertainty threshold:

```text
selection rule: maximum validation voxel-error F1
normalized entropy threshold: 0.2315673828125
validation F1: 0.445940
validation precision: 0.361972
validation recall: 0.580631
```

Validation postprocessing comparison:

| Policy | Validation macro Dice |
|---|---:|
| none | **0.891598** |
| largest component per class | 0.891068 |
| remove components below 1 ml | 0.890245 |

The frozen policy is therefore `none`. This maximized the declared validation selection metric. Consequently, final
mesh files faithfully retain all predicted components, including possible small false-positive components.

Qualitative validation overlays were generated for `s0249`, `s1017`, and `s1161`, each showing CT, ground truth,
prediction, error, and calibrated entropy:

```text
outputs/phase6/validation/qualitative_overlays/
```

Validation report: `outputs/phase6/validation/validation_report.json`.

## 11. Frozen protocol and one-shot locked test

Before any test CT or target was opened, Phase 6 wrote:

```text
outputs/phase6/frozen_protocol.json
```

Frozen protocol SHA-256:

```text
b0a5f3239922d99741569a25d3e202b7f7ac1287cb823c4d9e37f9696ddb0759
```

The protocol binds:

- manifest, organ-schema, preprocessing, training-config, and checkpoint hashes;
- fitted temperature;
- entropy error threshold;
- postprocessing policy;
- exact official test-case ID hash and count;
- one-shot/resume rules;
- predeclared 3D mesh-case selection rule.

Only after this freeze did the workflow write a test authorization receipt and allow the special test loader to read
test data. Existing development loaders remain fail-closed for test records.

The one-shot test ran all **89/89** official test cases. It wrote an access ledger and per-case atomic results, allowing
only exact case-level recovery under the same protocol hash. No value was tuned after test access.

Final test summary:

```text
status: complete_one_shot
test cases: 89
macro foreground Dice (mean of per-class means): 0.8725907970
calibration sample voxels: 1,780,000
NLL at frozen temperature: 0.01746580
15-bin ECE: 0.00069325
multiclass Brier: 0.00860401
voxel accuracy: 0.9943202
uncertainty/error AUROC: 0.9792497
uncertainty/error AUPRC: 0.3446428
frozen-threshold precision: 0.3726579
frozen-threshold recall: 0.5312098
```

Per-class test results:

| Organ | Mean Dice | Mean HD95 (mm) | Valid Dice cases |
|---|---:|---:|---:|
| liver | 0.927352 | 13.659 | 70 |
| spleen | 0.945284 | 11.512 | 65 |
| kidney_left | 0.845150 | 38.662 | 68 |
| kidney_right | 0.869348 | 24.287 | 64 |
| stomach | 0.846183 | 28.207 | 66 |
| pancreas | 0.718165 | 8.765 | 65 |
| aorta | 0.872917 | 75.363 | 77 |
| inferior_vena_cava | 0.795250 | 26.487 | 72 |
| lung_left | 0.959690 | 71.473 | 73 |
| lung_right | 0.946568 | 87.019 | 73 |

The variation in valid cases comes from partial-field CT coverage. Ten test subjects contain none of the ten selected
ground-truth organs, so patient-level macro Dice is undefined for those subjects. Following the already locked
empty-target policy, the patient bootstrap excludes those ten cases:

```text
valid patients: 79
excluded empty-target patients: 10
patient-level mean macro Dice: 0.851159
95% nonparametric bootstrap CI: [0.807936, 0.888138]
bootstrap replicates: 2,000
```

The first generated bootstrap summary accidentally propagated the ten undefined values and produced blank CI fields.
This reporting-only bug was corrected from the already committed per-case JSON files without reopening test data or
rerunning inference. The before/after report hashes and scope are recorded in:

```text
outputs/phase6/locked_test/bootstrap_empty_target_correction.json
```

Primary test artifacts:

```text
outputs/phase6/locked_test/test_report.json
outputs/phase6/locked_test/test_access_receipt.json
outputs/phase6/locked_test/test_access_log.json
outputs/phase6/locked_test/cases/                 # 89 per-case JSON files
outputs/phase6/locked_test/uncertainty_samples/   # 89 sampled-logit/error files
```

The official test split has now been accessed and evaluated once. It must not be used for future tuning.

## 12. 3D reconstruction and digital twin

The mesh case was selected before test access using only manifest CT geometry:

```text
rule: closest to the test-set 75th-percentile physical volume among cases
      whose three native axes are at least 200 voxels; case ID breaks ties
selected case: s0244
native shape: 288 × 288 × 287
native spacing: 1.5 × 1.5 × 1.5 mm
test labels used in mesh-case selection: false
```

Reconstruction method:

1. Use the frozen best-checkpoint prediction and frozen postprocessing policy.
2. Run marching cubes separately on every predicted foreground class.
3. Transform voxel-space vertices through the resampled RAS affine.
4. Store world coordinates in patient-space millimetres.
5. Export full-resolution PLY and STL per organ.
6. Combine all ten organs into a portable GLB scene.
7. Build a self-contained Plotly HTML viewer.
8. Color each organ consistently; PLY vertex colors blend toward yellow as calibrated normalized entropy increases.
9. Cap faces only in the browser HTML representation for responsiveness; the disk mesh exports remain full resolution.

Mesh inventory:

| Organ | Vertices | Faces | Watertight |
|---|---:|---:|---|
| liver | 51,004 | 102,028 | yes |
| spleen | 14,000 | 28,004 | yes |
| kidney_left | 17,703 | 35,168 | no |
| kidney_right | 13,624 | 27,256 | yes |
| stomach | 33,817 | 68,006 | no |
| pancreas | 10,997 | 21,986 | yes |
| aorta | 13,981 | 27,856 | no |
| inferior_vena_cava | 10,729 | 21,356 | no |
| lung_left | 20,010 | 39,484 | no |
| lung_right | 36,402 | 72,262 | no |

Final 3D artifacts:

```text
outputs/phase6/digital_twin/s0244_digital_twin.glb
outputs/phase6/digital_twin/s0244_digital_twin.html
outputs/phase6/digital_twin/s0244_digital_twin_preview.png
outputs/phase6/digital_twin/individual_organs/*.ply
outputs/phase6/digital_twin/individual_organs/*.stl
outputs/phase6/digital_twin/mesh_report.json
```

The GLB contains ten geometries. The HTML can be opened in a browser; the GLB can be opened in Blender or another
GLB-compatible 3D viewer.

## 13. Main source-code structure

```text
src/dataset/
├── audit.py                 # dataset inspection support
├── label_mapping.py         # strict 10-organ schema
├── loader.py                # manifest-driven NIfTI loading and test lock
├── manifest.py              # immutable manifest creation/fingerprints
├── preprocessing.py         # RAS, spacing, HU normalization, patch sampling
└── selected_mask_audit.py

src/training/
├── config.py                # strict baseline config validation
├── data.py                  # cache, exact split datasets, augmentations
├── model.py                 # UNet/loss/optimizer/scheduler factories
├── engine.py                # full training, AMP handling, validation, resume
├── gates.py                 # one-case overfit gate
├── provenance.py
└── workflow_config.py

src/evaluation/
├── metrics.py               # calibration, entropy/error, Dice, HD95, postprocessing
├── pipeline.py              # ordered validation → freeze → test workflow
└── mesh.py                  # marching cubes and PLY/STL/GLB/HTML exports
```

Important configurations:

```text
configs/dataset.yaml
configs/organs_v1.yaml
configs/environment_gpu.yaml
configs/training_workflow_v1.yaml
configs/training_gate1_overfit_v2.yaml
configs/training_gate2_pilot_v2.yaml
configs/training_gate3_full_v2.yaml
configs/phase6_uncertainty_test_mesh_v1.yaml
```

Notebook builders and utilities are in `scripts/`. The sixth notebook is generated by
`scripts/build_phase6_notebook.py` and executed by `scripts/execute_notebook.py`.

## 14. Notebook sequence and current status

| Notebook | Purpose | Current state |
|---|---|---|
| `code.ipynb` | Phase 3 preprocessing and visual QA | executed; passed |
| `04_baseline_training.ipynb` | model/data/GPU preflight with training disabled | executed; passed |
| `05_staged_training.ipynb` | Gate 1 → Gate 2 → full 50-epoch baseline | executed; all stages passed |
| `06_uncertainty_locked_test_and_3d_digital_twin.ipynb` | calibration, test, meshes | executed; complete |

Re-running notebook 06 verifies and reuses the completed validation report, frozen protocol, test report, and mesh
report. It does not retrain or repeat the one-shot test.

## 15. Verification status

After adding the Phase 6 evaluation and mesh code, the entire project test suite completed with:

```text
56 passed, 2 third-party deprecation warnings
```

The warnings come from MONAI/PyTorch JIT deprecation and are not test failures.

The final notebook has all nine code cells executed and contains zero saved exception outputs. The final GLB was
reloaded successfully as a 10-geometry `trimesh.Scene`.

## 16. Scientific interpretation and limitations

These points must remain explicit in any report, paper, viva, or future ChatGPT answer:

1. The project used **CT only**, not MRI.
2. The trained target is a 10-organ subset of the dataset's 117 structures.
3. The current model is a conventional 3D U-Net baseline, not yet a novel lightweight architecture.
4. The reported uncertainty is temperature-calibrated softmax predictive entropy. It is not a full Bayesian or
   epistemic uncertainty estimate because model dropout is zero and no ensemble was trained.
5. Temperature scaling calibrates confidence but does not change the argmax segmentation.
6. Many TotalSegmentator scans are partial field-of-view. Empty-target cases are retained and excluded from
   class/patient means only where the metric is undefined.
7. The test split has already been used once for final evaluation and must not be used for hyperparameter tuning,
   postprocessing selection, model selection, or qualitative case selection.
8. No independent external clinical dataset has been evaluated.
9. The dataset lacks independently verified stable patient UIDs beyond `meta.csv image_id`; the official split was
   treated as patient level.
10. Some reconstructed organ meshes are not watertight, and the validation-selected `none` policy can retain small
    false-positive components. Meshes are research outputs, not clinical/surgical devices.
11. The work must not be described as clinically validated or ready for patient care.

## 17. Rules for any future ChatGPT/Codex session

1. Work from `F:\csstdy` and use `.venv\Scripts\python.exe`.
2. Treat all original NIfTI files under `Totalsegmentator_dataset_v201/` as read-only.
3. Do not change the immutable manifest, schema, preprocessing, checkpoint, or frozen-protocol hashes silently.
4. Do not train unless the user explicitly asks for a new experiment.
5. Never tune anything on the existing 89-case test results.
6. Do not claim the uncertainty is epistemic.
7. Do not claim all 117 classes were trained; only the ten listed organs were targets.
8. Preserve the correction audit and all recovery archives.
9. If a new model, postprocessing policy, or uncertainty method is developed, select it using train/validation only
   and preferably use a new untouched external dataset for final confirmation.
10. Use the JSON reports as ground truth rather than reading approximate values from notebook display text.

## 18. Sensible next research steps

The end-to-end baseline objective is complete. Strong journal-oriented extensions would be new experiments, not
retroactive changes to the completed test:

- compare against a lighter architecture and report parameters, FLOPs, latency, VRAM, Dice, and HD95;
- implement a genuinely uncertainty-aware model or deep ensemble on train/validation;
- perform per-organ calibration and boundary-focused uncertainty analysis;
- add cross-validation within the development set while keeping the completed test untouched;
- evaluate on an external CT dataset;
- perform statistical paired comparisons and confidence intervals across cases;
- validate a mesh-cleaning/connected-component strategy on development data for a future protocol;
- quantify mesh surface error, topology, watertightness, and volume error;
- build an interactive clinical-style viewer linking CT slices, masks, uncertainty, and 3D surfaces;
- prepare manuscript figures, ablations, methods, reproducibility checklist, and limitations section.

## 19. Quick commands

Activate the environment in PowerShell:

```powershell
cd F:\csstdy
.\.venv\Scripts\Activate.ps1
```

Run all project tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Rebuild the sixth notebook source without rerunning inference immediately:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_phase6_notebook.py
```

Execute/reuse the sixth notebook artifacts:

```powershell
.\.venv\Scripts\python.exe .\scripts\execute_notebook.py `
  .\06_uncertainty_locked_test_and_3d_digital_twin.ipynb --timeout 1200
```

Rebuild the standalone project flowchart:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_project_flowchart.py
```

## 20. Short handoff summary

This is a completed, leakage-controlled **TotalSegmentator CT v2.0.1** 10-organ segmentation baseline. The project
audited 1,228 CTs and 143,676 masks, used the official 1,082/57/89 split, trained a 4.82 M-parameter MONAI 3D U-Net
for 50 epochs after two diagnostic gates, achieved validation macro Dice `0.8916`, fitted validation-only temperature
scaling and predictive entropy, froze all decisions, completed one-shot 89-case test evaluation with macro Dice
`0.8726` and uncertainty/error AUROC `0.9792`, and exported a 10-organ patient-space GLB/HTML/PLY/STL digital twin
for predeclared case `s0244`. The test is no longer available for tuning.
