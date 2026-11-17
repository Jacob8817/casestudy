# Review 2 Module 3 verification

Slides 8 and 10 inspected in CS16_Review2_Ten_References.pptx. The other three CS16 Review2 deck variants have identical Module 3 slide-10 requirements. No deck edits.

| Component | Status | Finding |
|---|---|---|
| NLL temperature calibration | FIXED | Existing log-temperature validation-NLL objective is correct. Added frozen replay; no temperature refit. |
| Priority Q(x) | FIXED | Existing 0.45 U + 0.35 V + 0.20 B preserved. Added frozen threshold replay and affine/finite-input checks. |
| Local flip ensembling | FIXED | Existing inverse flips and mean logits verified. Added ROI locations, candidate/visited/changed counts, selection masks and executable invariant. |

## Numerical evidence

Calibration: `{'temperature': 1.1064573224712873, 'nll_before': 0.03017535583331213, 'nll_after': 0.02988350496219222, 'nll_at_0_99_T': 0.029886539586344102, 'nll_at_1_01_T': 0.029886515018222207, 'derivative_wrt_inverse_temperature': 4.6889167874155644e-08, 'curvature_wrt_inverse_temperature': 0.07399794862011291, 'validation_samples': 100000, 'refitted': False}`.
Priority threshold: `0.4128813147544861`; verified sample F1: `0.4627231361568078`. No threshold adjustment.

| Case | ROIs | Q candidates | Selected in ROIs | Changed | Outside selection | Brain Dice before / after | Skull Dice before / after |
|---|---:|---:|---:|---:|---:|---|---|
| s0346 | 3 | 89012 | 63244 | 25565 | 0 | 0.98308366 / 0.98620748 | 0.90659879 / 0.90691277 |
| s0792 | 3 | 56007 | 53771 | 13783 | 0 | 0.98718407 / 0.98735222 | 0.95451516 / 0.95041104 |
| s0874 | 3 | 89804 | 67913 | 23947 | 0 | 0.98767829 / 0.98589531 | 0.91751789 / 0.91192101 |
| s1161 | 3 | 398065 | 182563 | 74607 | 0 | 0.62103524 / 0.63233235 | 0.73398453 / 0.75622470 |
| s1260 | 3 | 61714 | 57797 | 14225 | 0 | 0.98894982 / 0.98690086 | 0.94700816 / 0.94829985 |

## ROI locations

Index ranges are half-open [start, stop), in the cached CT grid; JSON also records RAS-mm origins.

**s0346**
- [[47, 143], [88, 184], [116, 212]]
- [[47, 143], [110, 206], [56, 152]]
- [[119, 215], [23, 119], [21, 117]]

**s0792**
- [[12, 108], [13, 109], [0, 96]]
- [[16, 112], [12, 108], [13, 109]]
- [[12, 108], [24, 120], [13, 109]]

**s0874**
- [[20, 116], [56, 152], [116, 212]]
- [[21, 117], [88, 184], [85, 181]]
- [[46, 142], [18, 114], [0, 96]]

**s1161**
- [[82, 178], [128, 224], [142, 238]]
- [[115, 211], [117, 213], [403, 499]]
- [[62, 158], [62, 158], [0, 96]]

**s1260**
- [[20, 116], [16, 112], [0, 96]]
- [[39, 135], [39, 135], [0, 96]]
- [[39, 135], [39, 135], [0, 96]]

## Segmentation and surface metrics

| Case | Class | ASSD mm before / after | HD95 mm before / after | IoU before / after |
|---|---|---|---|---|
| s0346 | brain | 4.003569 / 2.287035 | 3.354102 / 2.121320 | 0.966730 / 0.972790 |
| s0346 | skull | 6.669031 / 4.765272 | 91.455112 / 4.743416 | 0.829155 / 0.829680 |
| s0792 | brain | 0.561777 / 0.563712 | 2.121320 / 1.500000 | 0.974692 / 0.975020 |
| s0792 | skull | 0.436128 / 0.465440 | 1.500000 / 1.500000 | 0.912988 / 0.905508 |
| s0874 | brain | 1.276649 / 0.765863 | 1.500000 / 1.500000 | 0.975657 / 0.972183 |
| s0874 | skull | 5.897334 / 6.319580 | 5.408327 / 6.538348 | 0.847606 / 0.838102 |
| s1161 | brain | 158.499824 / 153.362252 | 643.977096 / 644.450784 | 0.450363 / 0.462344 |
| s1161 | skull | 70.706234 / 68.261959 | 548.875088 / 546.817486 | 0.579760 / 0.608007 |
| s1260 | brain | 0.505063 / 0.590134 | 1.500000 / 1.500000 | 0.978141 / 0.974140 |
| s1260 | skull | 0.551128 / 0.533577 | 1.500000 / 1.500000 | 0.899350 / 0.901683 |

## Integration and changes

Actual frozen checkpoint -> calibrated probabilities/entropy -> saved phase3 anatomical V -> Q -> frozen selection threshold -> local frozen inference -> inverse transforms -> mean logits -> selected-only replacement -> saved segmentation/surface checks was executed on all five cases.
Notebook 18 and its generator now load historical calibration/thresholds and write to a new replay directory. CASE_001 files and all protected original outputs were hash-checked unchanged. No test payloads, retraining or split changes.

Files changed:
- `src/uncertainty/brain_skull.py`
- `src/refinement/rescue.py`
- `scripts/build_anatomy_phases4_10_notebook.py`
- `18_anatomy_aware_framework_phases4_10.ipynb`
- `scripts/verify_module3.py`
- `tests/test_module3_verification.py`

## Limitations

- Five development-validation cases, also used historically for calibration/threshold selection; these are not independent generalization estimates.
- Dataset reference labels are not newly expert-verified patient ground truth.
- V ambiguity uses phase3 uncalibrated probabilities; preserved to retain frozen threshold validity.
- B is a dilated/Gaussian-smoothed voxel boundary signal, not physical curvature.
- TTA averages four logit fields (identity plus three single-axis flips), not all eight flip combinations. Positive scalar T does not change argmax of their mean.
- Overlapping ROIs use deterministic later-ROI-wins replacement; candidate count differs from visited and actually changed voxel counts.
- Surface distances use existing deterministic caps at 200,000 boundary points, not exact full-surface distances.
- roi_margin_voxels=8 is an unused legacy config entry; actual ROIs are fixed 96 cubes centered on weighted components. No margin promised in Module 3.
- All anatomical V components recomputed end-to-end on s0346; saved component aggregation, probability ambiguity and SDF sign verified on all five cases.

Full machine-readable evidence: [verification_report.json](verification_report.json). Executed unit/regression results: [tests.xml](../tests.xml).
