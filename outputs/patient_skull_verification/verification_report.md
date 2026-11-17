# CASE_001 skull reconstruction verification

**Assessment:** New anatomical and confidence-colored outputs have exactly the same patient geometry. Technical provenance and geometry checks pass. Independent anatomical accuracy is not established; the reference is a CT-aligned prediction requiring expert review.

## Patient verification

The original patient reconstruction is CASE_001. The old phase-8 confidence model is **s0346**, an official validation subject. Comparing them as identical anatomy was invalid. CASE_001 bone and brain kernels have matching patient/study identifiers but distinct reconstruction series; DICOM manifests retain the exact UIDs and hashes.

CT: `F:\csstdy\outputs\10_totalsegmentator_sharp_skull\input\CASE_001_bone_kernel_ct.nii.gz`. SHA-256: `29deec8768052bee85f26b2eec8abe773128532202391c8c777e8a7600307db1`. Shape: [512, 512, 227]; spacing: [0.5557460784912109, 0.5557460784912109, 0.75] mm; array orientation: ['L', 'A', 'S']; physical coordinates: RAS+ mm.

Every slice in each 227-slice DICOM series was checked at five known pixel locations using DICOM LPS orientation/position/spacing, converted to NIfTI RAS. Pixel HU differences were zero. This establishes conversion and grid consistency, not a clinical landmark validation.

## Confirmed causes and unresolved hypotheses

- **Different skulls:** confirmed patient/source mismatch, CASE_001 versus s0346.
- **Confidence-dependent loss:** old phase 8 colored the phase-7 adaptive mesh. It did not threshold confidence voxels. The old confidence/adaptive geometry identity is True. Geometry reduction occurred upstream in spatial vertex clustering; full/native geometry is now the default.
- **Facial detail:** patient mismatch, different segmentation sources and native versus 1.5 mm model-grid resolution prevent a direct same-anatomy conclusion. Notebook 09 additionally uses step-size 2 and five smoothing iterations. No unsupported facial bone was synthesized.
- **Neural rendering:** notebook 11 uses predicted SIREN HU >=250 within a dilated skull neighborhood, a minimum 100-voxel component size, at most 12 components and marching-cubes step-size 2. This is another derived approximation of CASE_001, not the acquired CT or an independent anatomical reference. It is retained as a historical experiment.
- **Mandible:** craniofacial task has a separate mandible label. Earlier skull-only rendering omitted that separate label. Notebook 07 also globally retained the largest component, which can discard separate anatomy. The available mandible reaches the inferior acquired boundary; a complete lower jaw is outside the available evidence.
- **Cranial opening:** visible absence of a continuous bone rim on native CT slices 80, 107, 133, 160 and 187. It is not explained solely by mesh coloring. Preserved unchanged. Postoperative etiology is plausible from the supplied annotations but is not established here; do not label it a fracture.
- **Detached fragments:** source-label components are enumerated with physical volume, location, HU statistics and FOV contact. Their medical validity is unresolved; all are retained.
- **Surface fragmentation:** old s0346 full mesh had 3 non-manifold edges; adaptive clustering increased this to 3661. It merged spatially close vertices and removed degenerate/duplicate faces. New CASE_001 mesh has 0 non-manifold edges and 1256 boundary edges. Native voxel surfaces retain voxel-scale irregularity. Acquired-FOV boundaries are not capped.
- **Alignment:** native CASE_001 labels pass grid checks. Earlier notebook 07 converts array coordinates through DICOM directions/origin into LPS, while NIfTI workflows export RAS. Direct overlays require the documented LPS-to-RAS sign conversion, not arbitrary fitting. New exports use the NIfTI affine.

## Corrections and confidence definition

`src/geometry/patient_surface.py`: native-resolution mask surfaces, no FOV padding, no smoothing/decimation/filtering, explicit affine sampling, fail-closed patient/hash checks, unavailable values grey and geometry digests.
`src/geometry/confidence.py`: phase 8 now uses baseline_full_resolution.ply; rejects entropy-grid mismatches; no nearest-boundary extrapolation; checks geometry identity and reports invalid coverage. Existing phase outputs are preserved.
`configs/anatomy_aware_phases4_10_v1.yaml`: red/yellow/green legend. No training, split, calibration or selection parameters changed.
`scripts/infer_patient_confidence.py`: frozen phase-3 MONAI inference on CASE_001 CT resampled in RAS at 1.5 mm, using established two HU windows and fixed existing temperature. Full logits/probabilities saved. This is external-case inference, not held-out dataset evaluation.
`scripts/verify_patient_skull.py`: DICOM/source audit, per-class native surfaces, quantitative checks, exports and this report. `tests/test_patient_surface.py`: regression checks.

Main CASE_001 legend: **P(skull | CT) = softmax(logits / 1.1064573224712873)[2]**. Red=0, yellow=0.5, green=1; grey=unavailable. Background and brain are retained in the softmax denominator; no foreground-only renormalization. Mandible/teeth have no dedicated model probability and remain grey. These are MONAI predictions sampled on TotalSegmentator geometry, not confidence emitted by TotalSegmentator.

Diagnostic entropy is -sum(p log p)/log(3), across background/brain/skull. Higher entropy means greater predictive uncertainty. It is neither epistemic uncertainty nor class-specific skull probability. The legacy 1-entropy metric is retained only with explicit predictive-certainty labeling. Temperature was not refitted; calibration transfer to this external scan is unverified.

## Quantitative verification

| Label | Voxels | Volume mm3 | Components (6-neighbor) |
|---|---:|---:|---:|
| skull | 2116682 | 490308.807 | 28 |
| mandible | 33992 | 7873.916 | 2 |
| teeth_upper | 5614 | 1300.429 | 9 |
| teeth_lower | 0 | 0.000 | 0 |

New combined mesh: 627,326 vertices, 1,253,806 triangles. Anatomical versus confidence displacement, Hausdorff distance and bounding-box difference: **0 mm**, from exact vertex/connectivity identity. Segmentation voxels added/removed: **0/0**. Removed component volume: **0 mm3**.
Skull confidence invalid coverage: 0.000000%. Full per-class/FOV/topology diagnostics are in JSON. Original-smoothed to native nearest-vertex distances: {'metric': 'symmetric nearest-vertex distances; not exact triangle distances', 'mean_mm': 0.0570348419215269, 'p95_mm': 0.08204211260425946, 'max_mm': 5.079384560565026, 'bounding_box_max_difference_mm': 0.412109375}.
Total vs craniofacial skull model agreement: Dice=0.868922, IoU=0.768225. This is not ground-truth accuracy; the class scopes differ.
Regression tests: {'tests': 19, 'failures': 0, 'errors': 0, 'skipped': 0}. Initial test invocation encountered a Windows temporary-directory permission error; rerun uses a workspace-local temporary directory.

## Visual evidence

- [Interactive skull + partial mandible/teeth confidence](skull_plus_mandible_confidence.html)
- [Interactive skull-only confidence](skull_confidence.html)
- [GLB](meshes/skull_plus_mandible_confidence.glb) / [anatomical PLY](meshes/skull_plus_mandible_anatomical.ply) / [colored PLY](meshes/skull_plus_mandible_confidence.ply)
- [Identical geometry comparison](figures/geometry_confidence_comparison.png) / [numeric legend](figures/confidence_legend.png)
- [Original versus corrected extraction](figures/original_vs_corrected.png) / [components before/after](figures/components_before_after.png)
- [CT intensity evidence](figures/ct_bone_evidence.png) / [alignment overlays](figures/alignment_axial.png)
- [Raw CT and opening overlays](figures/opening_raw_vs_overlay.png) / [opening levels](figures/cranial_opening.png)
- [Frontal](figures/frontal.png), [lateral](figures/lateral.png), [superior](figures/superior.png), [inferior](figures/inferior.png), [oblique](figures/oblique.png)
- [Predictive uncertainty only](figures/uncertainty_only.png)

- [Corrected s0346 full-resolution predictive-certainty display](s0346_corrected/s0346_confidence.html), explicitly a different patient.
- [Source notebook audit](notebook_source_audit.json), [file inventory](file_inventory.json), [per-component evidence](components.json), [export checks](artifact_checks.json).

## Completion conditions

| Condition | Status | Evidence |
|---|---|---|
| Both original outputs traceable | PASS | CASE_001 versus s0346: mismatch explicitly identified |
| Original outputs same patient | FAIL | Different patient and acquisition; new matched outputs generated |
| CT and masks aligned | PASS | Affines, shapes, all-slice DICOM pixel sampling, round trips |
| Skull and mandible class handling | PASS | Separate source labels retained; mandible partial at FOV |
| Confidence does not remove anatomy | PASS | No thresholding; exact geometry identity |
| Confidence matches selected anatomical surface | PASS | Identical vertices and triangles, including exported PLY |
| Cranial opening investigated | PASS | Raw CT and overlays through multiple native slices |
| No CT defect filled | PASS | Zero segmentation edits, no fill/closing/capping |
| Components anatomically justified | PASS | Conservative retain-all policy; per-component CT evidence recorded |
| Identical comparison cameras | PASS | Shared RAS bounds, parallel scale and view for every comparison |
| Automated regression tests executed | PASS | {'tests': 19, 'failures': 0, 'errors': 0, 'skipped': 0} |
| 3D and static exports | PASS | PLY, GLB, HTML and PNG |
| Modifications and limitations documented | PASS | Report and source audit |
| Independent anatomical accuracy verified | NOT_ESTABLISHED | No expert ground truth; no accuracy claim |

## Remaining limitations

- No independently expert-verified segmentation or labeled landmarks supplied.
- All selected reference labels remain model predictions, even when spatially and visually consistent with CT.
- Mandible extends beyond inferior CT FOV; no complete jaw can be reconstructed.
- No separate frontal/parietal/temporal/occipital/orbital/maxillary class labels; regional completeness cannot be certified.
- Thin bones and detached components need expert review; no component removed without evidence.
- External CASE_001 confidence calibration transfer is unverified; no calibration changes made.

No independently labeled anatomical landmarks or expert review is supplied. Major cranial regions are inspected visually but cannot be individually certified from a single composite skull label. Corrected means corrected extraction/provenance/confidence handling, not a clinically certified segmentation. Original experiment artifacts and protected data hashes remain unchanged. The official 1,228-subject split manifest was fingerprinted before/after; no test payload was evaluated or tuned.

## Reproduction

From repository root: `.venv/Scripts/python.exe scripts/verify_patient_skull.py`. Frozen inference is cached only when its source/checkpoint/calibration contract and output hashes match. Each surface is re-extracted deterministically; no stochastic mesh sampling. CUDA deterministic warnings are enabled, so cross-device bitwise inference reproducibility is not claimed.

Coordinate convention: [NiBabel affine documentation](https://nipy.org/nibabel/coordinate_systems.html). Temperature method: [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html). These references support implementation conventions, not patient-specific anatomical conclusions.
