"""Build Notebook 18, continuing the anatomy-aware framework through Phase 10."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "18_anatomy_aware_framework_phases4_10.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str, tag: str):
    cell = nbf.v4.new_code_cell(dedent(source).strip())
    cell.metadata["tags"] = [tag]
    return cell


cells = [
    md(
        """
        # 18 — Anatomy-aware framework: Phases 4–10

        This notebook continues the frozen Phase 1–3 brain/skull experiment. It adds calibrated predictive
        uncertainty, selective local rescue, reconstruction-aware surface evaluation, curvature/uncertainty-adaptive
        meshes, mesh confidence, a model-derived confidence envelope, and the final ablation package.

        Phases 4–9 use only the five development-validation cases. Phase 10 may access the official test set only
        after every preceding configuration, threshold, checkpoint, and representative mesh-selection rule has been
        serialized into a frozen protocol.
        """
    ),
    code(
        """
        import hashlib
        import json
        import os
        import sys
        import uuid
        from pathlib import Path

        import matplotlib.pyplot as plt
        import nibabel as nib
        import numpy as np
        import pandas as pd
        import torch
        import yaml
        from IPython.display import Image, display

        PROJECT_ROOT = Path.cwd().resolve()
        if not (PROJECT_ROOT / "src").is_dir():
            raise RuntimeError("Run from the F:/csstdy project root")
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        from src.geometry import (
            run_adaptive_meshes,
            run_confidence_envelopes,
            run_surface_confidence,
            run_surface_evaluation,
        )
        from src.refinement import run_validation_rescue
        from src.training.anatomy_phase2 import build_phase2_model, sha256_file
        from src.uncertainty import run_validation_uncertainty

        CONFIG_PATH = PROJECT_ROOT / "configs" / "anatomy_aware_phases4_10_v1.yaml"
        PHASE2_CONFIG_PATH = PROJECT_ROOT / "configs" / "anatomy_aware_v1.yaml"
        PHASE3_CONFIG_PATH = PROJECT_ROOT / "configs" / "anatomy_aware_phase3_v1.yaml"
        FRAMEWORK_REPORT_PATH = PROJECT_ROOT / "outputs" / "anatomy_aware_framework" / "framework_report.json"
        PHASE3_REPORT_PATH = PROJECT_ROOT / "outputs" / "anatomy_aware_framework" / "phase3" / "phase3_report.json"
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        phase2_config = yaml.safe_load(PHASE2_CONFIG_PATH.read_text(encoding="utf-8"))
        phase3_config = yaml.safe_load(PHASE3_CONFIG_PATH.read_text(encoding="utf-8"))
        framework_report = json.loads(FRAMEWORK_REPORT_PATH.read_text(encoding="utf-8"))
        phase3_report = json.loads(PHASE3_REPORT_PATH.read_text(encoding="utf-8"))
        assert framework_report["completed_phases"] == [1, 2, 3]
        assert phase3_report["status"] == "passed"
        assert phase3_report["official_test_split_accessed"] is False

        frozen_phase4 = json.loads((PROJECT_ROOT / config["outputs"]["phase4"] / "phase4_report.json").read_text())
        frozen_phase5 = json.loads((PROJECT_ROOT / config["outputs"]["phase5"] / "phase5_report.json").read_text())
        assert frozen_phase4["temperature_scaling"]["temperature"] == 1.1064573224712873
        replay_root = PROJECT_ROOT / "outputs/anatomy_aware_framework/replays" / uuid.uuid4().hex
        output_paths = {key: replay_root / key for key in config["outputs"] if key != "root"}
        for path in output_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((PROJECT_ROOT / "data" / "cache" / "anatomy_aware_phase2").glob("*/metadata.json"))
        ]
        validation_records = [item for item in records if item["split"] == "val"]
        assert len(records) == 103 and len(validation_records) == 5
        checkpoint_path = PROJECT_ROOT / config["locks"]["phase3_checkpoint"]
        assert sha256_file(checkpoint_path) == phase3_report["model_report"]["best_checkpoint_sha256"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = build_phase2_model(phase2_config).to("cuda")
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        DEVICE = torch.device("cuda")
        ROI = int(phase2_config["inference"]["roi_size"][0])
        OVERLAP = float(phase2_config["inference"]["overlap"])
        print("GPU:", torch.cuda.get_device_name(0))
        print("Frozen Phase 3 checkpoint:", checkpoint_path)
        """,
        "phase4-10-setup",
    ),
    md(
        """
        ## Phase 4 — Calibrated predictive uncertainty

        A single positive temperature is fitted on sampled validation logits by minimizing multiclass negative log
        likelihood. Temperature scaling does not change the argmax segmentation. Normalized three-class predictive
        entropy is saved as a voxelwise map and evaluated as a detector of segmentation error.
        """
    ),
    code(
        """
        phase4_report = run_validation_uncertainty(
            model,
            validation_records,
            config["phase4_uncertainty"],
            output_paths["phase4"],
            device=DEVICE,
            roi=ROI,
        overlap=OVERLAP,
        frozen_calibration_report=frozen_phase4,
            progress=lambda message: print(message, flush=True),
        )
        calibration_table = pd.DataFrame(
            [phase4_report["calibration_before"], phase4_report["calibration_after"]],
            index=["before", "after"],
        )
        display(calibration_table)
        display(pd.DataFrame([phase4_report["uncertainty_error_analysis"]]).T)
        print("Temperature:", phase4_report["temperature_scaling"]["temperature"])
        print("Entropy threshold:", phase4_report["selected_error_threshold"]["threshold"])
        """,
        "phase4-calibration",
    ),
    code(
        """
        example_id = "s0346"
        example_record = next(item for item in validation_records if item["case_id"] == example_id)
        image = np.load(example_record["image_path"], mmap_mode="r")[0]
        label = np.load(example_record["label_path"], mmap_mode="r")
        prediction = np.asarray(nib.load(str(output_paths["phase4"] / "cases" / example_id / "initial_prediction.nii.gz")).dataobj)
        entropy = np.asarray(nib.load(str(output_paths["phase4"] / "cases" / example_id / "predictive_entropy.nii.gz")).dataobj)
        z = int(np.rint(np.argwhere(label == 1)[:, 2].mean()))
        fig, axes = plt.subplots(1, 4, figsize=(15, 4), dpi=150)
        axes[0].imshow(image[:, :, z].T, cmap="gray", origin="lower", vmin=.35, vmax=.75); axes[0].set_title("CT")
        axes[1].imshow(label[:, :, z].T, cmap="viridis", origin="lower", vmin=0, vmax=2); axes[1].set_title("Reference")
        axes[2].imshow(prediction[:, :, z].T, cmap="viridis", origin="lower", vmin=0, vmax=2); axes[2].set_title("Phase 3 prediction")
        shown = axes[3].imshow(entropy[:, :, z].T, cmap="magma", origin="lower", vmin=0, vmax=1); axes[3].set_title("Calibrated entropy")
        fig.colorbar(shown, ax=axes[3], fraction=.046, pad=.04)
        for axis in axes: axis.axis("off")
        fig.tight_layout()
        phase4_figure = output_paths["phase4"] / "uncertainty_overlay_s0346.png"
        fig.savefig(phase4_figure, bbox_inches="tight")
        plt.show()
        """,
        "phase4-visual",
    ),
    md(
        """
        ## Phase 5 — Uncertainty/anatomy-triggered selective rescue

        The rescue priority combines calibrated entropy `U(x)`, Phase 3 anatomical violation `V(x)`, and a local
        boundary signal `B(x)`. A validation-selected threshold creates candidate components. At most three `96³`
        ROIs per case are refined by local flip test-time ensembling with the frozen Phase 3 model. Only selected
        voxels inside those ROIs may be replaced; the full volume is not passed through a second global network.
        """
    ),
    code(
        """
        phase5_report = run_validation_rescue(
            model,
            validation_records,
            phase4_report,
            config["phase5_rescue"],
            output_paths["phase3"] if "phase3" in output_paths else PROJECT_ROOT / "outputs/anatomy_aware_framework/phase3/qc_phase3",
            output_paths["phase4"],
            output_paths["phase5"],
        device=DEVICE,
        frozen_priority_threshold=frozen_phase5["selected_priority_threshold"],
        progress=lambda message: print(message, flush=True),
        )
        rescue_table = pd.DataFrame([
            {"organ": organ, **values} for organ, values in phase5_report["means"].items()
        ])
        display(rescue_table.style.format(precision=5))
        print("Mean rescued voxel fraction:", phase5_report["mean_rescue_voxel_fraction"])
        print("Total ROIs:", phase5_report["total_rois"])
        """,
        "phase5-rescue",
    ),
    code(
        """
        priority = np.asarray(nib.load(str(output_paths["phase5"] / "cases" / example_id / "rescue_priority.nii.gz")).dataobj)
        rescue_mask = np.asarray(nib.load(str(output_paths["phase5"] / "cases" / example_id / "rescue_mask.nii.gz")).dataobj)
        refined = np.asarray(nib.load(str(output_paths["phase5"] / "cases" / example_id / "refined_prediction.nii.gz")).dataobj)
        fig, axes = plt.subplots(1, 4, figsize=(15, 4), dpi=150)
        axes[0].imshow(prediction[:, :, z].T, cmap="viridis", origin="lower", vmin=0, vmax=2); axes[0].set_title("Initial")
        p = axes[1].imshow(priority[:, :, z].T, cmap="inferno", origin="lower", vmin=0, vmax=1); axes[1].set_title("Q(x)")
        axes[2].imshow(rescue_mask[:, :, z].T, cmap="Reds", origin="lower", vmin=0, vmax=1); axes[2].set_title("Selected rescue")
        axes[3].imshow(refined[:, :, z].T, cmap="viridis", origin="lower", vmin=0, vmax=2); axes[3].set_title("Refined")
        fig.colorbar(p, ax=axes[1], fraction=.046, pad=.04)
        for axis in axes: axis.axis("off")
        fig.tight_layout()
        phase5_figure = output_paths["phase5"] / "selective_rescue_s0346.png"
        fig.savefig(phase5_figure, bbox_inches="tight")
        plt.show()
        """,
        "phase5-visual",
    ),
    md(
        """
        ## Phase 6 — Reconstruction-aware surface evaluation

        Initial and selectively refined masks are evaluated using overlap, sensitivity, precision, ASSD, HD95,
        Hausdorff distance, surface RMSE, surface Dice, volume error, Chamfer distance, surface-area error, and mesh
        complexity. This is an evaluation/optimization bridge; it does not claim a differentiable renderer.
        """
    ),
    code(
        """
        phase6_report = run_surface_evaluation(
            validation_records,
            output_paths["phase4"],
            output_paths["phase5"],
            config["phase6_surface_evaluation"],
            output_paths["phase6"],
            progress=lambda message: print(message, flush=True),
        )
        surface_rows = []
        for organ, stages in phase6_report["aggregate"].items():
            for stage, metrics in stages.items():
                surface_rows.append({"organ": organ, "stage": stage, **metrics})
        surface_table = pd.DataFrame(surface_rows)
        surface_table.to_csv(output_paths["phase6"] / "surface_metrics.csv", index=False)
        display(surface_table[["organ", "stage", "dice", "assd_mm", "hd95_mm", "surface_dice_2mm", "symmetric_chamfer_mm", "surface_area_error_percent"]].style.format(precision=4))
        """,
        "phase6-surface-evaluation",
    ),
    md(
        """
        ## Phase 7 — Curvature/uncertainty-adaptive reconstruction

        The representative validation case was declared in configuration before reconstruction. Surface priority
        combines predictive entropy, vertex-normal curvature, local boundary complexity, and a spacing-anisotropy
        penalty. Low-priority regions use 3 mm vertex clustering, medium-priority regions 1.5 mm, and high-priority
        regions retain full marching-cubes detail.
        """
    ),
    code(
        """
        representative_id = config["phase7_adaptive_mesh"]["representative_validation_case"]
        mesh_records = [item for item in validation_records if item["case_id"] == representative_id]
        phase7_report = run_adaptive_meshes(
            mesh_records,
            output_paths["phase4"],
            output_paths["phase5"],
            config["phase7_adaptive_mesh"],
            output_paths["phase7"],
            progress=lambda message: print(message, flush=True),
        )
        mesh_rows = []
        for case in phase7_report["cases"]:
            for organ, metrics in case["organs"].items():
                mesh_rows.append({"case_id": case["case_id"], "organ": organ, **metrics})
        display(pd.DataFrame(mesh_rows).style.format(precision=4))
        """,
        "phase7-adaptive-mesh",
    ),
    md(
        """
        ## Phase 8 — Confidence-aware 3D surfaces

        Calibrated entropy is interpolated from the CT grid to every adaptive-mesh vertex. Vertex confidence is
        `1 − entropy`. Full PLY files retain vertex colours, while the representative brain/skull pair is also exported
        as GLB and an interactive HTML scene.
        """
    ),
    code(
        """
        phase8_report = run_surface_confidence(
            mesh_records,
            output_paths["phase4"],
            output_paths["phase7"],
            config["phase8_surface_confidence"],
            output_paths["phase8"],
            representative_case=representative_id,
            progress=lambda message: print(message, flush=True),
        )
        display(pd.DataFrame([
            {"case_id": case["case_id"], "organ": organ, **metrics}
            for case in phase8_report["cases"] for organ, metrics in case["organs"].items()
        ]).style.format(precision=4))
        print("Interactive confidence mesh:", phase8_report["interactive_html"])
        """,
        "phase8-confidence",
    ),
    md(
        """
        ## Phase 9 — Preliminary confidence envelope

        Inner and outer surfaces are displaced along vertex normals by at most 3 mm in proportion to normalized
        predictive entropy. This is explicitly a model-derived visualization, not a statistically calibrated physical
        confidence interval.
        """
    ),
    code(
        """
        phase9_report = run_confidence_envelopes(
            mesh_records,
            output_paths["phase8"],
            config["phase9_confidence_envelope"],
            output_paths["phase9"],
            representative_case=representative_id,
            progress=lambda message: print(message, flush=True),
        )
        display(pd.DataFrame([
            {"case_id": case["case_id"], "organ": organ, **metrics}
            for case in phase9_report["cases"] for organ, metrics in case["organs"].items()
        ]).style.format(precision=4))
        print("Envelope GLB:", phase9_report["combined_glb"])
        """,
        "phase9-envelope",
    ),
    md(
        """
        ## Phase 10 — Frozen protocol, ablation, and final evaluation

        The remaining cells are populated only after Phases 4–9 pass. Their hashes, calibration temperature, rescue
        threshold, model checkpoint, mesh selection, and all hyperparameters are frozen before official test access.
        """
    ),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.update(
    {
        "kernelspec": {"display_name": "csstdy (.venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
        "project": {
            "notebook_number": 18,
            "framework_phase": [4, 5, 6, 7, 8, 9, 10],
            "official_test_locked_until_phase10": True,
        },
    }
)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
