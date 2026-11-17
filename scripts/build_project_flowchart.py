"""Render the complete CT segmentation research workflow as PNG and SVG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
PNG_PATH = ROOT / "PROJECT_FLOWCHART.png"
SVG_PATH = ROOT / "PROJECT_FLOWCHART.svg"


def add_node(axis, x, y, width, height, title, detail, *, color, edge):
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=color,
        edgecolor=edge,
        linewidth=1.5,
        zorder=3,
    )
    axis.add_patch(box)
    axis.text(x, y + 0.019, title, ha="center", va="center", fontsize=9.8, fontweight="bold", color="#132238", zorder=4)
    axis.text(x, y - 0.017, detail, ha="center", va="center", fontsize=7.5, color="#25364d", linespacing=1.15, zorder=4)
    return box


def arrow(axis, start, end, *, color="#586a80", connection="arc3,rad=0"):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.45,
            color=color,
            connectionstyle=connection,
            shrinkA=4,
            shrinkB=4,
            zorder=2,
        )
    )


fig, axis = plt.subplots(figsize=(20, 12), dpi=170)
axis.set_xlim(0, 1)
axis.set_ylim(0, 1)
axis.axis("off")
fig.patch.set_facecolor("white")

axis.text(
    0.5,
    0.975,
    "AI-Driven CT Multi-Organ Segmentation, Uncertainty, and 3D Digital-Twin Pipeline",
    ha="center",
    va="top",
    fontsize=18,
    fontweight="bold",
    color="#10233d",
)
axis.text(
    0.5,
    0.944,
    "CS16 M.Tech research implementation — TotalSegmentator CT v2.0.1 (not MRI)",
    ha="center",
    va="top",
    fontsize=10.5,
    color="#475a73",
)

lane_specs = [
    (0.025, 0.325, "A  DATA FOUNDATION", "#eef6ff", "#2c6da4"),
    (0.35, 0.625, "B  MODEL & TRAINING", "#effaf5", "#267758"),
    (0.65, 0.975, "C  CALIBRATION, TEST & 3D", "#f8f1ff", "#744ba1"),
]
for left, right, label, fill, edge in lane_specs:
    axis.add_patch(Rectangle((left, 0.035), right - left, 0.88, facecolor=fill, edgecolor=edge, linewidth=1.1, alpha=0.55, zorder=0))
    axis.text((left + right) / 2, 0.897, label, ha="center", va="center", fontsize=11, fontweight="bold", color=edge)

node_width = 0.255
node_height = 0.105
x1, x2, x3 = 0.175, 0.4875, 0.8125
y_six = [0.82, 0.68, 0.54, 0.40, 0.26, 0.12]

left_nodes = [
    ("1. Research objective", "CT volume → multi-organ segmentation\n→ uncertainty → 3D digital twin"),
    ("2. Dataset extracted", "TotalSegmentator CT v2.0.1\n1,228 NIfTI CT subjects; 117 masks each"),
    ("3. Programmatic dataset audit", "1,228/1,228 CT valid; 143,676 masks\nno missing/corrupt structural records"),
    ("4. Locked 10-organ schema", "Liver, spleen, kidneys, stomach, pancreas,\naorta, IVC, left/right lungs + background"),
    ("5. Immutable manifest & split", "Official patient split: 1,082 train / 57 val / 89 test\ntest rejected before I/O until final protocol"),
    ("6. Deterministic preprocessing", "RAS; 1.5 mm; HU [−1000,1000] → [0,1]\n96³ class-balanced patches; fingerprinted cache"),
]
for index, ((title, detail), y) in enumerate(zip(left_nodes, y_six)):
    add_node(axis, x1, y, node_width, node_height, title, detail, color="#dceeff", edge="#2c6da4")
    if index:
        arrow(axis, (x1, y_six[index - 1] - node_height / 2), (x1, y + node_height / 2))

middle_nodes = [
    ("7. CUDA environment verified", "RTX 4060 Laptop GPU (8 GB)\nPyTorch 2.13 + CUDA 13.0; MONAI 1.6"),
    ("8. Baseline preflight", "MONAI 3D U-Net; 4.82 M parameters\n11 output channels; DiceCE; test still locked"),
    ("9. Gate 1 — one-case overfit", "32 fixed patches from s0014; 50 epochs\nloss ↓84.5%; represented-class Dice 0.9463"),
    ("10. Gate 2 — deterministic pilot", "16 train / 4 validation; 3 epochs\nforced pause + exact checkpoint resume passed"),
    ("11. Gate 3 — full baseline", "1,082 train / 57 validation; 50 epochs\nAMP overflow skip/backoff; exact resume; test locked"),
    ("12. Fixed best checkpoint", "Validation macro Dice 0.891595\nSHA-256: 81632bd7…; 50 epochs complete"),
]
for index, ((title, detail), y) in enumerate(zip(middle_nodes, y_six)):
    color = "#def4e9" if index not in {2, 4} else "#fff1d6"
    edge = "#267758" if index not in {2, 4} else "#b87514"
    add_node(axis, x2, y, node_width, node_height, title, detail, color=color, edge=edge)
    if index:
        arrow(axis, (x2, y_six[index - 1] - node_height / 2), (x2, y + node_height / 2))

right_y = [0.82, 0.705, 0.59, 0.475, 0.315, 0.20, 0.085]
right_nodes = [
    ("13. Validation calibration", "1.14 M sampled validation voxels\nscalar temperature T = 1.035455; ECE improved"),
    ("14. Uncertainty ↔ error", "Calibrated normalized predictive entropy\nvalidation error AUROC 0.9844; AUPRC 0.3510"),
    ("15. Qualitative QA & decisions", "CT / truth / prediction / error / entropy overlays\nthreshold 0.231567; postprocessing = none"),
    ("16. Freeze protocol before test", "Protocol SHA-256: b0a5f323…\ncheckpoint, T, threshold, policy, test IDs, mesh rule"),
    ("17. One-shot locked test", "89/89 cases; no test-time tuning\nmacro Dice 0.8726; uncertainty AUROC 0.9792"),
    ("18. Patient-space reconstruction", "Predeclared case s0244; marching cubes\nvoxel vertices transformed by RAS affine in millimetres"),
    ("19. Final digital-twin artifacts", "Combined 10-organ GLB + interactive HTML\nfull-resolution per-organ PLY/STL + PNG preview"),
]
for index, ((title, detail), y) in enumerate(zip(right_nodes, right_y)):
    if index < 4:
        color, edge = "#eadcff", "#744ba1"
    elif index == 4:
        color, edge = "#ffe1e1", "#b43636"
    else:
        color, edge = "#dff5df", "#347b38"
    add_node(axis, x3, y, node_width, 0.088, title, detail, color=color, edge=edge)
    if index:
        arrow(axis, (x3, right_y[index - 1] - 0.044), (x3, y + 0.044), color="#6d5a82" if index < 4 else "#586a80")

# Lane headers carry the left-to-right continuation without crossing nodes.
arrow(axis, (0.315, 0.897), (0.36, 0.897), color="#267758")
arrow(axis, (0.615, 0.897), (0.66, 0.897), color="#744ba1")

# Explicit leakage boundary between validation choices and test access.
boundary_y = 0.397
axis.plot([0.67, 0.955], [boundary_y, boundary_y], linestyle=(0, (6, 4)), linewidth=2.2, color="#b43636", zorder=1)
axis.text(
    0.8125,
    boundary_y + 0.013,
    "LOCKED-TEST BOUNDARY — first test CT/label access only below this line",
    ha="center",
    va="bottom",
    fontsize=8.3,
    fontweight="bold",
    color="#9a2929",
    zorder=5,
)

axis.text(
    0.5,
    0.018,
    "Research safeguards: immutable hashes • original NIfTI files read-only • patient-level official split • resumable atomic artifacts • no test-time tuning",
    ha="center",
    va="center",
    fontsize=9,
    color="#3d5068",
)

fig.savefig(PNG_PATH, bbox_inches="tight", facecolor="white")
fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Wrote {PNG_PATH}")
print(f"Wrote {SVG_PATH}")
