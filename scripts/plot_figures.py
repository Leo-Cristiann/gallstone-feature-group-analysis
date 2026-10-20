"""Regenerate the manuscript's feature-contrast, SCSO, and ROC figures."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def figure_2(clinical_workbook: Path) -> None:
    data = pd.read_excel(clinical_workbook, sheet_name="PairedBootstrap4Models")
    data = data.loc[data["Metric"] == "Accuracy"]
    order = ["B-A", "C-A", "D-B", "D-C"]
    labels = [
        "B − A (add bioimpedance)",
        "C − A (add laboratory)",
        "D − B (add laboratory)",
        "D − C (add bioimpedance)",
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.15))
    for optimizer, marker, offset in [
        ("GridSearch", "o", -0.115),
        ("SCSO", "s", 0.115),
    ]:
        rows = data.loc[data["Optimizer"] == optimizer].set_index("Contrast").loc[order]
        point = rows["MeanDifference"].to_numpy(float)
        lower = rows["CI95_Lower"].to_numpy(float)
        upper = rows["CI95_Upper"].to_numpy(float)
        ax.errorbar(point, np.arange(4) + offset,
                    xerr=np.vstack((point - lower, upper - point)), fmt=marker,
                    lw=1.75, markersize=5.8, capsize=3.1, label=optimizer)
    ax.axvline(0, color="0.45", linestyle="--", linewidth=1.1)
    ax.set_yticks(np.arange(4), labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.09, 0.41)
    ax.set_xlabel("Mean held-out accuracy difference (95% paired bootstrap CI)", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=2, loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout(pad=0.65)
    fig.savefig(RESULTS / "figure_2_feature_contrasts.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def figure_3(scso_workbook: Path) -> None:
    data = pd.read_excel(scso_workbook, sheet_name="SCSOConvergence")
    models = sorted(data["model"].unique())
    seeds = sorted(data["scso_seed"].unique())
    evaluations = sorted(data["n_evals"].unique())
    scenarios = [
        ("A_baseline", "A: Baseline"),
        ("B_plus_bia", "B: Bioimpedance"),
        ("C_plus_lab", "C: Laboratory"),
        ("D_full", "D: Full"),
    ]
    lookup = data.set_index(["scso_seed", "scenario", "model", "n_evals"])["best_cv_accuracy"]
    fig, ax = plt.subplots(figsize=(3.68, 2.85))
    for scenario, label in scenarios:
        by_seed = np.array([
            [np.mean([lookup.loc[(seed, scenario, model, k)] for model in models])
             for k in evaluations]
            for seed in seeds
        ])
        mean = by_seed.mean(axis=0)
        sd = by_seed.std(axis=0, ddof=1)
        line = ax.plot(evaluations, mean, lw=1.65, label=label)[0]
        ax.fill_between(evaluations, mean - sd, mean + sd,
                        color=line.get_color(), alpha=0.15, linewidth=0)
    ax.set_xlim(10, 100)
    ax.set_ylim(0.575, 0.815)
    ax.set_xticks([10, 30, 50, 70, 90, 100])
    ax.set_xlabel("Fitness evaluations", fontsize=9)
    ax.set_ylabel("Best 5-fold CV accuracy", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center", bbox_to_anchor=(0.52, 0.33), ncol=2, frameon=False, fontsize=7.1)
    fig.tight_layout(pad=0.7)
    fig.savefig(RESULTS / "figure_3_scso_convergence.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def figure_4(clinical_workbook: Path) -> None:
    curves = pd.read_excel(clinical_workbook, sheet_name="ROC_LR_PlotData")
    summary = pd.read_excel(clinical_workbook, sheet_name="ROCAUC95CI")
    scenarios = [
        ("A_baseline", "A: Baseline"),
        ("B_plus_bia", "B: Bioimpedance"),
        ("C_plus_lab", "C: Laboratory"),
        ("D_full", "D: Full"),
    ]
    fig, ax = plt.subplots(figsize=(7.3, 5.0), dpi=165)
    for scenario, label in scenarios:
        xy = curves.loc[(curves["Optimizer"] == "GridSearch") &
                        (curves["Scenario"] == scenario), ["FPR", "TPR"]].astype(float)
        report = summary.loc[(summary["Optimizer"] == "GridSearch") &
                             (summary["Scenario"] == scenario)]
        if len(xy) < 3 or len(report) != 1:
            raise ValueError(f"Incomplete ROC results for {scenario}")
        auc = float(report.iloc[0]["TestROC_AUC"])
        if not np.isclose(np.trapezoid(xy["TPR"], xy["FPR"]), auc, atol=1e-7):
            raise ValueError(f"ROC curve and AUC mismatch for {scenario}")
        ax.step(xy["FPR"], xy["TPR"], where="post", linewidth=2.15,
                label=f"{label} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.15,
            alpha=0.55, label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.02),
           xlabel="False positive rate (1 − Specificity)",
           ylabel="True positive rate (Sensitivity)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18, linewidth=0.7)
    ax.legend(loc="lower right", fontsize=8.4, frameon=True)
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_4_logistic_roc.png", dpi=400,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=["2", "3", "4", "all"], default="all")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    clinical = RESULTS / "clinical_evaluation.xlsx"
    scso = RESULTS / "scso_multiseed.xlsx"
    if args.figure in ("2", "all"):
        figure_2(clinical)
    if args.figure in ("3", "all"):
        figure_3(scso)
    if args.figure in ("4", "all"):
        figure_4(clinical)
    print(f"Figures saved in {RESULTS}")


if __name__ == "__main__":
    main()
