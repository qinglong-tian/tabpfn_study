"""Export the completed real-data run to Table 4 and a comparison figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np
import pandas as pd

from real_data_experiment import HERE, LABELS, TABLE_METHODS, summarize

COLORS = {"krr_test_oracle": "#757575", "pl": "#987326", "xgb": "#252525", "iw": "#346b94", "tabpfn": "#ad526d"}
MARKERS = {"krr_test_oracle": "s", "pl": "D", "xgb": "^", "iw": "o", "tabpfn": "*"}
DATASET_LABELS = {"airfoil": "Airfoil", "concrete": "Concrete"}


def export_results(results_dir, output_dir=None):
    results_dir = Path(results_dir)
    output_dir = results_dir if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((results_dir / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Cannot publish incomplete experimental results.")
    summary = summarize(results_dir)
    repeats = manifest["protocol"]["config"]["repeats"]
    indexed = summary.set_index(["dataset", "setting", "method"])
    rows = []
    md = ["# Corrected real-data covariate-shift results", "", f"Mean target test MSE ± standard error across {repeats} seeded 70/30 splits; lower is better.", "", "| Dataset | Shift | " + " | ".join(LABELS[m] for m in TABLE_METHODS) + " |", "|---|---|" + "---:|" * len(TABLE_METHODS)]
    for dataset in ("airfoil", "concrete"):
        for setting in ("beta1", "beta2"):
            row = [DATASET_LABELS[dataset], "β₁" if setting == "beta1" else "β₂"]
            cells = []
            best = min(indexed.loc[(dataset, setting, method), "mean"] for method in TABLE_METHODS)
            for method in TABLE_METHODS:
                stat = indexed.loc[(dataset, setting, method)]
                text = f"{stat['mean']:.2f} ± {stat['se']:.2f}"
                row.append(text)
                cells.append(f"**{text}**" if stat["mean"] == best else text)
            rows.append(row)
            md.append("| " + " | ".join(row[:2] + cells) + " |")
    table = pd.DataFrame(rows, columns=["Dataset", "Shift"] + [LABELS[m] for m in TABLE_METHODS])
    table.to_csv(output_dir / "table4.csv", index=False)
    md += ["", "The KRR test-label oracle selects its penalty using the same target test labels used for evaluation; it is an optimistic benchmark, not the true-conditional-mean oracle from the synthetic experiment.", "", "IW uses exponential density-ratio weights during both training and source-validation scoring. Every method is evaluated with ordinary MSE on the same resampled target observations. XGBoost uses five-fold CV with the original 32-candidate grid. TabPFN uses the V2 checkpoint and eight ensemble estimators.", "", "Features retain the public notebook's preprocessing: raw covariates standardized using source-training means and standard deviations; no log transforms. The β vectors operate on these standardized covariates. Both settings share each source split and baseline fits. Resampling is seeded separately for each dataset, shift, and repetition.", "", "Reported SE is the sample SD of the ten split-level MSEs divided by √10. Repeated splits overlap and share a finite dataset, so these SEs describe variability across repetitions and are not population-level confidence intervals. Bold identifies the smallest mean, without a significance claim.", "", "Full precision and the extra KRR source-selected comparator: `summary.csv`. Individual errors: `per_split_mse.csv`. Weight and resampling diagnostics: `diagnostics.csv`. Raw predictions, CV traces, indices, versions, checkpoint checksum, and protocol are retained in the results directory.", ""]
    (output_dir / "table4.md").write_text("\n".join(md))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.8), sharey=True)
    for row_no, dataset in enumerate(("airfoil", "concrete")):
        group = summary[(summary.dataset == dataset) & summary.method.isin(TABLE_METHODS)]
        xmax = float((group["mean"] + group["se"]).max()) * 1.10
        xmin = float((group["mean"] - group["se"]).min()) / 1.35
        if xmin <= 0:
            raise ValueError("Log-axis figure requires strictly positive mean-minus-SE.")
        for col_no, setting in enumerate(("beta1", "beta2")):
            ax = axes[row_no, col_no]
            for pos, method in enumerate(TABLE_METHODS):
                stat = indexed.loc[(dataset, setting, method)]
                # Point and interval express mean ± SE, not a confidence interval.
                ax.errorbar(stat["mean"], pos, xerr=stat["se"], fmt=MARKERS[method], color=COLORS[method], markersize=9 if method == "tabpfn" else 6, capsize=3, linewidth=1.5)
            ax.set_yticks(range(len(TABLE_METHODS)), [LABELS[m] for m in TABLE_METHODS])
            ax.set_ylim(len(TABLE_METHODS) - 0.5, -0.5)
            ax.set_xscale("log")
            ax.set_xlim(xmin, xmax)
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=20))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
            ax.set_xlabel("Target test MSE (log scale; lower is better)")
            ax.set_title(f"{DATASET_LABELS[dataset]} · " + ("β₁" if setting == "beta1" else "β₂"), loc="left", fontweight="bold", pad=12)
            ax.grid(axis="x", color="#e5e5e5", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0)
    fig.suptitle("Prediction under covariate shift", x=0.055, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.055, 0.925, f"Corrected IW training and evaluation · {repeats} repetitions · points show mean ± 1 SE", fontsize=11, color="#555555")
    fig.text(0.055, 0.028, "All methods use the same unweighted target-test MSE. KRR oracle uses test labels to select its penalty.\nLogarithmic axes match within each dataset row. Response units differ by dataset. SE reflects repeated-split variability.", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0.02, 0.08, 0.99, 0.9], h_pad=2.5, w_pad=2)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"real_data_comparison.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.3, 3.6))
    ax.axis("off")
    headers = ["Dataset", "Shift", "KRR oracle*", "PL", "Naive XGB", "IW XGB", "TabPFNv2"]
    visual_table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", colWidths=[0.105, 0.055, 0.17, 0.17, 0.17, 0.17, 0.16], bbox=[0.00, 0.28, 1, 0.61])
    visual_table.auto_set_font_size(False)
    visual_table.set_fontsize(10.5)
    for (r, c), cell in visual_table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.set_linewidth(0.5)
        cell.set_facecolor("#f0f3f5" if r == 0 else "white")
        if r == 0:
            cell.set_text_props(weight="bold")
    for r, (dataset, setting) in enumerate((d, s) for d in ("airfoil", "concrete") for s in ("beta1", "beta2")):
        best = int(np.argmin([indexed.loc[(dataset, setting, m), "mean"] for m in TABLE_METHODS]))
        visual_table[(r + 1, best + 2)].set_text_props(weight="bold")
    fig.text(0.04, 0.92, f"Table 4 · Corrected target prediction MSE (mean ± SE, {repeats} splits)", fontsize=15, fontweight="bold")
    fig.text(0.04, 0.10, "*KRR oracle selects its penalty using test labels; it is an optimistic benchmark. All methods share the same target-test metric.\nIW weights training and validation losses. Bold marks the smallest mean, without a significance claim.", fontsize=9.5, color="#555555")
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.9)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"table4.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results/real_data")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(export_results(args.results_dir, args.output_dir).to_string(index=False))
