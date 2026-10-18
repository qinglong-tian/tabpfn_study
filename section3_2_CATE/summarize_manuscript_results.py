"""Summarize the CATE rerun and reproduce the manuscript tables and figures.

The saved ``*_CATE`` values are already mean squared errors.  In particular,
the TabPFN values must not be squared again when they are summarized.
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SETUPS = ("A", "B", "C", "D", "E", "F")
SAMPLE_SIZES = (500, 1000, 2000)
NOISE_VARIANCES = (0.5, 1.0, 2.0)
EXPECTED_METHODS = {
    "AutoML": ("oracle", "dml", "dr", "slearner", "tlearner", "xlearner"),
    "TabPFN": ("dr_tabpfn", "slearner_tabpfn", "tlearner_tabpfn", "xlearner_tabpfn"),
}
METHOD_LABELS = {
    "oracle": "Oracle",
    "dml": "R",
    "dr": "DR",
    "slearner": "S",
    "tlearner": "T",
    "xlearner": "X",
}
FIGURE_METHOD_ORDER = ("S", "DR", "X", "T", "R", "Oracle")
TABLE_COLUMNS = (
    ("DR", "TabPFN"),
    ("DR", "AutoML"),
    ("S", "TabPFN"),
    ("S", "AutoML"),
    ("T", "TabPFN"),
    ("T", "AutoML"),
    ("X", "TabPFN"),
    ("X", "AutoML"),
    ("R", "AutoML"),
    ("Oracle", "AutoML"),
)
BACKEND_DIRECTORIES = {
    "AutoML": "cate_automl_paper",
    "TabPFN": "cate_tabpfn_paper",
}


def parse_args() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Reproduce the manuscript CATE summaries and figures."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_directory / "output" / "save_data",
        help="Directory containing cate_automl_paper and cate_tabpfn_paper.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=script_directory / "summary",
        help="Destination for summary tables, validation files, and figures.",
    )
    parser.add_argument("--expected-repetitions", type=int, default=100)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser.parse_args()


def normalize_method(method: str) -> str:
    return METHOD_LABELS[method.removesuffix("_tabpfn")]


def load_results(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    for backend, directory_name in BACKEND_DIRECTORIES.items():
        directory = output_root / directory_name
        for result_file in sorted(
            directory.glob("setup_*/ss_*/variance_*/d_*/case_*.pickle")
        ):
            with result_file.open("rb") as handle:
                result = pickle.load(handle)
            metadata = result.get("_metadata", {})
            setup = str(
                metadata.get("setup", result_file.parts[-5].removeprefix("setup_"))
            )
            n = int(metadata.get("n", result_file.parts[-4].removeprefix("ss_")))
            noise_variance = float(
                metadata.get(
                    "noise_variance",
                    result_file.parts[-3].removeprefix("variance_"),
                )
            )
            d = int(metadata.get("d", result_file.parts[-2].removeprefix("d_")))
            seed = int(metadata.get("seed", result_file.stem.removeprefix("case_")))
            n_x = int(metadata.get("n_heterogeneity_features", d))
            metadata_rows.append(
                {
                    "Setup": setup,
                    "N": n,
                    "D": d,
                    "SigmaSquared": noise_variance,
                    "Seed": seed,
                    "BaseLearner": backend,
                    "NFeaturesForCATE": n_x,
                    "File": str(result_file),
                }
            )
            found_methods = []
            for key, value in result.items():
                if not key.endswith("_CATE"):
                    continue
                method = key.removesuffix("_CATE")
                mse = float(value)
                found_methods.append(method)
                rows.append(
                    {
                        "Setup": setup,
                        "N": n,
                        "D": d,
                        "SigmaSquared": noise_variance,
                        "Method": normalize_method(method),
                        "MethodKey": method,
                        "BaseLearner": backend,
                        "Repetition": seed,
                        "MSE": mse,
                    }
                )
            if tuple(sorted(found_methods)) != tuple(sorted(EXPECTED_METHODS[backend])):
                raise ValueError(
                    f"Unexpected methods in {result_file}: {sorted(found_methods)}"
                )
    return pd.DataFrame(rows), pd.DataFrame(metadata_rows)


def validate_results(
    results: pd.DataFrame,
    metadata: pd.DataFrame,
    expected_repetitions: int,
) -> None:
    if results.empty or metadata.empty:
        raise ValueError("No CATE result files were found.")
    if not np.isfinite(results["MSE"]).all():
        raise ValueError("At least one stored MSE is not finite.")
    if (results["MSE"] < 0).any():
        raise ValueError("At least one stored MSE is negative.")
    if set(metadata["D"].unique()) != {6}:
        raise ValueError(f"Expected D=6, found {sorted(metadata['D'].unique())}.")
    if set(metadata["NFeaturesForCATE"].unique()) != {6}:
        raise ValueError(
            "Expected n_heterogeneity_features=6, found "
            f"{sorted(metadata['NFeaturesForCATE'].unique())}."
        )
    expected_seeds = set(range(1, expected_repetitions + 1))
    problems = []
    for backend in BACKEND_DIRECTORIES:
        backend_metadata = metadata[metadata["BaseLearner"] == backend]
        for setup in SETUPS:
            for n in SAMPLE_SIZES:
                for noise_variance in NOISE_VARIANCES:
                    mask = (
                        (backend_metadata["Setup"] == setup)
                        & (backend_metadata["N"] == n)
                        & (
                            backend_metadata["SigmaSquared"].eq(noise_variance)
                        )
                    )
                    seeds = set(backend_metadata.loc[mask, "Seed"].astype(int))
                    if seeds != expected_seeds:
                        problems.append(
                            {
                                "BaseLearner": backend,
                                "Setup": setup,
                                "N": n,
                                "SigmaSquared": noise_variance,
                                "MissingSeeds": sorted(expected_seeds - seeds),
                                "UnexpectedSeeds": sorted(seeds - expected_seeds),
                            }
                        )
    if problems:
        raise ValueError(f"Incomplete result grid: {problems[:5]}")


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "Setup",
        "N",
        "D",
        "SigmaSquared",
        "Method",
        "BaseLearner",
    ]
    summary = (
        results.groupby(group_columns, observed=True)["MSE"]
        .agg(
            Repetitions="size",
            MeanMSE="mean",
            MedianMSE="median",
            StandardDeviation="std",
            Q1=lambda values: values.quantile(0.25),
            Q3=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    summary["StandardError"] = summary["StandardDeviation"] / np.sqrt(
        summary["Repetitions"]
    )
    return summary


def manuscript_table(summary: pd.DataFrame) -> pd.DataFrame:
    medians = summary.pivot_table(
        index=["Setup", "N", "SigmaSquared"],
        columns=["Method", "BaseLearner"],
        values="MedianMSE",
        observed=True,
    )
    medians = medians.reindex(columns=pd.MultiIndex.from_tuples(TABLE_COLUMNS))
    medians.columns = [f"{method}_{backend}" for method, backend in TABLE_COLUMNS]
    return medians.reset_index()


def latex_value(value: float, rank: int) -> str:
    formatted = f"{value:.4f}"
    if rank == 0:
        return rf"\textbf{{{formatted}}}"
    if rank == 1:
        return rf"\underline{{{formatted}}}"
    return formatted


def write_latex_rows(table: pd.DataFrame, setups: tuple[str, ...], destination: Path) -> None:
    lines = ["% Generated by section3_2_CATE/summarize_manuscript_results.py"]
    value_columns = [f"{method}_{backend}" for method, backend in TABLE_COLUMNS]
    for setup_index, setup in enumerate(setups):
        setup_rows = table[table["Setup"] == setup].sort_values(
            ["N", "SigmaSquared"]
        )
        for row_index, (_, row) in enumerate(setup_rows.iterrows()):
            values = row[value_columns].astype(float).to_numpy()
            ranks = np.argsort(values, kind="stable")
            rank_by_column = {int(column_index): rank for rank, column_index in enumerate(ranks)}
            setup_cell = rf"\multirow{{9}}{{*}}{{{setup}}}" if row_index == 0 else ""
            n_cell = str(int(row["N"])) if row_index % 3 == 0 else ""
            sigma_squared = float(row["SigmaSquared"])
            sigma_cell = (
                str(int(sigma_squared))
                if sigma_squared.is_integer()
                else str(sigma_squared)
            )
            formatted_values = [
                latex_value(value, rank_by_column[index])
                for index, value in enumerate(values)
            ]
            cells = [setup_cell, n_cell, sigma_cell, *formatted_values]
            lines.append(" & ".join(cells) + " \\\\")
        if setup_index != len(setups) - 1:
            lines.append(r"\midrule")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bootstrap_median_interval(
    values: np.ndarray,
    bootstrap_repetitions: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(
        0,
        len(values),
        size=(bootstrap_repetitions, len(values)),
    )
    bootstrap_medians = np.median(values[bootstrap_indices], axis=1)
    low, high = np.quantile(bootstrap_medians, [0.025, 0.975])
    return float(low), float(high)


def plot_figure(
    results: pd.DataFrame,
    setup: str,
    n: int,
    noise_variance: float,
    destination: Path,
    bootstrap_repetitions: int,
) -> None:
    filtered = results[
        (results["Setup"] == setup)
        & (results["N"] == n)
        & (results["SigmaSquared"].eq(noise_variance))
    ]
    figure, axis = plt.subplots(figsize=(7.8, 5.8), dpi=100)
    centers = np.arange(len(FIGURE_METHOD_ORDER), dtype=float)
    width = 0.36
    colors = {"TabPFN": "#b11810", "AutoML": "#ffa5a1"}
    legend_handles = {}
    interval_seed = 0
    for method_index, method in enumerate(FIGURE_METHOD_ORDER):
        present_backends = [
            backend
            for backend in ("TabPFN", "AutoML")
            if not filtered[
                (filtered["Method"] == method)
                & (filtered["BaseLearner"] == backend)
            ].empty
        ]
        if len(present_backends) == 2:
            positions = {
                "TabPFN": centers[method_index] - width / 2,
                "AutoML": centers[method_index] + width / 2,
            }
        else:
            positions = {present_backends[0]: centers[method_index]}
        for backend in present_backends:
            values = filtered.loc[
                (filtered["Method"] == method)
                & (filtered["BaseLearner"] == backend),
                "MSE",
            ].to_numpy(dtype=float)
            median = float(np.median(values))
            low, high = bootstrap_median_interval(
                values,
                bootstrap_repetitions,
                seed=interval_seed,
            )
            interval_seed += 1
            container = axis.bar(
                positions[backend],
                median,
                width=width,
                color=colors[backend],
                yerr=np.array([[median - low], [high - median]]),
                error_kw={"ecolor": "#3f3f3f", "elinewidth": 2, "capsize": 0},
                label=backend,
            )
            legend_handles.setdefault(backend, container)
    sigma_label = (
        str(int(noise_variance))
        if float(noise_variance).is_integer()
        else str(noise_variance)
    )
    axis.set_title(rf"Setup {setup}-$n$={n}, $\sigma^2$={sigma_label}", pad=12)
    axis.set_xlabel("Estimator")
    axis.set_ylabel("MSE")
    axis.set_xticks(centers, FIGURE_METHOD_ORDER)
    axis.set_yscale("log")
    if setup == "E" and n == 500 and noise_variance == 2.0:
        axis.legend(
            [legend_handles["TabPFN"], legend_handles["AutoML"]],
            ["TabPFN", "AutoML"],
            title="Base Learner",
        )
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def write_figures(
    results: pd.DataFrame,
    figure_directory: Path,
    bootstrap_repetitions: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "mathtext.fontset": "dejavuserif",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "axes.labelsize": 30,
            "axes.titlesize": 30,
            "font.size": 25,
            "legend.fontsize": 20,
            "xtick.labelsize": 25,
            "ytick.labelsize": 25,
        }
    )
    for setup in SETUPS:
        for n, noise_variance in ((500, 2.0), (2000, 0.5)):
            variance_label = (
                str(int(noise_variance))
                if noise_variance.is_integer()
                else str(noise_variance)
            )
            destination = (
                figure_directory
                / f"setup_{setup}_n_{n}_var_{variance_label}.png"
            )
            plot_figure(
                results,
                setup,
                n,
                noise_variance,
                destination,
                bootstrap_repetitions,
            )


def main() -> None:
    args = parse_args()
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    results, metadata = load_results(args.output_root)
    validate_results(results, metadata, args.expected_repetitions)
    summary = summarize(results)
    table = manuscript_table(summary)

    results[results["BaseLearner"] == "AutoML"].to_csv(
        args.summary_dir / "cate_automl_full_results.csv",
        index=False,
    )
    results[results["BaseLearner"] == "TabPFN"].to_csv(
        args.summary_dir / "cate_tabpfn_full_results.csv",
        index=False,
    )
    metadata.to_csv(args.summary_dir / "CATE_result_inventory.csv", index=False)
    summary.to_csv(args.summary_dir / "CATE_summary_statistics.csv", index=False)
    table.to_csv(args.summary_dir / "CATE_median_manuscript_table.csv", index=False)
    write_latex_rows(
        table,
        ("A", "B", "C"),
        args.summary_dir / "CATE_median_table_A-C.tex",
    )
    write_latex_rows(
        table,
        ("D", "E", "F"),
        args.summary_dir / "CATE_median_table_D-F.tex",
    )
    write_figures(
        results,
        args.summary_dir / "figure",
        args.bootstrap_repetitions,
    )

    file_counts = metadata.groupby("BaseLearner").size().to_dict()
    print(f"AutoML result files: {file_counts.get('AutoML', 0)}")
    print(f"TabPFN result files: {file_counts.get('TabPFN', 0)}")
    print(f"Result rows: {len(results)}")
    print(f"Summary directory: {args.summary_dir}")
    print(f"Figures: {len(SETUPS) * 2}")


if __name__ == "__main__":
    main()
