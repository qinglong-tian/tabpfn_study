# TabPFN: One Model to Rule Them All?

This is the official implementation for the paper "TabPFN: One Model to Rule Them All?" (Journal of the American Statistical Association).

All implementations are in Python. The semi-supervised learning experiment additionally relies on R code for the SLZ baseline (via `rpy2`).

---

## Requirements

Install [TabPFN](https://github.com/PriorLabs/TabPFN) (TabPFNv2) before running any experiment:
```bash
pip install tabpfn
```
The pretrained model weights are downloaded automatically on first use. For offline environments, follow the TabPFN offline-usage instructions and pass `model_path=` wherever `TabPFNClassifier()` or `TabPFNRegressor()` is called.

Other key dependencies: `scikit-learn`, `numpy`, `pandas`, `econml`, `flaml`, `rpy2` (Study I only), `pycasso` (Section 4.1 only), `liac-arff` (Appendix D UQ real data only).

---

## Repository structure

| Directory | Paper section | Outputs reproduced |
|---|---|---|
| `section3_1_semisupervised/` | §3.1 Study I: semi-supervised estimation | Figure 1; Figures 8–13 (appendix); Tables 2–3 |
| `section3_2_CATE/` | §3.2 Study II: CATE estimation | Figure 2; Figure 3; Tables 7–8 (appendix) |
| `section3_3_covariate_shift/` | §3.3 Study III: prediction under covariate shift | Figure 4; Figure 15 (appendix); Table 4 |
| `section4_1_adaptivity_regression/` | §4.1 Adaptivity in regression | Figure 5; Table 5; Figure 21 (appendix) |
| `section4_2_adaptivity_classification/` | §4.2 Adaptivity in classification | Figures 6–7 |
| `appendixD_GP_comparison/` | Appendix D: comparison with Gaussian process regression | Figures 16–18 (appendix); Tables 9–10 (appendix); real-data UQ |

---

## Section 3.1 — Study I: Parameter estimation in a semi-supervised setting

```bash
cd section3_1_semisupervised
```

The SLZ method and simulation settings are implemented in R. You need `rpy2` to call R from Python. Before running the notebooks, set your `R_HOME` variable (find it by running `R.home()` in an R session).

### Simulation (Figure 1; Figures 8–13 in appendix)

Run the three Jupyter notebooks corresponding to the three working models:

- `run_linear.ipynb` — linear regression setting
- `run_logistic.ipynb` — logistic regression setting
- `run_quantile.ipynb` — quantile regression setting

Each notebook iterates over feature dimensions `p ∈ {4, …, 9}`, labeled sample sizes `n ∈ {300, 500}`, and unlabeled sample sizes `m ∈ {500, 1000, 2000}`, with 500 simulation replicates per configuration.

### Real data (Tables 2–3)

```bash
cd section3_1_semisupervised/real_data
python reg_ssl_comparison.py          # linear regression on 5 OpenML datasets (Table 2)
python clf_ssl_comparison_mean_imputation.py  # logistic regression on 5 OpenML datasets (Table 3)
```

Datasets are fetched automatically from OpenML. `rpy2` is required for the SLZ baseline.

---

## Section 3.2 — Study II: Heterogeneous treatment effects (CATE) estimation

```bash
cd section3_2_CATE
```

### Simulation (Figure 2; Tables 7–8 in appendix)

The main script is `CATE_comparison.py`. Key arguments:

| Argument | Description | Values used in paper |
|---|---|---|
| `--setup` | Data-generating setup | `A`, `B`, `C`, `D`, `E`, `F` |
| `--ss` | Training set size | `500`, `1000`, `2000` |
| `--seed` | Random seed | 1–100 |
| `--tabpfn` | Use TabPFN as base learner (omit for AutoML) | flag |
| `--d` | Feature dimension | `6` (default) |
| `--var` | Noise variance σ² | `0.5`, `1`, `2` |

Example (matches `run-demo.sh`):
```bash
python CATE_comparison.py --setup A --seed 1 --ss 500 --tabpfn
```

The paper reports the median test MSE over 100 repetitions for each of the 6 setups × 3 sample sizes × 3 noise levels (Tables 7–8 in the appendix). The two representative setups A and E are shown in Figure 2.

Helper files: `cate.py` defines all CATE estimators; `data_gen.py` generates synthetic datasets; `utils.py` contains utility functions.

### Real data — ACIC 2017 benchmark (Figure 3)

**Data download**: The ACIC 2017 semi-synthetic datasets are based on the IHDP covariates and available from Richard Hahn's website: https://math.la.asu.edu/~prhahn/. Download the 6,000 datasets and place them in `section3_2_CATE/real_data/`.

```bash
cd section3_2_CATE/real_data
bash run.sh          # runs all setup × scenario × method combinations
```

Or run individual configurations:
```bash
python acic2017.py --setup iid --scenario 000 --method s
```

Arguments: `--setup` ∈ {`iid`, `nonadditive`, `group_corr`}; `--scenario` ∈ {`000`, `001`, …, `111`}; `--method` ∈ {`s`, `t`, `x`, `dr`}.

---

## Section 3.3 — Study III: Prediction under covariate shift

```bash
cd section3_3_covariate_shift
```

### Simulation (Figure 4; Figure 15 in appendix)

Open and run all cells in `covariate_shift_simu.ipynb`.

The simulation considers 5 mean functions with Gaussian-mixture source/target covariate distributions, training sizes `n ∈ {500, 600, …, 1200}`, averaged over 500 replicates. `KRR.py` (adapted from https://github.com/kw2934/KRR) implements the kernel ridge regression baselines.

### Real data — Airfoil Self-Noise & Concrete Compressive Strength (Table 4)

Open and run all cells in `covariate_shift_application.ipynb`.

- **Airfoil** (n = 1503, 5 covariates): downloaded automatically from the UCI repository.
- **Concrete** (n = 1030, 8 covariates): bundled in `Concrete_Data.xls`.

Each dataset uses a 70/30 train–test split with test resampling to induce covariate shift (two β settings per dataset, 10 repetitions each).

Note: `lable_shift.ipynb` is supplementary exploratory analysis of label shift not reported in the main paper.

---

## Section 4.1 — Adaptivity in regression

```bash
cd section4_1_adaptivity_regression
```

### Prediction performance comparison (Figure 5; Table 5)

```bash
python comparisons.py --n 50 --d 100 --s 1 --seed 1
```

| Argument | Description | Values used in paper |
|---|---|---|
| `--n` | Training sample size | `50`, `500` |
| `--d` | Number of features | `100` |
| `--s` | Sparsity (number of non-zero coefficients) | `1`, `5`, `10`, `20`, `30` |
| `--seed` | Random seed | 1–100 |

The paper averages results over 100 seeds for each of 5 (sparsity) × 3 (design/beta type) × 4 (SNR) × 2 (sample size) = 120 configurations. See `run-demo.sh` for a minimal example.

### ALE plots for descriptive accuracy (Figure 21 in appendix)

Open and run `ALE_plot.ipynb`.

---

## Section 4.2 — Adaptivity in classification

```bash
cd section4_2_adaptivity_classification
```

Run both scripts (each iterates over all noise levels `ρ ∈ {0.1, 0.2, 0.3, 0.4}` with 1000 repetitions):

```bash
python knn.py              # kNN (clean) and kNN (noisy)
python comparison_LDA.py   # LDA (clean/noisy) and TabPFN (clean/noisy)
```

Combine the outputs of both scripts to reproduce Figures 6 and 7. Data are generated from models M1 (Gaussian class-conditional) and M2 (nonlinear Bayes boundary), with training sizes `n` up to 10,000.

---

## Appendix D — Comparison with Gaussian process regression

```bash
cd appendixD_GP_comparison
```

### 1D and 2D interpolation & extrapolation (Figures 16–18 in appendix)

- `extrapolation_1D.ipynb` — 1D experiments comparing TabPFN and GPR on linear, quadratic, step, and piecewise-linear functions; reproduces Figures 16–17.
- `extrapolation_2D.ipynb` — 2D experiments on the same function classes; reproduces Figure 18.

Training data are drawn from `[−1, 1]` (or `[−1, 1]²` in 2D); predictions are evaluated over `[−4, 4]` to assess both interpolation and extrapolation.

### D.4 Uncertainty quantification — synthetic data (Tables 9–10 in appendix)

```bash
cd appendixD_GP_comparison/UQ
python UQ.py --n 11 --seed 1 --f linear
```

| Argument | Description | Values used in paper |
|---|---|---|
| `--n` | Training set size | `11`, `31` |
| `--seed` | Random seed | 1–100 |
| `--f` | True function | `linear`, `quadratic`, `step`, `piecewiselinear` |

This computes coverage of the 95% prediction interval for f (the true function) across the interpolation and extrapolation regions, comparing TabPFN and GPR. Results are saved as pickle files in `output_coverage/raw_data/`. The full experiment is 4 functions × 2 sizes × 100 seeds = 800 runs; use `submit.sh` for batch execution on a cluster.

Aggregate and visualize results (Tables 9–10) in `summary.ipynb`.

### D.4 Uncertainty quantification — real data (new experiment)

**Data download**: Download the following 5 datasets in ARFF format from [OpenML](https://www.openml.org/) and place them in `appendixD_GP_comparison/UQ/data/reg/`:

| Dataset name | OpenML ID | Target column |
|---|---|---|
| kin8nm | 189 | `y` |
| house_8L | 218 | `price` |
| puma8NH | 225 | `thetadd3` |
| BNG (stock) | 1200 | `company10` |
| cmc | 45052 | `class` |

```bash
cd appendixD_GP_comparison/UQ
python real_data_calibration.py --dataset kin8nm --seed 1
```

This calibrates the predictive distribution for y (the response) on real datasets, evaluating coverage at multiple PI levels, PIT, and CRPS. Results are saved in `output_realdata/raw/`. Run all 5 datasets × 20 seeds via:

```bash
bash submit_realdata.sh
```

Aggregate and visualize results in `real_data_summary.ipynb`.

---

## License

This code is offered under the [Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) license.
