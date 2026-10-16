# Corrected real-data covariate-shift experiments

This reruns **Table 4** for Airfoil and Concrete under both beta vectors, with
ten repetitions per setting. The changes address the real-data notebook's IW
training/evaluation error and its missing beta2 runs. The synthetic experiment
in `KRR.py` already passes density ratios to gradient-boosting training; its
results and Figures 4/15 have not been recomputed by this real-data correction.

## Results

- [Updated Table 4](results/real_data/table4.md), also as
  [CSV](results/real_data/table4.csv), [PNG](results/real_data/table4.png), and
  [PDF](results/real_data/table4.pdf).
- [Comparison figure](results/real_data/real_data_comparison.png), also as
  [PDF](results/real_data/real_data_comparison.pdf) and
  [SVG](results/real_data/real_data_comparison.svg).
- [Full-precision summary](results/real_data/summary.csv), including the extra
  KRR source-selected comparator retained from the original notebook.
- [All 240 split-level MSEs](results/real_data/per_split_mse.csv).
- [Weight and resampling diagnostics](results/real_data/diagnostics.csv).
- [Protocol, package versions, and checksums](results/real_data/manifest.json).

`runs/` contains a JSON record and compressed prediction archive for each of
the 40 setting/repetition combinations. These retain source/test indices,
resampled row indices, seeds, scaler parameters, predictions, labels, selected
hyperparameters, and density-weight diagnostics. `cv/` retains the weighted CV
scores for all candidates. `cache/` retains unweighted CV scores and baseline
predictions shared by the two shifts on each source split.

## What changed

1. IW fits XGBoost with `sample_weight` proportional to
   `exp(X_train @ shift_beta)`. Training weights have mean one, including inside
   each CV training fold, to keep XGBoost's regularization scale comparable.
   Ratios are computed stably and are not clipped or tuned.
2. IW hyperparameters are selected separately from the naive model, using
   density-weighted out-of-fold squared errors on source validation samples.
   Explicit scoring ensures that validation weights are used as well as
   training weights. Test labels do not enter XGBoost selection or training.
3. Final evaluation is ordinary MSE on the same shifted test observations for
   every method. There is no second application of the density ratio to test
   errors.
4. Both beta vectors from Appendix C.2 are included. Each dataset has the same
   ten source splits for both beta settings. Split seeds are 42 through 51;
   resampling has its own deterministic seed, recorded for every run.
5. XGBoost uses five-fold CV, consistent with the manuscript, rather than the
   old notebook's three folds. Its 32-candidate parameter grid is unchanged.
6. The notebook now calls the reusable experiment/export code and displays the
   completed, validated four-setting results instead of leaving partial runs
   and interrupted output in the file.

These are fresh results. The old resampling was unseeded, so changes in reported
numbers cannot be attributed solely to correcting IW. TabPFN is fixed to the
paper's V2 regression checkpoint, eight estimators, and explicit split seeds;
no model substitution or tuning on the observed results is used.

## Protocol and interpretation

For each split, use 70% of the observations as source training data and 30% as
the source test reservoir. Fit `StandardScaler` on source training observations
only. As in the public notebook, the features are **raw features followed by
standardization**, without logarithmic transforms. The beta vectors act on
these standardized features. This explicitly records the public implementation;
the manuscript's Airfoil description mentions log-transformed features, which
is a separate preprocessing discrepancy and is not silently introduced here.

Resample the reservoir with replacement, preserving its size, with probabilities
proportional to `exp(X_test @ shift_beta)`. The same resampled observations and
labels are used by all methods. Heavy tilting can produce few unique target
rows or very small training-weight effective sample sizes; see the diagnostics.
The experiment does not cap weights to improve any method's results.

The two vectors are:

| Dataset | beta1 | beta2 |
|---|---|---|
| Airfoil | `[-1, 1, -1, 1, -1]` | `[-1, 0, 0, 0, 1]` |
| Concrete | `[-1, 1, -1, 1, -1, 1, -1, 1]` | `[-1, 0, 0, 0, 0, 0, 0, 1]` |

KRR's RBF kernel, source split, pseudo-label imputation, and lambda grid retain
the original notebook's implementation. The real-data **KRR test-label oracle**
uses target test labels to choose the penalty and reports MSE on that same set.
This is an optimistic, test-informed benchmark. It is **not** Wang's oracle
that has access to the true conditional mean in the synthetic experiment. It
is labeled explicitly rather than presented as a deployable method.

Each table cell is the mean MSE plus/minus the sample standard deviation of
the ten MSE values divided by sqrt(10). These repeated splits overlap; the SEs
describe variation across repetitions, not population-level confidence
intervals or significance tests. Bold only identifies the lowest mean.

## Reproduce

Use Python 3.9 and install `requirements-real-data.txt` in an isolated environment.
Supply the original V2 checkpoint `tabpfn-v2-regressor.ckpt` (not V2.5 or newer).
The exact checkpoint SHA-256 is recorded in the manifest. Airfoil is bundled in
`data/`, and Concrete remains in `Concrete_Data.xls`.

From this directory:

```bash
pip install -r requirements-real-data.txt
export TABPFN_DISABLE_TELEMETRY=1
export TABPFN_ALLOW_CPU_LARGE_DATASET=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
python real_data_experiment.py \
  --model-path /path/to/tabpfn-v2-regressor.ckpt \
  --output-dir results/real_data_rerun
python validate_real_data_results.py --results-dir results/real_data_rerun --refit-check
python export_real_data_results.py --results-dir results/real_data_rerun
python -m unittest test_real_data_experiment.py -v
```

The runner resumes completed repetitions only when the full protocol hash
matches, including data/model/runner checksums and dependency versions. Use a
new output directory when changing the protocol. CPU is the recorded device;
different libraries or hardware can change numerical results.

To inspect the committed outputs, open `covariate_shift_application.ipynb` and
run all cells. To rerun inside the notebook, set `RERUN=True` and provide
`TABPFN_V2_MODEL_PATH`; notebook reruns go into a separate directory.

The validator independently reconstructs every split and target draw from its
seed, verifies all saved prediction-level MSEs without evaluation weights,
checks CV selections, and recomputes every mean and SE. The regression tests
check that weights reach every fit and validation score, and that a zero shift
reduces IW to ordinary XGBoost.

Reference protocol: [paper, Section 3.3 and Appendix C.2](https://arxiv.org/html/2505.20003v2).
