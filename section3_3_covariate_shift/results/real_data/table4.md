# Corrected real-data covariate-shift results

Mean target test MSE ± standard error across 10 seeded 70/30 splits; lower is better.

| Dataset | Shift | KRR test-label oracle | PL | Naive XGBoost | IW XGBoost | TabPFNv2 |
|---|---|---:|---:|---:|---:|---:|
| Airfoil | β₁ | 12.60 ± 1.51 | 14.12 ± 1.66 | 11.64 ± 3.93 | 11.74 ± 3.90 | **3.11 ± 0.76** |
| Airfoil | β₂ | 8.47 ± 0.55 | 8.99 ± 0.62 | 3.57 ± 0.44 | 3.60 ± 0.19 | **1.65 ± 0.15** |
| Concrete | β₁ | 47.54 ± 13.16 | 97.23 ± 20.44 | 5.73 ± 1.40 | 5.28 ± 0.71 | **2.56 ± 0.45** |
| Concrete | β₂ | 61.12 ± 22.44 | 120.60 ± 31.10 | 7.80 ± 1.05 | 6.99 ± 0.85 | **4.95 ± 0.67** |

The KRR test-label oracle selects its penalty using the same target test labels used for evaluation; it is an optimistic benchmark, not the true-conditional-mean oracle from the synthetic experiment.

IW uses exponential density-ratio weights during both training and source-validation scoring. Every method is evaluated with ordinary MSE on the same resampled target observations. XGBoost uses five-fold CV with the original 32-candidate grid. TabPFN uses the V2 checkpoint and eight ensemble estimators.

Features retain the public notebook's preprocessing: raw covariates standardized using source-training means and standard deviations; no log transforms. The β vectors operate on these standardized covariates. Both settings share each source split and baseline fits. Resampling is seeded separately for each dataset, shift, and repetition.

Reported SE is the sample SD of the ten split-level MSEs divided by √10. Repeated splits overlap and share a finite dataset, so these SEs describe variability across repetitions and are not population-level confidence intervals. Bold identifies the smallest mean, without a significance claim.

Full precision and the extra KRR source-selected comparator: `summary.csv`. Individual errors: `per_split_mse.csv`. Weight and resampling diagnostics: `diagnostics.csv`. Raw predictions, CV traces, indices, versions, checkpoint checksum, and protocol are retained in the results directory.
