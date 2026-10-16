"""Independently reconstruct splits, sampling, errors and SEs from saved runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from real_data_experiment import HERE, METHODS, SHIFT_VECTORS, load_dataset


def validate(results_dir, refit_check=False):
    results_dir = Path(results_dir)
    manifest = json.loads((results_dir / "manifest.json").read_text())
    assert manifest["complete"], "Run incomplete"
    protocol = manifest["protocol"]
    assert hashlib.sha256((HERE / "real_data_experiment.py").read_bytes()).hexdigest() == protocol["runner_sha256"]
    assert hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() == manifest["protocol_hash"]
    config = protocol["config"]
    records, independent = [], []
    for dataset in protocol["datasets"]:
        X, y, path = load_dataset(dataset)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol["data_sha256"][dataset]
        for repeat in range(config["repeats"]):
            train, pool = train_test_split(np.arange(len(y)), test_size=config["test_size"], random_state=config["seed"] + repeat)
            assert not set(train) & set(pool)
            assert len(set(train) | set(pool)) == len(y)
            scaler = StandardScaler().fit(X[train])
            X_train, X_pool = scaler.transform(X[train]), scaler.transform(X[pool])
            for setting_no, (setting, beta) in enumerate(SHIFT_VECTORS[dataset].items()):
                file = results_dir / "runs" / f"{dataset}_{setting}_{repeat:02d}.json"
                r = json.loads(file.read_text())
                records.append(r)
                assert_array_equal(r["train_indices"], train)
                assert_array_equal(r["test_pool_indices"], pool)
                assert_array_equal(r["shift_beta"], beta)
                assert_allclose(r["scaler_mean"], scaler.mean_)
                assert_allclose(r["scaler_scale"], scaler.scale_)
                assert r["split_seed"] == config["seed"] + repeat
                assert r["resample_seed"] == config["seed"] + 10000 * (1 + ["airfoil", "concrete"].index(dataset)) + 1000 * setting_no + repeat
                # Independent normalization and explicit reconstruction of draws.
                probs = np.exp(X_pool @ beta)
                probs /= probs.sum()
                idx = np.random.default_rng(r["resample_seed"]).choice(len(pool), len(pool), replace=True, p=probs)
                assert_array_equal(idx, r["resampled_pool_indices"])
                w = np.exp(X_train @ beta)
                assert_allclose(r["training_weight_ess"], w.sum() ** 2 / (w @ w))
                assert_allclose(r["target_reservoir_ess"], 1 / (probs @ probs))
                assert r["n_unique_target"] == len(set(idx))
                with np.load(file.with_suffix(".npz")) as pred:
                    assert_array_equal(pred["y_true"], y[pool[idx]])
                    assert_array_equal(pred["source_row"], pool[idx])
                    for method in METHODS:
                        assert pred[method].shape == pred["y_true"].shape
                        assert np.isfinite(pred[method]).all()
                        error = float(sum(float(a - b) ** 2 for a, b in zip(pred["y_true"], pred[method])) / len(idx))
                        assert_allclose(error, r["mse"][method], rtol=1e-12, atol=1e-12)
                        independent.append({"dataset": dataset, "setting": setting, "repeat": repeat, "method": method, "mse": error})
                # Best IW parameters must minimize the recorded weighted CV risk.
                cv = json.loads((results_dir / "cv" / file.name).read_text())
                assert len(cv) == 32
                assert r["iw_best"] == min(cv, key=lambda row: row["validation_mse"])
                ordinary = json.loads((results_dir / "cache" / f"{dataset}_{repeat:02d}.json").read_text())
                assert ordinary["best"] == min(ordinary["cv"], key=lambda row: row["validation_mse"])
                if refit_check and repeat == 0:
                    # Independently refit one selected IW model per setting,
                    # and recompute its weighted OOF risk from raw inputs.
                    params = r["iw_best"]["params"]
                    def make_model():
                        return XGBRegressor(objective="reg:squarederror", tree_method="hist", n_jobs=1, random_state=r["split_seed"], eval_metric="rmse", **params)
                    oof = np.empty(len(train))
                    for fit_idx, valid_idx in KFold(n_splits=config["cv_folds"], shuffle=True, random_state=r["split_seed"]).split(X_train):
                        model = make_model().fit(X_train[fit_idx], y[train][fit_idx], sample_weight=w[fit_idx] / w[fit_idx].mean())
                        oof[valid_idx] = model.predict(X_train[valid_idx])
                    weighted_risk = np.sum(w * (y[train] - oof) ** 2) / w.sum()
                    assert_allclose(weighted_risk, r["iw_best"]["validation_mse"], rtol=1e-6)
                    model = make_model().fit(X_train, y[train], sample_weight=w / w.mean())
                    with np.load(file.with_suffix(".npz")) as pred:
                        assert_allclose(model.predict(X_pool[idx]), pred["iw"], rtol=1e-6, atol=1e-6)
    expected = len(protocol["datasets"]) * 2 * config["repeats"]
    assert len(records) == manifest["completed_runs"] == expected
    assert len(list((results_dir / "runs").glob("*.json"))) == expected
    raw = pd.DataFrame(independent)
    saved_raw = pd.read_csv(results_dir / "per_split_mse.csv")
    merged = raw.merge(saved_raw, on=["dataset", "setting", "repeat", "method"], validate="one_to_one")
    assert len(merged) == expected * len(METHODS)
    assert_allclose(merged.mse_x, merged.mse_y, rtol=1e-12, atol=1e-12)
    saved = pd.read_csv(results_dir / "summary.csv").set_index(["dataset", "setting", "method"])
    for key, group in raw.groupby(["dataset", "setting", "method"]):
        values = group.mse.to_numpy()
        mean = sum(values) / len(values)
        se = np.sqrt(sum((values - mean) ** 2) / (len(values) - 1) / len(values))
        assert_allclose(saved.loc[key, "mean"], mean, rtol=1e-12)
        assert_allclose(saved.loc[key, "se"], se, rtol=1e-12)
        assert saved.loc[key, "count"] == config["repeats"]
    print(f"Verified {expected} runs, {len(independent)} method-level MSEs, all seeds/draws/indices, and all means/SEs.")
    if refit_check:
        print("Independently refitted IW models and weighted CV risk for repetition 0 of all four settings.")
    return {"runs": expected, "mse_values": len(independent), "iw_refit_checks": 4 if refit_check else 0, "status": "passed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results/real_data")
    parser.add_argument("--refit-check", action="store_true")
    args = parser.parse_args()
    validate(args.results_dir, args.refit_check)
