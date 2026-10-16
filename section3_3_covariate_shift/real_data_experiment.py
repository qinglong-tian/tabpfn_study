"""Reproducible real-data covariate-shift experiment (replacement for Table 4).

IW weights SOURCE training losses and source-validation losses. Every final
score is ordinary MSE on exactly the same resampled target observations.
The real-data KRR oracle uses target test labels to select lambda: it is an
optimistic test-label benchmark, not the conditional-mean oracle of KRR.py.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
AIRFOIL_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
SHIFT_VECTORS = {
    "airfoil": {"beta1": [-1, 1, -1, 1, -1], "beta2": [-1, 0, 0, 0, 1]},
    "concrete": {"beta1": [-1, 1, -1, 1, -1, 1, -1, 1], "beta2": [-1, 0, 0, 0, 0, 0, 0, 1]},
}
XGB_GRID = {
    "n_estimators": [200, 400], "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
METHODS = ["krr_test_oracle", "pl", "krr_naive", "xgb", "iw", "tabpfn"]
TABLE_METHODS = ["krr_test_oracle", "pl", "xgb", "iw", "tabpfn"]
LABELS = {"krr_test_oracle": "KRR test-label oracle", "pl": "PL", "krr_naive": "KRR source-selected", "xgb": "Naive XGBoost", "iw": "IW XGBoost", "tabpfn": "TabPFNv2"}


@dataclass(frozen=True)
class Config:
    repeats: int = 10
    seed: int = 42
    cv_folds: int = 5
    test_size: float = 0.3
    rho: float = 0.5
    lambda_ratio: float = 2.0
    tabpfn_estimators: int = 8
    device: str = "cpu"
    threads: int = 4


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def training_weights(X, shift_beta):
    """Exponential density ratios, normalized to mean 1 for XGB regularization.

    Subtracting max(log weight) avoids overflow without clipping ratios.
    """
    log_weights = np.asarray(X) @ np.asarray(shift_beta)
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.mean()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Density ratios must be positive and finite; no clipping is applied.")
    return weights


def resample_target(X, y, shift_beta, seed):
    probabilities = training_weights(X, shift_beta)
    probabilities /= probabilities.sum()
    idx = np.random.default_rng(seed).choice(len(X), size=len(X), replace=True, p=probabilities)
    return X[idx], y[idx], idx, probabilities


def xgb_model(params, seed):
    return XGBRegressor(objective="reg:squarederror", tree_method="hist", n_jobs=1,
                        random_state=seed, eval_metric="rmse", **params)


def fit_xgboost_cv(X, y, seed, cv_folds=5, weights=None, workers=4, grid=None):
    """Select by pooled out-of-fold source MSE or density-weighted target risk.

    Explicit scoring avoids GridSearchCV's unweighted scorer silently ignoring
    sample_weight. No target labels enter training or hyperparameter selection.
    Each training fold's weights have mean one. Validation weights retain one
    common scale, so pooling is independent of fold size/weight concentration.
    """
    folds = list(KFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(X))
    candidates = list(ParameterGrid(XGB_GRID if grid is None else grid))

    def score(params):
        predictions = np.empty(len(y), dtype=float)
        for train, valid in folds:
            model = xgb_model(params, seed)
            fit_args = {} if weights is None else {"sample_weight": weights[train] / weights[train].mean()}
            model.fit(X[train], y[train], **fit_args)
            predictions[valid] = model.predict(X[valid])
        loss = float(np.average((y - predictions) ** 2, weights=weights))
        return {"params": params, "validation_mse": loss}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        scores = list(executor.map(score, candidates))
    best = min(scores, key=lambda row: row["validation_mse"])
    model = xgb_model(best["params"], seed)
    fit_args = {} if weights is None else {"sample_weight": weights / weights.mean()}
    model.fit(X, y, **fit_args)
    return model, best, scores


def krr_predictions(X_train, y_train, X_test, y_test, rho=0.5, ratio=2.0):
    """Preserve the public notebook's RBF KRR estimators and lambda grid."""
    n = len(X_train)
    n1 = int((1 - rho) * n)
    X1, X2 = X_train[:n1], X_train[n1:]
    y1, y2 = y_train[:n1], y_train[n1:]
    imputer = KernelRidge(kernel="rbf", alpha=0.1 / n).fit(X2, y2)
    pseudo_y = imputer.predict(X_test)
    m = int(np.ceil(np.log(n / 0.1) / np.log(ratio))) + 1
    lambdas = (0.1 / n) * ratio ** np.arange(m)
    predictions, source_losses = [], []
    for lbd in lambdas:
        model = KernelRidge(kernel="rbf", alpha=lbd).fit(X1, y1)
        source_losses.append(mean_squared_error(y2, model.predict(X2)))
        predictions.append(model.predict(X_test))
    predictions = np.asarray(predictions)
    choices = {
        "krr_naive": int(np.argmin(source_losses)),
        "pl": int(np.argmin(np.mean((predictions - pseudo_y) ** 2, axis=1))),
        "krr_test_oracle": int(np.argmin(np.mean((predictions - y_test) ** 2, axis=1))),
    }
    return {name: predictions[idx] for name, idx in choices.items()}, {name: float(lambdas[idx]) for name, idx in choices.items()}


def load_dataset(name):
    if name == "airfoil":
        path = HERE / "data/airfoil_self_noise.dat"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(AIRFOIL_URL, path)
        values = pd.read_csv(path, sep="\t", header=None).to_numpy(dtype=float)
    elif name == "concrete":
        path = HERE / "Concrete_Data.xls"
        values = pd.read_excel(path, engine="xlrd").to_numpy(dtype=float)
    else:
        raise ValueError(name)
    expected = {"airfoil": (1503, 6), "concrete": (1030, 9)}[name]
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f"Unexpected/missing data in {name}: {values.shape}")
    return values[:, :-1], values[:, -1], path


def resolve_model_path(model_path):
    if model_path is not None:
        path = Path(model_path).expanduser().resolve()
    else:
        # Pin to the paper's V2 checkpoint, even if a later library changes defaults.
        raise ValueError("Pass --model-path to the TabPFNv2 tabpfn-v2-regressor.ckpt checkpoint.")
    if not path.is_file() or path.name != "tabpfn-v2-regressor.ckpt":
        raise ValueError("An existing tabpfn-v2-regressor.ckpt is required (do not substitute V2.5).")
    return path


def run_experiment(output_dir, model_path, config=Config(), datasets=("airfoil", "concrete")):
    """Run all four settings; resume only when the complete protocol matches."""
    import torch
    from tabpfn import TabPFNRegressor
    torch.set_num_threads(config.threads)
    output_dir = Path(output_dir)
    model_path = resolve_model_path(model_path)
    loaded = {name: load_dataset(name) for name in datasets}
    protocol = {
        "config": asdict(config), "datasets": list(datasets),
        "shift_vectors": {name: SHIFT_VECTORS[name] for name in datasets},
        "xgb_grid": XGB_GRID, "model_sha256": sha256(model_path),
        "data_sha256": {name: sha256(data[2]) for name, data in loaded.items()},
        "runner_sha256": sha256(__file__),
        "preprocessing": "Raw features (no log transform); StandardScaler fit on source training only.",
        "evaluation": "Ordinary MSE on the shared resampled target set; no evaluation weights.",
        "iw_cv": "Pooled density-weighted out-of-fold squared error; fold training weights mean 1.",
        "oracle": "KRR lambda selected with target test labels; optimistic, not a true-mean oracle.",
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost", "tabpfn", "torch", "matplotlib", "xlrd"]},
        "python": platform.python_version(), "platform": platform.platform(),
    }
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["protocol_hash"] != protocol_hash:
            raise ValueError("Existing results use a different protocol; use a new output directory.")
    else:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True)
        manifest = {"protocol_hash": protocol_hash, "protocol": protocol, "started_utc": datetime.now(timezone.utc).isoformat(), "base_commit": commit.stdout.strip(), "complete": False}
        save_json(manifest_path, manifest)

    for name in datasets:
        X, y, _ = loaded[name]
        for repeat in range(config.repeats):
            seed = config.seed + repeat
            done = [output_dir / "runs" / f"{name}_{setting}_{repeat:02d}.json" for setting in SHIFT_VECTORS[name]]
            if all(path.exists() for path in done):
                continue
            started = time.monotonic()
            train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=config.test_size, random_state=seed)
            scaler = StandardScaler().fit(X[train_idx])
            X_train, X_test = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]
            cache = output_dir / "cache" / f"{name}_{repeat:02d}.npz"
            cache_meta = cache.with_suffix(".json")
            if cache.exists() and cache_meta.exists():
                with np.load(cache) as stored:
                    naive_prediction, tabpfn_prediction = stored["xgb"], stored["tabpfn"]
                naive_info = json.loads(cache_meta.read_text())
            else:
                print(f"{name} split {repeat + 1}/{config.repeats}: fitting naive XGB and TabPFNv2", flush=True)
                naive, best, cv = fit_xgboost_cv(X_train, y_train, seed, config.cv_folds, workers=config.threads)
                naive_prediction = naive.predict(X_test)
                tabpfn = TabPFNRegressor(model_path=str(model_path), device=config.device, n_estimators=config.tabpfn_estimators, n_jobs=1, random_state=seed)
                tabpfn.fit(X_train, y_train)
                tabpfn_prediction = tabpfn.predict(X_test)
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, xgb=naive_prediction, tabpfn=tabpfn_prediction)
                naive_info = {"best": best, "cv": cv}
                save_json(cache_meta, naive_info)
                del tabpfn, naive

            for setting_no, (setting, shift_beta) in enumerate(SHIFT_VECTORS[name].items()):
                result_path = output_dir / "runs" / f"{name}_{setting}_{repeat:02d}.json"
                if result_path.exists():
                    continue
                resample_seed = config.seed + 10000 * (1 + ["airfoil", "concrete"].index(name)) + 1000 * setting_no + repeat
                Xt, yt, idx, probabilities = resample_target(X_test, y_test, shift_beta, resample_seed)
                w_train = training_weights(X_train, shift_beta)
                iw, best, cv = fit_xgboost_cv(X_train, y_train, seed, config.cv_folds, weights=w_train, workers=config.threads)
                predictions, lambdas = krr_predictions(X_train, y_train, Xt, yt, config.rho, config.lambda_ratio)
                predictions.update(xgb=naive_prediction[idx], iw=iw.predict(Xt), tabpfn=tabpfn_prediction[idx])
                errors = {method: float(mean_squared_error(yt, pred)) for method, pred in predictions.items()}
                record = {
                    "dataset": name, "setting": setting, "repeat": repeat, "split_seed": seed,
                    "resample_seed": resample_seed, "shift_beta": shift_beta, "mse": errors,
                    "selected_lambda": lambdas, "xgb_best": naive_info["best"], "iw_best": best,
                    "n_train": len(y_train), "n_target": len(yt), "n_unique_target": len(np.unique(idx)),
                    "training_weight_ess": float(w_train.sum() ** 2 / np.sum(w_train ** 2)),
                    "training_weight_max": float(w_train.max()),
                    "target_reservoir_ess": float(1 / np.sum(probabilities ** 2)),
                    "train_indices": train_idx.tolist(), "test_pool_indices": test_idx.tolist(),
                    "resampled_pool_indices": idx.tolist(), "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                }
                # Store predictions before marking the repetition complete.
                pred_path = result_path.with_suffix(".npz")
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(pred_path, y_true=yt, source_row=test_idx[idx], **predictions)
                save_json(output_dir / "cv" / f"{name}_{setting}_{repeat:02d}.json", cv)
                save_json(result_path, record)
                print(f"  {setting}: " + ", ".join(f"{m}={errors[m]:.4f}" for m in TABLE_METHODS), flush=True)
            print(f"  split elapsed: {time.monotonic() - started:.1f}s", flush=True)

    records = read_records(output_dir)
    expected = len(datasets) * 2 * config.repeats
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} completed runs, found {len(records)}")
    manifest.update(complete=True, completed_runs=len(records), finished_utc=datetime.now(timezone.utc).isoformat())
    save_json(manifest_path, manifest)
    return summarize(output_dir, records)


def read_records(output_dir):
    return [json.loads(path.read_text()) for path in sorted((Path(output_dir) / "runs").glob("*.json"))]


def summarize(output_dir, records=None):
    output_dir = Path(output_dir)
    records = read_records(output_dir) if records is None else records
    rows = [{"dataset": r["dataset"], "setting": r["setting"], "repeat": r["repeat"], "method": m, "mse": r["mse"][m]} for r in records for m in METHODS]
    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "per_split_mse.csv", index=False)
    summary = raw.groupby(["dataset", "setting", "method"], sort=False)["mse"].agg(["mean", "std", "count"]).reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary.to_csv(output_dir / "summary.csv", index=False)
    diagnostics = pd.DataFrame([{k: r[k] for k in ["dataset", "setting", "repeat", "split_seed", "resample_seed", "n_train", "n_target", "n_unique_target", "training_weight_ess", "training_weight_max", "target_reservoir_ess"]} for r in records])
    diagnostics.to_csv(output_dir / "diagnostics.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/real_data")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.repeats < 2 or args.cv_folds < 2 or args.threads < 1:
        parser.error("At least two repetitions and CV folds, and one thread, are required.")
    config = Config(repeats=args.repeats, seed=args.seed, cv_folds=args.cv_folds, threads=args.threads, device=args.device)
    print(run_experiment(args.output_dir, args.model_path, config).to_string(index=False))


if __name__ == "__main__":
    main()
