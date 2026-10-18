"""Real-data predictive-distribution calibration for TabPFN and GPR.

The five benchmark datasets are regression datasets from OpenML. Files are
downloaded on demand and checked against the checksum reported by OpenML.
"""

import argparse
import hashlib
import os
import pickle
import urllib.request
from pathlib import Path

import arff
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "reg"
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "TABPFN_REGRESSOR_CHECKPOINT",
        SCRIPT_DIR / "tabpfn-v2-regressor.ckpt",
    )
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_realdata" / "raw"

N_MAX = 1250  # subsample total before split -> 1000 train, 250 test

# name: (filename, target, OpenML dataset ID, OpenML file ID, MD5 checksum)
DATASETS = {
    "kin8nm": (
        "kin8nm.arff", "y", 189, 3626, "b36414070701ab8bc985104c3ebd9b77"
    ),
    "house_8L": (
        "house_8L.arff", "price", 218, 3655,
        "36801c359064386f44f4da0366c1837e",
    ),
    "puma8NH": (
        "puma8NH.arff", "thetadd3", 225, 3662,
        "8a9f0d2deb26d1c7900f6a422361bdb0",
    ),
    "BNG_stock": (
        "BNG_stock.arff", "company10", 1200, 150694,
        "512f4273a0dc853fd2a31b09362ad32a",
    ),
    "bank8FM": (
        "bank8FM.arff", "rej", 572, 1787308,
        "80ce861eb79b6fdb85a5f27facf88984",
    ),
}

# Nominal coverage levels for central prediction intervals.
PI_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

# Union of PI endpoints and the interior grid used for CRPS/PIT.
_pi_lowers = [(1 - p) / 2 for p in PI_LEVELS]
_pi_uppers = [(1 + p) / 2 for p in PI_LEVELS]
_interior = [round(k * 0.05, 2) for k in range(1, 20)]
QUANTILE_LEVELS = sorted(set(_pi_lowers + _pi_uppers + _interior))
INTERIOR_IDX = [QUANTILE_LEVELS.index(alpha) for alpha in _interior]
INTERIOR_ALPHA = [QUANTILE_LEVELS[i] for i in INTERIOR_IDX]


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dataset(dataset_name: str, data_dir: Path) -> Path:
    """Return a verified local ARFF path, downloading it when absent."""
    filename, _, _, file_id, expected_md5 = DATASETS[dataset_name]
    path = data_dir / filename
    data_dir.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        url = f"https://www.openml.org/data/v1/download/{file_id}"
        print(f"Downloading {dataset_name} from {url}", flush=True)
        urllib.request.urlretrieve(url, path)

    observed_md5 = md5sum(path)
    if observed_md5 != expected_md5:
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected_md5}, "
            f"observed {observed_md5}"
        )
    return path


def load_dataset(dataset_name: str, data_dir: Path):
    _, target, _, _, _ = DATASETS[dataset_name]
    path = ensure_dataset(dataset_name, data_dir)
    with path.open("r", encoding="utf-8") as file_obj:
        data = arff.load(file_obj)

    df = pd.DataFrame(data["data"], columns=[a[0] for a in data["attributes"]])
    for col in df.select_dtypes([object]):
        if df[col].apply(lambda value: isinstance(value, bytes)).any():
            df[col] = df[col].str.decode("utf-8")
    df = df.dropna()

    X = df.drop(columns=[target])
    try:
        y = pd.to_numeric(df[target], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Target {target!r} in {dataset_name!r} is not continuous numeric data"
        ) from exc

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)
    X = X.astype(float)
    return X.values, y.values


def pit_from_quantiles(y_true, quantile_matrix, alpha_levels):
    """Compute PIT by linear CDF interpolation, clipped outside the grid.

    The predictive quantile grid spans 0.025--0.975. Observations below the
    first predicted quantile receive PIT 0 and observations above the last
    predicted quantile receive PIT 1.
    """
    alpha = np.asarray(alpha_levels)
    pit = np.empty(y_true.shape[0])
    for i, y_value in enumerate(y_true):
        quantiles = np.maximum.accumulate(quantile_matrix[:, i])
        if y_value <= quantiles[0]:
            pit[i] = 0.0
        elif y_value >= quantiles[-1]:
            pit[i] = 1.0
        else:
            upper_idx = np.searchsorted(quantiles, y_value, side="right")
            lower_idx = upper_idx - 1
            fraction = (
                (y_value - quantiles[lower_idx])
                / (quantiles[upper_idx] - quantiles[lower_idx] + 1e-12)
            )
            pit[i] = alpha[lower_idx] + fraction * (
                alpha[upper_idx] - alpha[lower_idx]
            )
    return pit


def crps_from_quantiles(y_true, quantile_matrix, alpha_levels):
    """Approximate per-sample CRPS by integrating quantile (pinball) loss."""
    alpha = np.asarray(alpha_levels)
    scores = np.zeros(len(y_true))
    for k, level in enumerate(alpha):
        residual = y_true - quantile_matrix[k]
        scores += np.where(
            residual >= 0,
            level * residual,
            (level - 1) * residual,
        )
    step = np.mean(np.diff(alpha))
    return 2 * step * scores


def fit_tabpfn(
    X_train,
    y_train,
    X_test,
    *,
    model_path: Path,
    seed: int,
    device: str,
):
    if not model_path.exists():
        raise FileNotFoundError(
            f"TabPFN checkpoint not found: {model_path}. Pass --model-path."
        )
    regressor = TabPFNRegressor(
        model_path=model_path,
        random_state=seed,
        device=device,
    )
    regressor.fit(X_train, y_train)
    result = regressor.predict(
        X_test,
        output_type="full",
        quantiles=QUANTILE_LEVELS,
    )
    return np.asarray(result["quantiles"])


def fit_gpr(
    X_train,
    y_train,
    X_test,
    *,
    seed: int,
    n_restarts: int,
    n_jobs: int,
):
    kernel_candidates = [
        ConstantKernel() * RBF(),
        ConstantKernel() * Matern(nu=1.5),
        ConstantKernel() * Matern(nu=2.5),
    ]
    alpha_candidates = [0.05, 0.1, 0.15, 0.2]

    def fit_candidate(kernel, alpha):
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha**2,
            n_restarts_optimizer=n_restarts,
            random_state=seed,
        )
        gpr.fit(X_train, y_train)
        return gpr.log_marginal_likelihood(), gpr, alpha

    fitted = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(fit_candidate)(kernel, alpha)
        for kernel in kernel_candidates
        for alpha in alpha_candidates
    )
    _, best_gpr, best_alpha = max(fitted, key=lambda item: item[0])

    mu, covariance = best_gpr.predict(X_test, return_cov=True)
    sigma = np.sqrt(np.diag(covariance) + best_alpha**2)
    return np.asarray([
        norm.ppf(level, loc=mu, scale=sigma) for level in QUANTILE_LEVELS
    ])


def compute_metrics(y_test, quantile_matrix):
    metrics = {}
    coverage = []
    width_95 = None
    for pi_level in PI_LEVELS:
        lower_q = (1 - pi_level) / 2
        upper_q = (1 + pi_level) / 2
        lower = quantile_matrix[QUANTILE_LEVELS.index(lower_q)]
        upper = quantile_matrix[QUANTILE_LEVELS.index(upper_q)]
        coverage.append(np.mean((lower <= y_test) & (y_test <= upper)))
        if pi_level == 0.95:
            width_95 = np.mean(upper - lower)

    metrics["coverage"] = np.asarray(coverage)
    metrics["width_95"] = width_95
    metrics["pit"] = pit_from_quantiles(
        y_test,
        quantile_matrix,
        QUANTILE_LEVELS,
    )
    metrics["crps"] = np.mean(
        crps_from_quantiles(
            y_test,
            quantile_matrix[INTERIOR_IDX],
            INTERIOR_ALPHA,
        )
    )
    return metrics


def run_one(
    dataset_name: str,
    seed: int,
    save_folder: Path,
    *,
    data_dir: Path,
    model_path: Path,
    device: str,
    n_max: int,
    gpr_restarts: int,
    gpr_jobs: int,
):
    rng = np.random.default_rng(seed)
    X, y = load_dataset(dataset_name, data_dir)

    if len(y) > n_max:
        sample_idx = rng.choice(len(y), size=n_max, replace=False)
        X, y = X[sample_idx], y[sample_idx]

    split_seed = int(rng.integers(1_000_000))
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=split_seed,
    )

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    _, target, openml_id, openml_file_id, checksum = DATASETS[dataset_name]
    output = {
        "dataset": dataset_name,
        "openml_dataset_id": openml_id,
        "openml_file_id": openml_file_id,
        "openml_md5": checksum,
        "target": target,
        "seed": seed,
        "split_seed": split_seed,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "pi_levels": PI_LEVELS,
        "quantile_levels": QUANTILE_LEVELS,
        "pit_definition": (
            "linear interpolation over predicted quantiles on the "
            "0.025--0.975 grid; clipped to 0/1 outside that range"
        ),
    }

    q_tabpfn = fit_tabpfn(
        X_train,
        y_train,
        X_test,
        model_path=model_path,
        seed=seed,
        device=device,
    )
    for key, value in compute_metrics(y_test, q_tabpfn).items():
        output[f"TabPFN_{key}"] = value

    q_gpr = fit_gpr(
        X_train,
        y_train,
        X_test,
        seed=seed,
        n_restarts=gpr_restarts,
        n_jobs=gpr_jobs,
    )
    for key, value in compute_metrics(y_test, q_gpr).items():
        output[f"GP_{key}"] = value

    save_folder.mkdir(parents=True, exist_ok=True)
    save_path = save_folder / f"{dataset_name}_seed{seed}.pickle"
    with save_path.open("wb") as file_obj:
        pickle.dump(output, file_obj)
    print(f"Saved {save_path}", flush=True)
    return save_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-max", type=int, default=N_MAX)
    parser.add_argument("--gpr-restarts", type=int, default=3)
    parser.add_argument("--gpr-jobs", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_one(
        args.dataset,
        args.seed,
        args.output_dir,
        data_dir=args.data_dir,
        model_path=args.model_path,
        device=args.device,
        n_max=args.n_max,
        gpr_restarts=args.gpr_restarts,
        gpr_jobs=args.gpr_jobs,
    )
