import os
import arff
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm
from tabpfn import TabPFNRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "./data/reg")
MODEL_PATH = os.path.join(SCRIPT_DIR, "./tabpfn-v2-regressor.ckpt")

N_MAX = 1250  # subsample total before split -> ~1000 train, ~250 test

DATASETS = {
    "kin8nm":    ("kin8nm.arff",    "y"),
    "house_8L":  ("house_8L.arff",  "price"),
    "puma8NH":   ("puma8NH.arff",   "thetadd3"),
    "BNG_stock": ("BNG_stock.arff", "company10"),
    "cmc":       ("cmc.arff",       "class"),
}

# Nominal coverage levels for calibration plot (central PIs)
PI_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

# All quantile levels needed (union of PI bounds and interior grid for CRPS/PIT)
_pi_lowers = [(1 - p) / 2 for p in PI_LEVELS]
_pi_uppers = [(1 + p) / 2 for p in PI_LEVELS]
_interior = [round(k * 0.05, 2) for k in range(1, 20)]  # 0.05, 0.10, ..., 0.95
QUANTILE_LEVELS = sorted(set(_pi_lowers + _pi_uppers + _interior))

# Interior quantile indices (for CRPS)
INTERIOR_IDX = [QUANTILE_LEVELS.index(a) for a in _interior]
INTERIOR_ALPHA = [QUANTILE_LEVELS[i] for i in INTERIOR_IDX]


def load_dataset(dataset_name):
    fname, target = DATASETS[dataset_name]
    path = os.path.join(DATA_DIR, fname)
    with open(path, "r") as f:
        data = arff.load(f)
    df = pd.DataFrame(data["data"], columns=[a[0] for a in data["attributes"]])
    for col in df.select_dtypes([object]):
        if df[col].apply(lambda x: isinstance(x, bytes)).any():
            df[col] = df[col].str.decode("utf-8")
    df = df.dropna()

    X = df.drop(columns=[target])
    y = df[target]

    if y.dtype == "object":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)
    y = y.astype(float)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)
    X = X.astype(float)

    return X.values, y.values


def pit_from_quantiles(y_true, quantile_matrix, alpha_levels):
    """
    Interpolate CDF to compute PIT values u_i = F_hat_i(y_i).
    quantile_matrix: shape (K, n_test), row k = predicted quantiles at alpha_levels[k]
    """
    alpha = np.array(alpha_levels)
    n = y_true.shape[0]
    pit = np.empty(n)
    for i in range(n):
        q = quantile_matrix[:, i]
        y = y_true[i]
        if y <= q[0]:
            pit[i] = 0.0
        elif y >= q[-1]:
            pit[i] = 1.0
        else:
            k = np.searchsorted(q, y) - 1
            frac = (y - q[k]) / (q[k + 1] - q[k] + 1e-12)
            pit[i] = alpha[k] + frac * (alpha[k + 1] - alpha[k])
    return pit


def crps_from_quantiles(y_true, quantile_matrix, alpha_levels):
    """Per-sample CRPS via pinball loss approximation over a uniform quantile grid."""
    alpha = np.array(alpha_levels)
    scores = np.zeros(len(y_true))
    for k, a in enumerate(alpha):
        u = y_true - quantile_matrix[k]
        scores += np.where(u >= 0, a * u, (a - 1) * u)
    step = np.mean(np.diff(alpha))
    return 2 * step * scores


def fit_tabpfn(X_train, y_train, X_test):
    regressor = TabPFNRegressor(model_path=MODEL_PATH)
    regressor.fit(X_train, y_train)
    result = regressor.predict(X_test, output_type="full",
                               quantiles=QUANTILE_LEVELS)
    # result['quantiles'] is a list of K arrays, each of shape (n_test,)
    quantile_matrix = np.array(result["quantiles"])  # (K, n_test)
    return quantile_matrix


def fit_gpr(X_train, y_train, X_test):
    kernel_candidates = [
        ConstantKernel() * RBF(),
        ConstantKernel() * Matern(nu=1.5),
        ConstantKernel() * Matern(nu=2.5),
    ]
    alpha_candidates = [0.05, 0.1, 0.15, 0.2]

    best_score = -np.inf
    best_gpr = None
    best_alpha = 0.1

    for kernel in kernel_candidates:
        for alpha in alpha_candidates:
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha**2,
                                           n_restarts_optimizer=3)
            gpr.fit(X_train, y_train)
            score = gpr.log_marginal_likelihood()
            if score > best_score:
                best_score = score
                best_gpr = gpr
                best_alpha = alpha

    mu, cov = best_gpr.predict(X_test, return_cov=True)
    sigma = np.sqrt(np.diag(cov) + best_alpha**2)

    # Compute quantile matrix from Gaussian predictive distribution
    quantile_matrix = np.array([
        norm.ppf(a, loc=mu, scale=sigma) for a in QUANTILE_LEVELS
    ])  # (K, n_test)
    return quantile_matrix


def compute_metrics(y_test, quantile_matrix):
    metrics = {}

    # Coverage at each PI level
    coverage = []
    width_95 = None
    for pi in PI_LEVELS:
        lower_q = (1 - pi) / 2
        upper_q = (1 + pi) / 2
        lower = quantile_matrix[QUANTILE_LEVELS.index(lower_q)]
        upper = quantile_matrix[QUANTILE_LEVELS.index(upper_q)]
        cov = np.mean((lower <= y_test) & (y_test <= upper))
        coverage.append(cov)
        if pi == 0.95:
            width_95 = np.mean(upper - lower)

    metrics["coverage"] = np.array(coverage)
    metrics["width_95"] = width_95

    # PIT values
    metrics["pit"] = pit_from_quantiles(
        y_test, quantile_matrix, QUANTILE_LEVELS)

    # CRPS (using interior quantiles only)
    interior_matrix = quantile_matrix[INTERIOR_IDX]
    metrics["crps"] = np.mean(crps_from_quantiles(y_test, interior_matrix,
                                                  INTERIOR_ALPHA))
    return metrics


def run_one(dataset_name, seed, save_folder):
    rng = np.random.default_rng(seed)

    X, y = load_dataset(dataset_name)

    # Subsample for GPR feasibility
    if len(y) > N_MAX:
        idx = rng.choice(len(y), size=N_MAX, replace=False)
        X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=int(rng.integers(1e6))
    )

    # Scale features (fit on train only)
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    # Scale target (fit on train only; calibration is scale-invariant but
    # standardizing makes CRPS and width comparable across datasets)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    output = {"n_train": len(y_train), "n_test": len(y_test),
              "pi_levels": PI_LEVELS, "quantile_levels": QUANTILE_LEVELS}

    q_tabpfn = fit_tabpfn(X_train, y_train, X_test)
    for k, v in compute_metrics(y_test, q_tabpfn).items():
        output[f"TabPFN_{k}"] = v

    q_gpr = fit_gpr(X_train, y_train, X_test)
    for k, v in compute_metrics(y_test, q_gpr).items():
        output[f"GP_{k}"] = v

    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder,
                             f"{dataset_name}_seed{seed}.pickle")
    with open(save_path, "wb") as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASETS.keys()))
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    save_folder = os.path.join(SCRIPT_DIR, "output_realdata", "raw")
    run_one(args.dataset, args.seed, save_folder)
