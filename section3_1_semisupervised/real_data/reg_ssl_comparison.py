"""Semi-supervised regression comparison on OpenML datasets."""

import argparse
import hashlib
import os
import pathlib
import pickle
import urllib.request

import arff
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor
from tqdm import tqdm


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "reg"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reg"
DEFAULT_MODEL_PATH = pathlib.Path(
    os.environ.get(
        "TABPFN_REGRESSOR_CHECKPOINT",
        SCRIPT_DIR / "tabpfn-v2-regressor.ckpt",
    )
)
R_SCRIPT1 = SCRIPT_DIR / "semi_supervised_methods.R"
R_SCRIPT2 = SCRIPT_DIR / "SupervisedEstimation.R"

# Dataset: TabArena on OpenML.
# Selection criteria: at least 5,000 samples and fewer than 1,000 features.
REGRESSION_TARGETS = {
    "diamonds": "price",
    "fifa": "wage_eur",
    "Food_Delivery_Time": "Time_taken(min)",
    "miami_housing": "SALE_PRC",
    "superconductivity": "critical_temp",
    "houses": "LnMedianHouseValue",
    "wine_quality": "median_wine_quality",
    "weather_izmir": "Mean_temperature",
    "bank8FM": "rej",
    "physiochemical_protein": "RMSD",
    "nyc-taxi-green-dec-2016": "tipamount",
    "BNG_stock": "company10",
    "puma8NH": "thetadd3",
    "house_8L": "price",
    "kin8nm": "y",
}

# name: (OpenML dataset ID, OpenML file ID, MD5 checksum)
OPENML_SOURCES = {
    "kin8nm": (
        189,
        3626,
        "b36414070701ab8bc985104c3ebd9b77",
    ),
    "house_8L": (
        218,
        3655,
        "36801c359064386f44f4da0366c1837e",
    ),
    "puma8NH": (
        225,
        3662,
        "8a9f0d2deb26d1c7900f6a422361bdb0",
    ),
    "BNG_stock": (
        1200,
        150694,
        "512f4273a0dc853fd2a31b09362ad32a",
    ),
    "bank8FM": (
        572,
        1787308,
        "80ce861eb79b6fdb85a5f27facf88984",
    ),
}


def md5sum(path: pathlib.Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dataset(dataset: str, data_dir: pathlib.Path) -> pathlib.Path:
    path = data_dir / f"{dataset}.arff"
    source = OPENML_SOURCES.get(dataset)
    if not path.exists():
        if source is None:
            raise FileNotFoundError(
                f"Dataset not found: {path}. Download its ARFF file from OpenML "
                "or pass --data-dir."
            )
        _, file_id, _ = source
        data_dir.mkdir(parents=True, exist_ok=True)
        url = f"https://www.openml.org/data/v1/download/{file_id}"
        print(f"Downloading {dataset} from {url}", flush=True)
        urllib.request.urlretrieve(url, path)

    if source is not None:
        expected_md5 = source[2]
        observed_md5 = md5sum(path)
        if observed_md5 != expected_md5:
            raise ValueError(
                f"Checksum mismatch for {path}: expected {expected_md5}, "
                f"observed {observed_md5}"
            )
    return path


def load_arff_file(file_path: pathlib.Path) -> pd.DataFrame:
    """Load an ARFF file with proper encoding handling."""
    with file_path.open("r", encoding="utf-8") as file_obj:
        data = arff.load(file_obj)
    df = pd.DataFrame(
        data["data"],
        columns=[attribute[0] for attribute in data["attributes"]],
    )
    for col in df.select_dtypes([object]):
        if df[col].apply(lambda value: isinstance(value, bytes)).any():
            df[col] = df[col].str.decode("utf-8")
    return df


def preprocess_dataset(df: pd.DataFrame, target_col: str):
    """Drop missing rows, encode predictors, and standardize X and y."""
    initial_n = len(df)
    df_clean = df.dropna()
    print(f"Removed {initial_n - len(df_clean)} rows with missing values")

    if target_col not in df_clean.columns:
        raise KeyError(
            f"Target column {target_col!r} not found. "
            f"Available columns: {df_clean.columns.tolist()}"
        )

    X = df_clean.drop(columns=[target_col])
    try:
        y = pd.to_numeric(df_clean[target_col], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Target {target_col!r} is not continuous numeric data"
        ) from exc

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(
            X,
            columns=categorical_cols,
            drop_first=True,
            dtype=int,
        )
    else:
        X = X.copy()

    if numerical_cols:
        x_scaler = StandardScaler()
        X[numerical_cols] = x_scaler.fit_transform(X[numerical_cols])

    y_scaler = StandardScaler()
    y = pd.Series(
        y_scaler.fit_transform(y.to_numpy().reshape(-1, 1)).ravel(),
        index=y.index,
        name=y.name,
    )
    print(f"Final processed shape: {X.shape}")
    return X.astype(float), y


def combine_xy(X1, y1, X2=None, y2=None):
    X_parts = [X1]
    y_parts = [pd.Series(y1)]
    if X2 is not None and y2 is not None:
        X_parts.append(X2)
        y_parts.append(pd.Series(y2))
    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True).rename("target")
    return pd.concat([y, X], axis=1)


def setup_r_env(tau_0=0.5, option="i"):
    ro.r(f'setwd("{SCRIPT_DIR.as_posix()}")')
    ro.r("suppressMessages(try(library(caret), silent=TRUE))")
    ro.r(f'source("{R_SCRIPT1.as_posix()}")')
    ro.r(f'source("{R_SCRIPT2.as_posix()}")')

    if option in ["i", "W1", "S1"]:
        r_type = "linear"
    elif option in ["ii", "W2", "S2"]:
        r_type = "logistic"
    else:
        r_type = "quantile"

    ro.r(f"tau_0 <- {tau_0}")
    ro.r(f'option <- "{option}"')
    ro.r(f'type <- "{r_type}"')
    return ro


def to_r_matrix(r_session, df: pd.DataFrame, name_in_r: str):
    """Send a pandas DataFrame to R as a numeric matrix."""
    with localconverter(r_session.default_converter + pandas2ri.converter):
        r_session.globalenv[name_in_r] = (
            df.reset_index(drop=True).infer_objects().astype(float)
        )
    r_session.r(f"{name_in_r} <- as.matrix({name_in_r})")


def run_r_estimations(
    r_session,
    df_labeled,
    df_imputed,
    df_train_imputed,
    df_all_imputed,
    X_unlabeled,
):
    to_r_matrix(r_session, df_labeled, "data_labeled")
    r_session.r(
        "estimation_sup <- SupervisedEst(data_labeled, tau=tau_0, "
        "option=option); theta_sup <- estimation_sup$Est.coef"
    )
    to_r_matrix(r_session, df_imputed, "data_imputed")
    r_session.r(
        "estimation_imputed <- SupervisedEst(data_imputed, tau=tau_0, "
        "option=option); theta_imputed <- estimation_imputed$Est.coef"
    )
    to_r_matrix(r_session, df_train_imputed, "data_train_imputed")
    r_session.r(
        "estimation_train_imputed <- SupervisedEst(data_train_imputed, "
        "tau=tau_0, option=option); "
        "theta_train_imputed <- estimation_train_imputed$Est.coef"
    )
    to_r_matrix(r_session, df_all_imputed, "data_all_imputed")
    r_session.r(
        "estimation_all_imputed <- SupervisedEst(data_all_imputed, "
        "tau=tau_0, option=option); "
        "theta_all_imputed <- estimation_all_imputed$Est.coef"
    )
    to_r_matrix(r_session, X_unlabeled, "data_unlabeled")
    r_session.r(
        "estimation_SLZ <- PSSE(data_labeled, data_unlabeled, type=type, "
        "tau=tau_0); theta_SLZ <- estimation_SLZ$Hattheta"
    )

    theta_supervised = np.asarray(r_session.r("theta_sup"), dtype=float)
    theta_imputed = np.asarray(r_session.r("theta_imputed"), dtype=float)
    theta_all_imputed = np.asarray(
        r_session.r("theta_all_imputed"),
        dtype=float,
    )
    theta_train_imputed = np.asarray(
        r_session.r("theta_train_imputed"),
        dtype=float,
    )
    theta_slz = np.asarray(r_session.r("theta_SLZ"), dtype=float)
    theta_debiased = theta_all_imputed + (
        theta_supervised - theta_train_imputed
    )
    return {
        "supervised": theta_supervised,
        "imputed": theta_imputed,
        "debiased": theta_debiased,
        "SLZ": theta_slz,
    }


def save_results(results, output_path: pathlib.Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file_obj:
        pickle.dump(results, file_obj)


def comparison(
    dataset: str,
    *,
    label_size: int,
    n_reps: int,
    start_seed: int,
    data_dir: pathlib.Path,
    output_dir: pathlib.Path,
    model_path: pathlib.Path,
    device: str,
    prediction_batch_size: int,
    resume: bool,
):
    if not model_path.exists():
        raise FileNotFoundError(
            f"TabPFN checkpoint not found: {model_path}. Pass --model-path."
        )

    data_path = ensure_dataset(dataset, data_dir)
    X, y = preprocess_dataset(
        load_arff_file(data_path),
        REGRESSION_TARGETS[dataset],
    )

    r_session = setup_r_env(tau_0=0.5, option="i")
    df_full = combine_xy(X, y)
    to_r_matrix(r_session, df_full, "data_labelled_full")
    r_session.r(
        "estimation_oracle <- SupervisedEst(data_labelled_full, "
        "tau=tau_0, option=option); "
        "theta_oracle <- estimation_oracle$Est.coef"
    )
    theta_oracle = np.asarray(r_session.r("theta_oracle"), dtype=float)

    source = OPENML_SOURCES.get(dataset)
    metadata = {
        "dataset": dataset,
        "target": REGRESSION_TARGETS[dataset],
        "openml_dataset_id": source[0] if source else None,
        "openml_file_id": source[1] if source else None,
        "openml_md5": source[2] if source else None,
        "label_percent": label_size,
        "n_reps": n_reps,
        "start_seed": start_seed,
        "n_samples": len(X),
        "n_features": X.shape[1],
    }
    output_path = output_dir / f"n_{label_size}" / f"{dataset}.pickle"
    if resume and output_path.exists():
        with output_path.open("rb") as file_obj:
            results = pickle.load(file_obj)
        previous_dataset = results.get("metadata", {}).get("dataset")
        if previous_dataset not in (None, dataset):
            raise ValueError(
                f"Cannot resume {dataset!r} from results for {previous_dataset!r}"
            )
        results["truth"] = theta_oracle
        results["metadata"] = metadata
    else:
        results = {"truth": theta_oracle, "metadata": metadata}

    n_samples = len(X)
    labeled_sample_size = min(5000, int(label_size * n_samples / 100))
    for seed in tqdm(range(start_seed, start_seed + n_reps)):
        result_keys = [
            f"{method}_{seed}"
            for method in ("vanilla", "TabPFNI", "TabPFND", "SLZ")
        ]
        if resume and all(key in results for key in result_keys):
            print(f"seed={seed} already complete; skipping", flush=True)
            continue
        rng = np.random.default_rng(seed)
        labeled_idx = rng.choice(
            n_samples,
            labeled_sample_size,
            replace=False,
        )
        unlabeled_mask = np.ones(n_samples, dtype=bool)
        unlabeled_mask[labeled_idx] = False
        unlabeled_idx = np.flatnonzero(unlabeled_mask)

        X_train = X.iloc[labeled_idx]
        y_train = y.iloc[labeled_idx]
        X_test = X.iloc[unlabeled_idx]

        regressor = TabPFNRegressor(
            model_path=model_path,
            random_state=seed,
            device=device,
        )
        regressor.fit(X_train, y_train)
        X_predict = pd.concat([X_train, X_test], ignore_index=True)
        predictions = np.concatenate([
            regressor.predict(X_predict.iloc[start:start + prediction_batch_size])
            for start in range(0, len(X_predict), prediction_batch_size)
        ])
        preds_train = predictions[:len(X_train)]
        preds_test = predictions[len(X_train):]

        df_labeled = combine_xy(X_train, y_train)
        df_imputed = combine_xy(X_train, y_train, X_test, preds_test)
        df_train_imputed = combine_xy(X_train, preds_train)
        df_all_imputed = combine_xy(
            X_train,
            preds_train,
            X_test,
            preds_test,
        )
        thetas = run_r_estimations(
            r_session,
            df_labeled=df_labeled,
            df_imputed=df_imputed,
            df_train_imputed=df_train_imputed,
            df_all_imputed=df_all_imputed,
            X_unlabeled=X_test,
        )

        results[f"vanilla_{seed}"] = (
            thetas["supervised"] - theta_oracle
        ) ** 2
        results[f"TabPFNI_{seed}"] = (
            thetas["imputed"] - theta_oracle
        ) ** 2
        results[f"TabPFND_{seed}"] = (
            thetas["debiased"] - theta_oracle
        ) ** 2
        results[f"SLZ_{seed}"] = (thetas["SLZ"] - theta_oracle) ** 2
        save_results(results, output_path)

        means = [
            np.mean(results[f"{method}_{seed}"])
            for method in ("vanilla", "TabPFNI", "TabPFND", "SLZ")
        ]
        print(
            f"seed={seed} mean squared coefficient errors: {means}",
            flush=True,
        )

    print(f"Saved {output_path}", flush=True)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="bank8FM",
        choices=REGRESSION_TARGETS,
    )
    parser.add_argument("--label-size", type=int, default=10)
    parser.add_argument("--n-reps", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--model-path",
        type=pathlib.Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prediction-batch-size", type=int, default=400)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    comparison(
        args.dataset,
        label_size=args.label_size,
        n_reps=args.n_reps,
        start_seed=args.start_seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        device=args.device,
        prediction_batch_size=args.prediction_batch_size,
        resume=args.resume,
    )
