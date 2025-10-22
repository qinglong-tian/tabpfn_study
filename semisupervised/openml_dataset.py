# ======== CONFIG ========
DATASET_ID = 871        # <— change OpenML dataset here
TEST_SIZE  = 0.3
N_REPEATS  = 5
DEVICE     = "cuda"        # or "cuda"

# R settings
R_HOME     = r"C:/PROGRA~1/R/R-44~1.0"
R_SCRIPT1  = "semi_supervised_methods.R"
R_SCRIPT2  = "SupervisedEstimation.R"
TAU_0      = 0.5
OPTION     = "ii"         # i/W1/S1: linear; ii/W2/S2: logistic; iii/W3/S3: quantile
# ========================

import os
import pathlib
import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from tabpfn import TabPFNRegressor, TabPFNClassifier

# -------------------------------
# 1) Data utilities
# -------------------------------
def load_openml(dataset_id: int):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    n_classes = dataset.qualities.get("NumberOfClasses")
    if n_classes == 0.0:
        task = "regression"
    else:
        task = "classification"

    print(f"Dataset: {dataset.name} | Target: {dataset.default_target_attribute} | Task: {task} | X={X.shape}, y={y.shape}")
    return X, y, task

def encode_y(y):
    """Encode categorical target into 0..K-1 integers (returns pd.Series)."""
    if pd.api.types.is_numeric_dtype(y):
        return pd.Series(y, name="target")
    le = LabelEncoder()
    return pd.Series(le.fit_transform(y), name="target")

def combine_xy(X1, Y1, X2=None, Y2=None):
    """Return DataFrame with 'target' first, then features."""
    Xs = [X1]
    ys = [pd.Series(Y1)]
    if X2 is not None and Y2 is not None:
        Xs.append(X2); ys.append(pd.Series(Y2))
    X = pd.concat(Xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True).rename("target")
    return pd.concat([y, X], axis=1)

# -------------------------------
# 2) Modeling (TabPFN)
# -------------------------------
def fit_tabpfn(X_train, X_test, y_train, y_test, task, device=DEVICE):
    X_train_np = X_train.to_numpy().astype("float32")
    X_test_np  = X_test.to_numpy().astype("float32")
    if task == "regression":
        model = TabPFNRegressor(device=device)
        model.fit(X_train_np, y_train.to_numpy().astype("float32"))
        preds = model.predict(X_test_np)
        _ = r2_score(y_test, preds)
    else:
        model = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
        model.fit(X_train_np, y_train.to_numpy())
        preds = model.predict(X_test_np)
        _ = accuracy_score(y_test, preds)
    return model, preds

# -------------------------------
# 3) R bridge
# -------------------------------
def setup_r_env(R_HOME, R_SCRIPT1, R_SCRIPT2, tau_0, option):
    os.environ["R_HOME"] = R_HOME
    os.environ["R_USER"] = os.environ["R_HOME"]

    import rpy2.robjects as ro
    wd = pathlib.Path.cwd().as_posix()
    ro.r(f'setwd("{wd}")')

    ro.r('suppressMessages(library(MASS))')
    ro.r('suppressMessages(try(library(caret), silent=TRUE))')
    ro.r('suppressMessages(library(quantreg))')
    ro.r(f'source("{R_SCRIPT1}")')
    ro.r(f'source("{R_SCRIPT2}")')

    if option in ["i", "W1", "S1"]:
        r_type = "linear"
    elif option in ["ii", "W2", "S2"]:
        r_type = "logistic"
    else:
        r_type = "quantile"

    ro.r(f'tau_0 <- {tau_0}')
    ro.r(f'option <- "{option}"')
    ro.r(f'type <- "{r_type}"')
    return ro

def to_r_matrix(ro, df: pd.DataFrame, name_in_r: str):
    """Send a pandas DataFrame to R as a numeric matrix."""
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv[name_in_r] = df.reset_index(drop=True).infer_objects().astype(float)
    ro.r(f'{name_in_r} <- as.matrix({name_in_r})')

def run_r_estimations(ro, df_labeled, df_full, df_imputed, df_train_imputed, df_all_imputed, X_unlabeled):
    # Push to R and run estimators
    to_r_matrix(ro, df_labeled, "data_labeled")
    ro.r('estimation_sup <- SupervisedEst(data_labeled, tau=tau_0, option=option); theta_sup <- estimation_sup$Est.coef')

    to_r_matrix(ro, df_full, "data_labelled_full")
    ro.r('estimation_oracle <- SupervisedEst(data_labelled_full, tau=tau_0, option=option); theta_oracle <- estimation_oracle$Est.coef')

    to_r_matrix(ro, df_imputed, "data_imputed")
    ro.r('estimation_imputed <- SupervisedEst(data_imputed, tau=tau_0, option=option); theta_imputed <- estimation_imputed$Est.coef')

    to_r_matrix(ro, df_train_imputed, "data_train_imputed")
    ro.r('estimation_train_imputed <- SupervisedEst(data_train_imputed, tau=tau_0, option=option); theta_train_imputed <- estimation_train_imputed$Est.coef')

    to_r_matrix(ro, df_all_imputed, "data_all_imputed")
    ro.r('estimation_all_imputed <- SupervisedEst(data_all_imputed, tau=tau_0, option=option); theta_all_imputed <- estimation_all_imputed$Est.coef')

    to_r_matrix(ro, X_unlabeled, "data_unlabeled")
    ro.r('estimation_SLZ <- PSSE(data_labeled, data_unlabeled, type=type, tau=tau_0); theta_SLZ <- estimation_SLZ$Hattheta')

    # Fetch back to Python
    theta_supervised     = np.asarray(ro.r('theta_sup'), dtype=float)
    theta_oracle         = np.asarray(ro.r('theta_oracle'), dtype=float)
    theta_imputed        = np.asarray(ro.r('theta_imputed'), dtype=float)
    theta_all_imputed    = np.asarray(ro.r('theta_all_imputed'), dtype=float)
    theta_train_imputed  = np.asarray(ro.r('theta_train_imputed'), dtype=float)
    theta_SLZ            = np.asarray(ro.r('theta_SLZ'), dtype=float)

    # Debias
    theta_bias     = theta_supervised - theta_train_imputed
    theta_debiased = theta_all_imputed + theta_bias

    return {
        "oracle":     theta_oracle,
        "supervised": theta_supervised,
        "imputed":    theta_imputed,
        "debiased":   theta_debiased,
        "SLZ":        theta_SLZ,
    }

# -------------------------------
# 4) Evaluation
# -------------------------------
def evaluate_vs_oracle(thetas: dict):
    theta_oracle = thetas["oracle"]
    print(thetas)
    rows = []
    for name in ["supervised", "imputed", "debiased", "SLZ"]:
        diff = thetas[name] - theta_oracle
        diff = diff[1:]
        l1   = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        rows.append({"method": name, "L1": l1, "RMSE": rmse})
    return pd.DataFrame(rows)

# -------------------------------
# 5) One full run (one split)
# -------------------------------
def run_once(ro, X, y, task, seed):
    # one-hot features (simple & consistent)
    X_enc = pd.get_dummies(X, drop_first=True)

    # classification: ensure numeric labels (0..K-1)
    y_enc = encode_y(y) if task == "classification" else pd.Series(y, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc, test_size=TEST_SIZE, random_state=seed
    )

    # TabPFN predictions
    _, y_pred_test  = fit_tabpfn(X_train, X_test,  y_train, y_test,  task, device=DEVICE)
    _, y_pred_train = fit_tabpfn(X_train, X_train, y_train, y_train, task, device=DEVICE)

    print(accuracy_score(y_test, y_pred_test))

    # Build datasets for R side
    df_labeled       = combine_xy(X_train, y_train)                            # train only
    df_full          = combine_xy(X_train, y_train, X_test, y_test)            # oracle
    df_imputed       = combine_xy(X_train, y_train, X_test, y_pred_test)       # test imputed
    df_train_imputed = combine_xy(X_train, y_pred_train)                        # train imputed
    df_all_imputed   = combine_xy(X_train, y_pred_train, X_test, y_pred_test)  # all imputed

    thetas = run_r_estimations(
        ro,
        df_labeled=df_labeled,
        df_full=df_full,
        df_imputed=df_imputed,
        df_train_imputed=df_train_imputed,
        df_all_imputed=df_all_imputed,
        X_unlabeled=X_test
    )
    return evaluate_vs_oracle(thetas)

# -------------------------------
# 6) Main: repeat K times and average
# -------------------------------
def main():
    # data
    X, y, task = load_openml(DATASET_ID)

    # R session (one setup, reused across repeats)
    ro = setup_r_env(R_HOME, R_SCRIPT1, R_SCRIPT2, TAU_0, OPTION)

    # run repeats
    dfs = []
    for seed in range(N_REPEATS):
        print(f"\n=== Repeat {seed+1}/{N_REPEATS} (seed={seed}) ===")
        df_metrics = run_once(ro, X, y, task, 100*seed)
        dfs.append(df_metrics)

    all_metrics = pd.concat(dfs).groupby("method", as_index=True).agg(["mean", "std"])
    print("\n===== Average over repeats =====")
    print(all_metrics.apply(lambda s: s.map(lambda v: round(v, 6))))

if __name__ == "__main__":
    main()
