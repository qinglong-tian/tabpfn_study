import os
import pickle
import argparse
import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV


def generate_var(rho, d):
    idx = np.arange(1, d + 1)
    return np.power(rho, abs(np.subtract.outer(idx, idx)))


def create_sparse_vector(p, s):
    # Create array of zeros
    vec = np.zeros(p)

    # Calculate indices for 1s
    indices = np.linspace(0, p - 1, s, dtype=int)

    # Set those indices to 1
    vec[indices] = 1

    return vec


def data_gen(n, d, s, rho, nu=0.1, beta_type=1, X_var="identity", seed=0):
    np.random.seed(seed)
    if X_var == "identity":
        X = np.random.randn(n, d)
        beta = np.concatenate([np.ones(s), np.zeros(d - s)])
        error_var = np.sum(beta**2) / nu

    elif X_var == "band":
        var = generate_var(rho, d)
        mean = np.zeros(d)
        X = np.random.multivariate_normal(mean, var, n)
        if beta_type == 1:
            beta = create_sparse_vector(d, s)
        elif beta_type == 2:
            beta = np.concatenate([np.ones(s), np.zeros(d - s)])
        error_var = np.sum(var.dot(beta) * beta) / nu
    f = X.dot(beta)
    y = f + np.sqrt(error_var) * np.random.randn(n)

    return X, f, y


def main(n, d, s, rho, seed, beta_type, X_var, save_folder):
    if beta_type == 1:
        selected_coordinates = np.arange(s)
    elif beta_type == 2:
        selected_coordinates = (create_sparse_vector(d, s) == 1)

    output = {}

    for nu in [0.05, 0.25, 1.22, 6]:
        # various SNR ratios
        X, f, y = data_gen(n + 1000,
                           d,
                           s,
                           rho,
                           nu=nu,
                           beta_type=beta_type,
                           X_var=X_var,
                           seed=seed)
        np.random.seed(seed)
        # randomly shuffle data
        idx = np.random.choice(n + 1000, n + 1000, replace=False)
        X = X[idx]
        f = f[idx]
        y = y[idx]

        # split dataset
        X_train = X[:n]
        y_train = y[:n]
        X_test = X[n:]
        y_test = f[n:]

        # TabPFN
        regressor = TabPFNRegressor()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        output["nu_" + str(nu) + "_TabPFN_MSE"] = np.mean((y_pred - y_test)**2)
        model = LinearRegression()
        model.fit(X_test[:, selected_coordinates], y_pred)
        r2_sklearn = model.score(X_test[:, selected_coordinates], y_pred)
        output["nu_" + str(nu) + "_TabPFN_coef"] = (model.intercept_,
                                                    model.coef_)
        output["nu_" + str(nu) + "_TabPFN_R2"] = r2_sklearn

        # LASSO
        lasso_cv = LassoCV(alphas=None,
                           cv=5,
                           n_jobs=-1,
                           max_iter=10000,
                           random_state=seed)
        lasso_cv.fit(X_train, y_train)
        y_pred = lasso_cv.predict(X_test)
        output["nu_" + str(nu) + "_LASSO_MSE"] = np.mean((y_pred - y_test)**2)
        output["nu_" + str(nu) + "_LASSO_coef"] = (lasso_cv.intercept_,
                                                   lasso_cv.coef_)
        r2_sklearn = lasso_cv.score(X_test, y_pred)
        output["nu_" + str(nu) + "_LASSO_R2"] = r2_sklearn

        model = LinearRegression()
        model.fit(X_test[:, selected_coordinates], y_pred)
        r2_sklearn = model.score(X_test[:, selected_coordinates], y_pred)
        output["nu_" + str(nu) + "_LASSOReg_coef"] = (model.intercept_,
                                                      model.coef_)
        output["nu_" + str(nu) + "_LASSOReg_R2"] = r2_sklearn

        # Oracle
        model = LinearRegression()
        model.fit(X_train[:, selected_coordinates], y_train)
        y_pred = model.predict(X_test[:, selected_coordinates])
        output["nu_" + str(nu) + "_Oracle_MSE"] = np.mean((y_pred - y_test)**2)
        output["nu_" + str(nu) + "_Oracle_coef"] = (model.intercept_,
                                                    model.coef_)

    save_file = os.path.join(
        save_folder,
        "case_" + str(seed) + "_beta_" + str(beta_type) + "_" + X_var +
        ".pickle",
    )

    f = open(save_file, "wb")
    pickle.dump(output, f)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear regression")
    parser.add_argument("--seed",
                        type=int,
                        default=1,
                        help="index of repetition")
    parser.add_argument("--n", type=int, default=500, help="Training set size")
    parser.add_argument("--d",
                        type=int,
                        default=6,
                        help="dimension of covariates")
    parser.add_argument("--s", type=int, default=5, help="sparsity")

    args = parser.parse_args()
    n = int(args.n)
    seed = args.seed
    d = args.d
    s = args.s
    rho = 0.35
    save_folder = os.path.join("./save_data", "ss_" + str(n), "d_" + str(d),
                               "s_" + str(s))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    # orthogonal design
    main(n, d, s, rho, seed, 1, "identity", save_folder)
    # band design
    main(n, d, s, rho, seed, 1, "band", save_folder)
    main(n, d, s, rho, seed, 2, "band", save_folder)
