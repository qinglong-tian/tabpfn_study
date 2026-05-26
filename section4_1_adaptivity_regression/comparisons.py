import os
import pickle
import argparse
import pycasso
import numpy as np
from sklearn.model_selection import KFold


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


# CV evaluation function
def cv_score_model(model_func, X_data, y_data, k=5, seed=1):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    cv_scores = []

    for train_idx, val_idx in kf.split(X_data):
        X_train_fold, X_val_fold = X_data[train_idx], X_data[val_idx]
        y_train_fold, y_val_fold = y_data[train_idx], y_data[val_idx]

        model = model_func(X_train_fold, y_train_fold)
        y_val_pred = model.predict(X_val_fold)
        mse = np.mean((y_val_pred - y_val_fold)**2)
        cv_scores.append(mse)

    return np.mean(cv_scores)


def cv_score_pycasso_by_ratio(X_data,
                              y_data,
                              penalty="l1",
                              gamma=None,
                              k=5,
                              n_lambdas=50,
                              seed=1):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    # set lambda_min_ratio
    lambda_min_ratio = 0.05

    # save the CV score for each fold
    all_cv_scores = []

    for train_idx, val_idx in kf.split(X_data):
        X_train_fold, X_val_fold = X_data[train_idx], X_data[val_idx]
        y_train_fold, y_val_fold = y_data[train_idx], y_data[val_idx]

        # Train model for each lambda
        if gamma is not None:
            model = pycasso.Solver(X_train_fold,
                                   y_train_fold,
                                   useintercept=False,
                                   lambdas=(n_lambdas, lambda_min_ratio),
                                   family="gaussian",
                                   gamma=gamma,
                                   penalty=penalty)
        else:
            model = pycasso.Solver(X_train_fold,
                                   y_train_fold,
                                   useintercept=False,
                                   lambdas=(n_lambdas, lambda_min_ratio),
                                   family="gaussian",
                                   penalty=penalty)
        model.train()

        fold_scores = []
        for i in range(n_lambdas):
            y_val_pred = model.predict(X_val_fold, lambdidx=i)
            mse = np.mean((y_val_pred - y_val_fold)**2)
            fold_scores.append(mse)

        all_cv_scores.append(fold_scores)

    # return CV scores across fold
    return np.mean(all_cv_scores, axis=0)


def data_gen(n, d, s, rho, nu=0.1, beta_type=1, X_var="identity", rng=None):
    if X_var == "identity":
        X = rng.normal(size=(n, d))
        if beta_type == 1:
            beta = create_sparse_vector(d, s)
        elif beta_type == 2:
            beta = np.concatenate([np.ones(s), np.zeros(d - s)])
        error_var = np.sum(beta**2) / nu

    elif X_var == "band":
        var = generate_var(rho, d)
        mean = np.zeros(d)
        X = rng.multivariate_normal(mean, var, n)
        if beta_type == 1:
            beta = create_sparse_vector(d, s)
        elif beta_type == 2:
            beta = np.concatenate([np.ones(s), np.zeros(d - s)])
        error_var = np.sum(var.dot(beta) * beta) / nu
    f = X.dot(beta)
    y = f + np.sqrt(error_var) * rng.normal(size=n)

    return X, f, y


def main(n, d, s, rho, seed, beta_type, X_var, save_folder):
    # load dataset
    save_file = os.path.join(
        "./output/save_data_revision",
        "ss_" + str(n),
        "d_" + str(d),
        "s_" + str(s),
        "case_" + str(seed) + "_beta_1_" + X_var + ".pickle",
    )
    with open(save_file, 'rb') as f:
        results = pickle.load(f)
    output = {}

    for nu in [1.22, 6]:
        rng = np.random.default_rng(seed)
        X, f, y = data_gen(n + 1000,
                           d,
                           s,
                           rho,
                           nu=nu,
                           beta_type=beta_type,
                           X_var=X_var,
                           rng=rng)
        # randomly shuffle data
        idx = rng.choice(n + 1000, n + 1000, replace=False)
        X = X[idx]
        f = f[idx]
        y = y[idx]

        # split dataset
        X_train = X[:n]
        y_train = results["nu_" + str(nu) + "_TabPFN_predict"][:n]
        X_test = X[n:]
        y_test = y[n:]
        f_test = f[n:]

        # LASSO
        # LASSO with 5-fold CV
        lasso_cv_scores = cv_score_pycasso_by_ratio(X_train,
                                                    y_train,
                                                    penalty="mcp",
                                                    seed=seed)

        best_lasso_idx = np.argmin(lasso_cv_scores)

        lasso_model = pycasso.Solver(X_train,
                                     y_train,
                                     useintercept=False,
                                     lambdas=(50, 0.05),
                                     family="gaussian",
                                     penalty="mcp")
        lasso_model.train()
        # y_pred = lasso_model.predict(X_test, lambdidx=best_lasso_idx)
        output["nu_" + str(nu) +
               "_coef"] = lasso_model.coef()['beta'][best_lasso_idx]

    save_file = os.path.join(
        save_folder,
        "case_" + str(seed) + "_beta_" + str(beta_type) + "_" + X_var +
        ".pickle",
    )

    f = open(save_file, "wb")
    pickle.dump(output, f)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CATE estimation")
    parser.add_argument("--seed",
                        type=int,
                        default=1,
                        help="index of repetition")
    parser.add_argument("--n", type=int, default=500, help="Training set size")
    parser.add_argument("--d",
                        type=int,
                        default=100,
                        help="dimension of covariates")
    parser.add_argument("--s", type=int, default=1, help="sparsity")

    args = parser.parse_args()
    n = int(args.n)
    seed = args.seed
    d = args.d
    s = args.s
    rho = 0.35
    save_folder = os.path.join("./output/bias_variance_mcp", "ss_" + str(n),
                               "d_" + str(d), "s_" + str(s))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    main(n, d, s, rho, seed, 1, "identity", save_folder)
    # main(n, d, s, rho, seed, 1, "band", save_folder)
    # main(n, d, s, rho, seed, 2, "band", save_folder)
