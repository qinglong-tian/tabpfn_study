import os
import pickle
import argparse
import numpy as np
from tabpfn import TabPFNRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, ConstantKernel)


def linear(x):
    return x


def quadratic(x):
    return x**2


def step(x):
    return np.array([0 if t < 0 else 1 for t in x])


def piecewise_linear(x):
    """
    Create piecewise linear function connecting specified points
    
    Parameters:
    x : array-like - Input values
    breakpoints : list - x-coordinates of breakpoints (must include 0 and 1)
    values : list - y-values at breakpoints
    """
    breakpoints = [-1, -0.6, 0.2, 0.6, 1]
    values = [0, 1, -0.5, 0.7, 0]
    breakpoints = np.array(breakpoints)
    values = np.array(values)
    return np.interp(x, breakpoints, values)


# for rep in range(10):
#     for n in [11, 31]:
#         for i, f in enumerate([linear, quadratic, step, piecewise_linear]):


def compute_coverage(seed, n, f, fname, save_folder):
    output = {}
    rng = np.random.default_rng(seed)
    X_train = np.linspace(-1, 1, n)
    epsilon = 0.1 * rng.normal(size=n)
    y_train = f(X_train) + epsilon

    # TabPFN
    regressor = TabPFNRegressor(model_path="./tabpfn-v2-regressor.ckpt")
    regressor.fit(X_train.reshape((-1, 1)), y_train)
    X_test = np.arange(-4.5, 4.6, 0.1)
    y_pred = regressor.predict(X_test.reshape((-1, 1)),
                               output_type="full",
                               quantiles=[0.025, 0.975])
    y_true = f(X_test) + 0.1 * rng.normal(size=X_test.shape[0])
    lower = y_pred['quantiles'][0]
    upper = y_pred['quantiles'][1]
    coverage = (lower <= y_true) & (y_true <= upper)
    length = upper - lower
    output['TabPFN_cover'] = coverage
    output['TabPFN_length'] = length

    # GP
    # Define multiple kernel candidates
    kernel_candidates = [
        ConstantKernel() * RBF(),
        ConstantKernel() * Matern(nu=1.5),
        ConstantKernel() * RationalQuadratic(),
        ConstantKernel() * ExpSineSquared(),
        ConstantKernel() * RBF() + ConstantKernel() * Matern()
    ]
    alpha_candidates = [0.05, 0.1, 0.15, 0.2]

    best_score = -np.inf
    best_gpr = None

    for kernel in kernel_candidates:
        for alpha in alpha_candidates:
            gpr = GaussianProcessRegressor(kernel=kernel,
                                           alpha=alpha**2,
                                           n_restarts_optimizer=10)
            gpr.fit(X_train.reshape((-1, 1)), y_train)
            score = gpr.log_marginal_likelihood()
            # print(alpha, score)

            if score > best_score:
                best_score = score
                best_gpr = gpr
                best_alpha = alpha

    y_pred_gp, y_cov_gp = gpr.predict(X_test.reshape((-1, 1)), return_cov=True)
    upper = y_pred_gp + 1.96 * np.sqrt(np.diag(y_cov_gp) + best_alpha**2)
    lower = y_pred_gp - 1.96 * np.sqrt(np.diag(y_cov_gp) + best_alpha**2)
    coverage = (lower <= y_true) & (y_true <= upper)
    length = 2 * 1.96 * np.sqrt(np.diag(y_cov_gp) + best_alpha**2)
    output['GP_cover'] = coverage
    output['GP_length'] = length

    # for key, value in output.items():
    #     print(key, value.mean())

    save_file = os.path.join(
        save_folder,
        "case_" + str(seed) + "_func_" + fname + "_n_" + str(n) + ".pickle",
    )

    f = open(save_file, "wb")
    pickle.dump(output, f)
    f.close()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CATE estimation")
    parser.add_argument("--seed",
                        type=int,
                        default=1,
                        help="index of repetition")
    parser.add_argument("--n", type=int, default=11, help="Training set size")
    parser.add_argument("--f",
                        type=str,
                        default="linear",
                        help="which model to generate data")

    args = parser.parse_args()
    n = int(args.n)
    seed = args.seed
    f = args.f

    save_folder = "./output_coverage/raw_data"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    fname = {
        "linear": linear,
        "quadratic": quadratic,
        "step": step,
        "piecewiselinear": piecewise_linear
    }

    compute_coverage(seed, n, fname[f], f, save_folder)
