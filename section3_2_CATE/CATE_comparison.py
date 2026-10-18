import os
import pickle
import argparse
import time
import platform
import numpy as np
from importlib.metadata import PackageNotFoundError, version
from cate import *
from data_gen import *
from utils import rmse


def get_package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CATE estimation")
    parser.add_argument("--seed",
                        type=int,
                        default=1,
                        help="index of repetition")
    parser.add_argument("--ss",
                        type=int,
                        default=500,
                        help="Training set size")
    parser.add_argument("--setup",
                        type=str,
                        default="A",
                        choices=["A", "B", "C", "D", "E", "F"],
                        help="which model to generate data")
    parser.add_argument("--tabpfn",
                        action='store_true',
                        help="whether use tabpfn")
    parser.add_argument("--d",
                        type=int,
                        default=6,
                        help="dimension of confounders")
    parser.add_argument("--noise-variance",
                        "--sigma-squared",
                        "--var",
                        dest="noise_variance",
                        type=float,
                        default=0.5,
                        help="variance of the Gaussian outcome noise")
    parser.add_argument("--output-root",
                        type=str,
                        default="./output/save_data",
                        help="root directory for simulation results")
    parser.add_argument("--save-predictions",
                        action="store_true",
                        help="save test-set predictions and truths")
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="overwrite an existing result for this configuration")

    args = parser.parse_args()
    n = int(args.ss)
    seed = args.seed
    setup = args.setup
    noise_variance = args.noise_variance
    d = args.d

    if args.tabpfn:
        save_folder = os.path.join(args.output_root, "cate_tabpfn_paper",
                                   "setup_" + str(setup), "ss_" + str(n),
                                   "variance_" + str(noise_variance),
                                   "d_" + str(d))
    else:
        save_folder = os.path.join(args.output_root, "cate_automl_paper",
                                   "setup_" + str(setup), "ss_" + str(n),
                                   "variance_" + str(noise_variance),
                                   "d_" + str(d))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, "case_" + str(seed) + ".pickle")
    if os.path.exists(save_file) and not args.overwrite:
        print(f"Skipping existing result: {save_file}")
        raise SystemExit(0)

    # Generate data
    base_fn, tau_fn, prop_fn, dist = get_data_generator(setup)
    y, T, X, Xtest = gen_data(n,
                              d,
                              base_fn,
                              tau_fn,
                              prop_fn,
                              noise_variance,
                              dist,
                              seed,
                              reference_seed=123)

    if args.tabpfn:
        methods = [
            # oracle_gen_tabpfn(base_fn, tau_fn, prop_fn), dml_tabpfn,
            dr_tabpfn,
            slearner_tabpfn,
            tlearner_tabpfn,
            xlearner_tabpfn
        ]
        method_name = [
            # 'oracle_tabpfn', 'dml_tabpfn',
            'dr_tabpfn',
            'slearner_tabpfn',
            'tlearner_tabpfn',
            'xlearner_tabpfn'
        ]

    else:
        methods = [
            oracle_gen(base_fn, tau_fn, prop_fn), dml, dr, slearner, tlearner,
            xlearner
        ]
        method_name = [
            'oracle', 'dml', 'dr', 'slearner', 'tlearner', 'xlearner'
        ]
    n_heterogeneity_features = {
        "A": 6,
        "B": 6,
        "C": 6,
        "D": 6,
        "E": 6,
        "F": 6,
    }

    final_result = {}
    method_runtime_seconds = {}

    for i, method in enumerate(methods):
        method_start = time.perf_counter()
        tau_est, est = method(y, T, X, Xtest,
                              n_heterogeneity_features[setup])
        method_runtime_seconds[method_name[i]] = time.perf_counter() - method_start
        mse = rmse(tau_fn(Xtest), tau_est)**2
        final_result[method_name[i] + "_CATE"] = mse
        if args.save_predictions:
            final_result[method_name[i] + "_est"] = tau_est

    if args.save_predictions:
        final_result["CATE_truth"] = tau_fn(Xtest)
    final_result["_metadata"] = {
        "setup": setup,
        "n": n,
        "d": d,
        "n_heterogeneity_features": n_heterogeneity_features[setup],
        "noise_variance": noise_variance,
        "noise_standard_deviation": np.sqrt(noise_variance),
        "seed": seed,
        "covariate_distribution": dist,
        "random_stream": "repetition-th draw after np.random.RandomState(123)",
        "test_size": Xtest.shape[0],
        "method_runtime_seconds": method_runtime_seconds,
        "reference": "CATE experiment settings in the current TabPFN-JASA manuscript",
        "python_version": platform.python_version(),
        "package_versions": {
            package_name: get_package_version(package_name)
            for package_name in ("econml", "flaml", "numpy", "scikit-learn",
                                 "tabpfn")
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }

    for key, value in final_result.items():
        if "CATE" in key:
            print(key, value)

    temporary_file = save_file + ".tmp"
    with open(temporary_file, "wb") as f:
        pickle.dump(final_result, f)
    os.replace(temporary_file, save_file)
