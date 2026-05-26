import os
import pickle
import argparse
from cate import *
from data_gen import *
from utils import rmse

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
                        help="which model to generate data")
    parser.add_argument("--tabpfn",
                        action='store_true',
                        help="whether use tabpfn")
    parser.add_argument("--d",
                        type=int,
                        default=6,
                        help="dimension of confounders")
    parser.add_argument("--var",
                        type=float,
                        default=0.5,
                        help="variance of noise")

    args = parser.parse_args()
    n = int(args.ss)
    seed = args.seed
    setup = args.setup
    sigma = args.var
    d = args.d

    if args.tabpfn:
        save_folder = os.path.join("./output/save_data/cate_tabpfn",
                                   "setup_" + str(setup), "ss_" + str(n),
                                   "var_" + str(sigma), "d_" + str(d))
    else:
        save_folder = os.path.join("./output/save_data/cate",
                                   "setup_" + str(setup), "ss_" + str(n),
                                   "var_" + str(sigma), "d_" + str(d))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    # Generate data
    base_fn, tau_fn, prop_fn, dist = get_data_generator(setup)
    y, T, X, Xtest = gen_data_drcase(n, d, base_fn, tau_fn, prop_fn, sigma,
                                     seed)

    if args.tabpfn:
        methods = [
            dr_tabpfn, slearner_tabpfn, tlearner_tabpfn, xlearner_tabpfn
        ]
        method_name = [
            'dr_tabpfn', 'slearner_tabpfn', 'tlearner_tabpfn',
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
    setups = ["A", "B", "C", "D", "E", "F"]

    nx = [4, 4, 4, 5, 4, 4]

    final_result = {}

    for i, method in enumerate(methods):
        tau_est, est = method(y, T, X, Xtest, nx[setups.index(setup)])
        mse = rmse(tau_fn(Xtest), tau_est)**2
        final_result[method_name[i] + "_CATE"] = mse
        final_result[method_name[i] + "_est"] = tau_est
        final_result[method_name[i] + "_truth"] = tau_fn(Xtest)

    for key, value in final_result.items():
        if "CATE" in key:
            print(key, value)

    save_file = os.path.join(
        save_folder,
        "case_" + str(seed) + ".pickle",
    )

    f = open(save_file, "wb")
    pickle.dump(final_result, f)
    f.close()
