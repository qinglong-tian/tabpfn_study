import os
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import BaseEstimator
from econml.dr import DRLearner
from econml.metalearners import XLearner
from tabpfn import TabPFNRegressor, TabPFNClassifier
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


class tabpfnT(BaseEstimator):

    def __init__(self):
        self.classifier = TabPFNClassifier(
            model_path="./tabpfn-v2-classifier.ckpt")

    def fit(self, X, T, **kwargs):
        self.classifier.fit(X, T)
        return self

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)


class tabpfnY(BaseEstimator):

    def __init__(self):
        self.regressor = TabPFNRegressor(
            model_path="./tabpfn-v2-regressor.ckpt")

    def fit(self, X, y, **kwargs):
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        return self.regressor.predict(X)


class tabpfnfinal(BaseEstimator):

    def __init__(self):
        self.regressor = TabPFNRegressor(
            model_path="./tabpfn-v2-regressor.ckpt")

    def fit(self, X, y, **kwargs):
        self.regressor = self.regressor.fit(X, y)
        return self

    def predict(self, X):
        return self.regressor.predict(X)


def slearner_tabpfn(y, T, X):
    X_new = np.hstack([X, T.reshape(-1, 1)])

    regressor = TabPFNRegressor(model_path='./tabpfn-v2-regressor.ckpt')
    regressor.fit(X_new, y)

    X_test1 = np.hstack([X, np.ones((X.shape[0], 1))])
    X_test0 = np.hstack([X, np.zeros((X.shape[0], 1))])

    y_test1 = regressor.predict(X_test1)
    y_test0 = regressor.predict(X_test0)
    tau_est = y_test1 - y_test0
    return tau_est


def sbcf_tabpfn(y, T, X):
    classifier = TabPFNClassifier(model_path='./tabpfn-v2-classifier.ckpt')
    classifier.fit(X, T)
    scores = classifier.predict_proba(X)[:, 1]

    X_new = np.hstack([X, T.reshape(-1, 1), scores.reshape(-1, 1)])

    regressor = TabPFNRegressor(model_path='./tabpfn-v2-regressor.ckpt')
    regressor.fit(X_new, y)

    X_test1 = np.hstack([X, np.ones((X.shape[0], 1)), scores.reshape(-1, 1)])
    X_test0 = np.hstack([X, np.zeros((X.shape[0], 1)), scores.reshape(-1, 1)])

    y_test1 = regressor.predict(X_test1)
    y_test0 = regressor.predict(X_test0)
    tau_est = y_test1 - y_test0
    return tau_est


def xbcf_tabpfn(y, T, X):
    prop_model = TabPFNClassifier(model_path="./tabpfn-v2-classifier.ckpt")
    prop_model.fit(X, T)
    prop_scores = prop_model.predict_proba(X)[:, 1]

    X = np.hstack([X, prop_scores.reshape(-1, 1)])
    est = XLearner(models=tabpfnfinal(), propensity_model=tabpfnT())
    est.fit(y, T, X=X)
    return est.effect(X)

    # X_new = np.hstack([X, T.reshape(-1, 1), prop_scores.reshape(-1, 1)])
    # regressor = TabPFNRegressor(model_path='./tabpfn-v2-regressor.ckpt')
    # regressor.fit(X_new, y)

    # X_test1 = np.hstack([X, np.ones((X.shape[0], 1)), prop_scores.reshape(-1,1)])
    # X_test0 = np.hstack([X, np.zeros((X.shape[0], 1)), prop_scores.reshape(-1,1)])

    # y_test1 = regressor.predict(X_test1)
    # y_test0 = regressor.predict(X_test0)
    # tau_est = y_test1 - y_test0
    # return tau_est


def tlearner_tabpfn(y, T, X):
    regressor1 = TabPFNRegressor(model_path="./tabpfn-v2-regressor.ckpt")
    regressor1.fit(X[T == 1], y[T == 1])
    y_test1 = regressor1.predict(X)

    regressor0 = TabPFNRegressor(model_path="./tabpfn-v2-regressor.ckpt")
    regressor0.fit(X[T == 0], y[T == 0])
    y_test0 = regressor0.predict(X)
    tau_est = y_test1 - y_test0
    return tau_est


def xlearner_tabpfn(y, T, X):
    est = XLearner(models=tabpfnfinal(), propensity_model=tabpfnT())
    est.fit(y, T, X=X)
    return est.effect(X)


def dr_tabpfn(y, T, X):
    est = DRLearner(model_regression=tabpfnY(),
                    model_propensity=tabpfnT(),
                    model_final=tabpfnfinal(),
                    random_state=123)

    est.fit(y, T, X=X)
    return est.effect(X)


def main(setup, scenario, method, save_folder):
    print(setup, scenario, method)
    # for key, val in output.items():
    #     print(key, val)
    X = pd.read_csv('contest_data/X.csv', header=None)
    columns_to_extract = [0, 2, 9, 13, 14, 20, 23, 42]
    # if method !="x":
    #     X = X.to_numpy()[:,columns_to_extract]
    # else:
    X = X.iloc[:, columns_to_extract]
    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=[20, 23], drop_first=False)
    # Convert back to numpy array
    X = X.to_numpy(dtype=float)

    tau_true = np.loadtxt(
        os.path.join('contest_data', setup, scenario, 'dgp.csv'),
        delimiter=',',
        skiprows=1,  # Skip header
        dtype=float)[:, 0]

    nrep = 250
    att_rmse = np.zeros(nrep)
    cate_rmse = np.zeros(nrep)

    for i in tqdm(range(nrep)):
        if setup != "nonadditive":
            simulated_data = np.loadtxt(os.path.join('contest_data', setup,
                                                     scenario,
                                                     str(i + 1) + '.csv'),
                                        delimiter=',')
        else:
            simulated_data = np.loadtxt(os.path.join('contest_data', setup,
                                                     scenario,
                                                     str(i + 1) + '.csv'),
                                        delimiter=',',
                                        skiprows=1)
        T, y = simulated_data[:, 0], simulated_data[:, 1]
        # print(X.shape, T.shape, y.shape)
        if method == "s":
            tau_est = slearner_tabpfn(y, T, X)
        elif method == "t":
            tau_est = tlearner_tabpfn(y, T, X)
        elif method == "x":
            tau_est = xlearner_tabpfn(y, T, X)
        elif method == "dr":
            tau_est = dr_tabpfn(y, T, X)
        elif method == "xbcf":
            tau_est = xbcf_tabpfn(y, T, X)
        elif method == "sbcf":
            tau_est = sbcf_tabpfn(y, T, X)
        tau_att_true = np.mean(tau_true[T == 1])
        tau_att_est = np.mean(tau_est[T == 1])
        rmse_cate = np.sqrt(np.mean((tau_est - tau_true)**2))
        rmse_att = np.abs(tau_att_true - tau_att_est)
        att_rmse[i] = rmse_att
        cate_rmse[i] = rmse_cate

    # print(att_rmse, cate_rmse)
    # save files
    save_file = os.path.join(
        save_folder,
        "rmse" + "_" + method + ".pickle",
    )
    output = {}
    output["att_rmse"] = att_rmse
    output["cate_rmse"] = cate_rmse

    print("CATE RMSE", cate_rmse.mean())

    f = open(save_file, "wb")
    pickle.dump(output, f)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACIC 2017 CATE estimation")
    parser.add_argument("--setup",
                        type=str,
                        default='iid',
                        help="which setting")
    parser.add_argument("--scenario", type=str, default='000')
    parser.add_argument("--method", type=str, default='s')
    args = parser.parse_args()
    setup = args.setup
    scenario = args.scenario
    method = args.method
    save_folder = os.path.join("./output_sbcf/", setup, scenario)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    main(setup, scenario, method, save_folder)
