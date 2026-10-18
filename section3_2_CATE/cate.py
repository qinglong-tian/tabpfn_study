import numpy as np
from flaml import AutoML
from econml.dr import DRLearner
from econml.metalearners import XLearner
from econml.dml import NonParamDML
from tabpfn import TabPFNRegressor
from utils import *


####################################
# methods with automl
####################################
def oracle_gen(base_fn, tau_fn, prop_fn):
    def oracle(y, T, X, Xtest, n_x):
        est = NonParamDML(model_y=OracleY(base_fn=base_fn,
                                          tau_fn=tau_fn,
                                          prop_fn=prop_fn),
                          model_t=OracleT(prop_fn=prop_fn),
                          model_final=final_stage(),
                          discrete_treatment=True,
                          cv=1,
                          random_state=123)
        if n_x == X.shape[1]:
            est.fit(y, T, X=X)
            return est.effect(Xtest), est
        else:
            est.fit(y, T, X=X[:, :n_x], W=X[:, n_x:])
            return est.effect(Xtest[:, :n_x]), est

    return oracle


def dml(y, T, X, Xtest, n_x):
    model_y = first_stage_reg(X, y)
    model_t = first_stage_clf(X, T)
    est = NonParamDML(model_y=model_y,
                      model_t=model_t,
                      model_final=final_stage(),
                      discrete_treatment=True,
                      cv=10,
                      random_state=123)
    if n_x == X.shape[1]:
        est.fit(y, T, X=X)
        return est.effect(Xtest), est
    else:
        est.fit(y, T, X=X[:, :n_x], W=X[:, n_x:])
        return est.effect(Xtest[:, :n_x]), est


def dr(y, T, X, Xtest, n_x):
    model_regression = first_stage_reg(np.hstack([X, T.reshape(-1, 1)]), y)
    model_propensity = first_stage_clf(X, T)
    est = DRLearner(model_regression=model_regression,
                    model_propensity=model_propensity,
                    model_final=final_stage(),
                    min_propensity=.1,
                    cv=10,
                    random_state=123)
    if n_x == X.shape[1]:
        est.fit(y, T, X=X)
        return est.effect(Xtest), est
    else:
        est.fit(y, T, X=X[:, :n_x], W=X[:, n_x:])
        return est.effect(Xtest[:, :n_x]), est


def slearner(y, T, X, Xtest, n_x):
    X_new = np.hstack([T.reshape(-1, 1), X])
    overall_model = first_stage_reg(X_new, y)
    overall_model.fit(X_new, y)
    treated_outcome = overall_model.predict(
        np.hstack([np.ones((Xtest.shape[0], 1)), Xtest]))
    control_outcome = overall_model.predict(
        np.hstack([np.zeros((Xtest.shape[0], 1)), Xtest]))
    tau_est = treated_outcome - control_outcome
    return tau_est, overall_model


def tlearner(y, T, X, Xtest, n_x):
    regressor1 = AutoML(task='regression',
                        time_budget=60,
                        early_stop=True,
                        eval_method='cv',
                        n_splits=5,
                        metric='mse',
                        verbose=0)
    regressor1.fit(X[T == 1], y[T == 1])

    y_test1 = regressor1.predict(Xtest)

    regressor0 = AutoML(task='regression',
                        time_budget=60,
                        early_stop=True,
                        eval_method='cv',
                        n_splits=5,
                        metric='mse',
                        verbose=0)
    regressor0.fit(X[T == 0], y[T == 0])
    y_test0 = regressor0.predict(Xtest)
    tau_est = y_test1 - y_test0
    return tau_est, (regressor0, regressor1)


def xlearner(y, T, X, Xtest, n_x):
    # model_regression = first_stage_reg(np.hstack([X, T.reshape(-1, 1)]), y)
    model_propensity = first_stage_clf(X, T)
    est = XLearner(
        models=final_stage(),
        #    cate_models=model_regression,
        propensity_model=model_propensity)
    # min_propensity=.1,
    # cv=10,
    # random_state=123)
    est.fit(y, T, X=X)
    return est.effect(Xtest), est


####################################
# methods with TabPFN
####################################
def oracle_gen_tabpfn(base_fn, tau_fn, prop_fn):

    def oracle_tabpfn(y, T, X, Xtest, n_x):
        est = NonParamDML(model_y=OracleY(base_fn=base_fn,
                                          tau_fn=tau_fn,
                                          prop_fn=prop_fn),
                          model_t=OracleT(prop_fn=prop_fn),
                          model_final=tabpfnfinal(),
                          discrete_treatment=True,
                          cv=1,
                          random_state=123)
        if n_x == X.shape[1]:
            est.fit(y, T, X=X)
            return est.effect(Xtest), est
        else:
            est.fit(y, T, X=X[:, :n_x], W=X[:, n_x:])
            return est.effect(Xtest[:, :n_x]), est

    return oracle_tabpfn


def dml_tabpfn(y, T, X, Xtest, n_x):
    est = NonParamDML(model_y=tabpfnY(),
                      model_t=tabpfnT(),
                      model_final=tabpfnfinal(),
                      discrete_treatment=True,
                      cv=10,
                      random_state=123)
    if n_x == X.shape[1]:
        est.fit(y, T, X=X)
        return est.effect(Xtest), est
    else:
        est.fit(y, T, X=X[:, :n_x], W=X[:, n_x:])
        return est.effect(Xtest[:, :n_x]), est


def dr_tabpfn(y, T, X, Xtest, n_x):
    est = DRLearner(model_regression=tabpfnY(),
                    model_propensity=tabpfnT(),
                    model_final=tabpfnfinal(),
                    min_propensity=.1,
                    cv=10,
                    random_state=123)
    if n_x == X.shape[1]:
        est.fit(y, T, X=X)
        return est.effect(Xtest), est
    else:
        est.fit(y, T, X=X[:, :n_x], W=X[:, n_x:])
        return est.effect(Xtest[:, :n_x]), est


def slearner_tabpfn(y, T, X, Xtest, n_x):
    X_new = np.hstack([T.reshape(-1, 1), X])

    overall_model = TabPFNRegressor(model_path=TABPFN_REGRESSOR_MODEL_PATH,
                                    device=TABPFN_DEVICE)
    overall_model.fit(X_new, y)
    treated_outcome = overall_model.predict(
        np.hstack([np.ones((Xtest.shape[0], 1)), Xtest]))
    control_outcome = overall_model.predict(
        np.hstack([np.zeros((Xtest.shape[0], 1)), Xtest]))
    tau_est = treated_outcome - control_outcome
    return tau_est, overall_model


def tlearner_tabpfn(y, T, X, Xtest, n_x):
    regressor1 = TabPFNRegressor(model_path=TABPFN_REGRESSOR_MODEL_PATH,
                                 device=TABPFN_DEVICE)
    regressor1.fit(X[T == 1], y[T == 1])
    y_test1 = regressor1.predict(Xtest)

    regressor0 = TabPFNRegressor(model_path=TABPFN_REGRESSOR_MODEL_PATH,
                                 device=TABPFN_DEVICE)
    regressor0.fit(X[T == 0], y[T == 0])
    y_test0 = regressor0.predict(Xtest)
    tau_est = y_test1 - y_test0
    return tau_est, (regressor0, regressor1)


# def xlearner_tabpfn(y, T, X, Xtest, n_x):
#     regressor1 = TabPFNRegressor(model_path="./tabpfn-v2-regressor.ckpt")
#     regressor1.fit(X[T == 1, :n_x], y[T == 1])
#     y_test1 = regressor1.predict(Xtest[:, :n_x])

#     regressor0 = TabPFNRegressor(model_path="./tabpfn-v2-regressor.ckpt")
#     regressor0.fit(X[T == 0, :n_x], y[T == 0])
#     y_test0 = regressor0.predict(Xtest[:, :n_x])

#     return est.effect(Xtest[:, :n_x]), est


def xlearner_tabpfn(y, T, X, Xtest, n_x):

    est = XLearner(
        models=tabpfnfinal(),
        #    cate_models=model_regression,
        propensity_model=tabpfnT())
    # min_propensity=.1,
    # cv=10,
    # random_state=123)
    est.fit(y, T, X=X)
    return est.effect(Xtest), est
