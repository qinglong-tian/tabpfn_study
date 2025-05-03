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
                          cv=5,
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
                      cv=5,
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
                    cv=2,
                    random_state=123)
    if n_x == X.shape[1]:
        est.fit(y, T, X=X)
        return est.effect(Xtest), est
    else:
        est.fit(y, T, X=X[:, :n_x], W=X[:, n_x:])
        return est.effect(Xtest[:, :n_x]), est


def slearner(y, T, X, Xtest, n_x):
    X_new = np.hstack([X[:, :n_x], T.reshape(-1, 1)])
    regressor = AutoML(task='regression',
                       time_budget=60,
                       early_stop=True,
                       eval_method='cv',
                       n_splits=2,
                       metric='mse',
                       verbose=0)
    # print(n_x, X_new.shape)
    regressor.fit(X_new, y)
    X_test1 = np.hstack([Xtest[:, :n_x], np.ones((Xtest.shape[0], 1))])
    X_test0 = np.hstack([Xtest[:, :n_x], np.zeros((Xtest.shape[0], 1))])
    # print(X_test1.shape, X_test0.shape)

    y_test1 = regressor.predict(X_test1)
    y_test0 = regressor.predict(X_test0)
    tau_est = y_test1 - y_test0
    return tau_est, regressor


def tlearner(y, T, X, Xtest, n_x):
    regressor1 = AutoML(task='regression',
                        time_budget=60,
                        early_stop=True,
                        eval_method='cv',
                        n_splits=2,
                        metric='mse',
                        verbose=0)
    regressor1.fit(X[T == 1, :n_x], y[T == 1])

    y_test1 = regressor1.predict(Xtest[:, :n_x])

    regressor0 = AutoML(task='regression',
                        time_budget=60,
                        early_stop=True,
                        eval_method='cv',
                        n_splits=2,
                        metric='mse',
                        verbose=0)
    regressor0.fit(X[T == 0, :n_x], y[T == 0])
    y_test0 = regressor0.predict(Xtest[:, :n_x])
    tau_est = y_test1 - y_test0
    return tau_est, (regressor0, regressor1)


def xlearner(y, T, X, Xtest, n_x):
    model_propensity = first_stage_clf(X, T)
    est = XLearner(models=final_stage(), propensity_model=model_propensity)
    est.fit(y, T, X=X[:, :n_x])
    return est.effect(Xtest[:, :n_x]), est


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
                    random_state=123)
    if n_x == X.shape[1]:
        est.fit(y, T, X=X)
        return est.effect(Xtest), est
    else:
        est.fit(y, T, X=X[:, :n_x], W=X[:, n_x:])
        return est.effect(Xtest[:, :n_x]), est


def slearner_tabpfn(y, T, X, Xtest, n_x):
    X_new = np.hstack([X[:, :n_x], T.reshape(-1, 1)])

    regressor = TabPFNRegressor()
    regressor.fit(X_new, y)

    X_test1 = np.hstack([Xtest[:, :n_x], np.ones((Xtest.shape[0], 1))])
    X_test0 = np.hstack([Xtest[:, :n_x], np.zeros((Xtest.shape[0], 1))])

    y_test1 = regressor.predict(X_test1)
    y_test0 = regressor.predict(X_test0)
    tau_est = y_test1 - y_test0
    return tau_est, regressor


def tlearner_tabpfn(y, T, X, Xtest, n_x):
    regressor1 = TabPFNRegressor()
    regressor1.fit(X[T == 1, :n_x], y[T == 1])
    y_test1 = regressor1.predict(Xtest[:, :n_x])

    regressor0 = TabPFNRegressor()
    regressor0.fit(X[T == 0, :n_x], y[T == 0])
    y_test0 = regressor0.predict(Xtest[:, :n_x])
    tau_est = y_test1 - y_test0
    return tau_est, (regressor0, regressor1)


def xlearner_tabpfn(y, T, X, Xtest, n_x):
    est = XLearner(models=tabpfnfinal(), propensity_model=tabpfnT())
    est.fit(y, T, X=X[:, :n_x])
    return est.effect(Xtest[:, :n_x]), est
