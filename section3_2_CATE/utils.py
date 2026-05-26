import numpy as np
from flaml import AutoML
from tabpfn import TabPFNClassifier, TabPFNRegressor
from sklearn.base import BaseEstimator, clone
import warnings

warnings.simplefilter('ignore')


####################################
# utilities
####################################
def rmse(ytrue, y):
    return np.sqrt(np.mean((ytrue.flatten() - y.flatten())**2))


class tabpfnT(BaseEstimator):

    def __init__(self):
        self.classifier = TabPFNClassifier()

    def fit(self, X, T, **kwargs):
        self.classifier.fit(X, T)
        return self

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)


class tabpfnY(BaseEstimator):

    def __init__(self):
        self.regressor = TabPFNRegressor()

    def fit(self, X, y, **kwargs):
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        return self.regressor.predict(X)


class tabpfnfinal(BaseEstimator):

    def __init__(self):
        self.regressor = TabPFNRegressor()

    def fit(self, X, y, **kwargs):
        self.regressor = self.regressor.fit(X, y)
        return self

    def predict(self, X):
        return self.regressor.predict(X)


class OracleY(BaseEstimator):

    def __init__(self, *, base_fn, tau_fn, prop_fn):
        self.base_fn = base_fn
        self.tau_fn = tau_fn
        self.prop_fn = prop_fn

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.base_fn(X) + self.tau_fn(X) * (self.prop_fn(X) - .5)


class OracleT(BaseEstimator):

    def __init__(self, *, prop_fn):
        self.prop_fn = prop_fn

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        prob = self.prop_fn(X).reshape(-1, 1)
        return np.hstack([1 - prob, prob])


class AutoMLWrapclf(BaseEstimator):

    def __init__(self, *, model):
        self.model = model

    def fit(self, X, y, **kwargs):
        self.model_ = clone(self.model)
        self.model_.fit(X, y, **kwargs)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class AutoMLWrapreg(BaseEstimator):

    def __init__(self, *, model):
        self.model = model

    def fit(self, X, y, **kwargs):
        self.model_ = clone(self.model)
        self.model_.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def first_stage_reg(X, y):
    automl = AutoML(task='regression',
                    time_budget=60,
                    early_stop=True,
                    eval_method='cv',
                    n_splits=2,
                    metric='mse',
                    verbose=0)
    automl.fit(X, y)
    best_est = automl.best_estimator
    return AutoMLWrapreg(
        model=clone(automl.best_model_for_estimator(best_est)))


def first_stage_clf(X, y):
    automl = AutoML(task='classification',
                    time_budget=60,
                    early_stop=True,
                    eval_method='cv',
                    n_splits=2,
                    metric='accuracy',
                    verbose=0)
    automl.fit(X, y)
    best_est = automl.best_estimator
    return AutoMLWrapclf(
        model=clone(automl.best_model_for_estimator(best_est)))


# We create a custom metric to handle sample weights as we want them in RLearner (NonParamDML).
# We want to be minimizing the loss: 1/n sum_i w_i (y_i - ypred_i)^2. The standard
# mse with sample weights would have minimized (1/sum_i w_i) sum_i w_i (y_i - ypred_i)^2.
def weighted_mse(
    X_val,
    y_val,
    estimator,
    labels,
    X_train,
    y_train,
    weight_val=None,
    weight_train=None,
    *args,
):
    y_pred = estimator.predict(X_val)
    weight_val = 1 if weight_val is None else weight_val
    weight_train = 1 if weight_train is None else weight_train
    error = (estimator.predict(X_val) - y_val)**2
    val_loss = np.mean(weight_val * error)
    error_train = (estimator.predict(X_train) - y_train)**2
    train_loss = np.mean(weight_train * error_train)
    return val_loss, {"val_loss": val_loss, "train_loss": train_loss}


def final_stage():
    return AutoMLWrapreg(model=AutoML(task='regression',
                                      time_budget=60,
                                      early_stop=True,
                                      eval_method='cv',
                                      n_splits=2,
                                      metric=weighted_mse,
                                      verbose=0))
