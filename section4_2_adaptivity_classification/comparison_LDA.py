import os
import copy
import pickle
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


#-------------------------------------------------
# Helper functions
#-------------------------------------------------
def generate_data(pi, n=1000, model_no=1, seed=1):
    rng = np.random.default_rng(seed)
    if model_no == 1:
        d = 5
        X = rng.normal(size=(n, d))
        mu = np.array([[3 / 2, 0, 0, 0, 0], [-3 / 2, 0, 0, 0, 0]])
        Y = rng.choice(2, size=n, p=[1 - pi, pi])
        X[Y == 0] = X[Y == 0] + mu[0]
        X[Y == 1] = X[Y == 1] + mu[1]
    elif model_no == 2:
        d = 5
        X = rng.uniform(size=(n, d))
        score = np.minimum(4 * (X[:, 0] - 0.5)**2 + 4 * (X[:, 1] - 0.5)**2, 1)
        Y = rng.binomial(1, score)
    return X, Y


def oracle_classifier(X, model_no=1, pi=0.5):
    if model_no == 1:
        eta = 1 / (1 + ((1 - pi) / pi) * np.exp(3 * X[:, 0]))
    elif model_no == 2:
        eta = np.minimum(4 * (X[:, 0] - 0.5)**2 + 4 * (X[:, 1] - 0.5)**2, 1)
    pred = (eta >= 0.5)
    return pred


def generate_noisy_label(y, noise, random_state=0):
    rng = np.random.default_rng(random_state)
    assert (noise >= 0.0) and (noise <= 1.0)

    if noise == 0.0:
        return y
    else:
        flip = rng.binomial(1, noise, size=y.shape[0])
        noisy_y = copy.deepcopy(y)
        noisy_y[flip == 1] = 1 - noisy_y[flip == 1]
    return noisy_y


#-------------------------------------------------
# Main function
#-------------------------------------------------
def main(model_option, noise, save_folder, pi=0.5):
    repeat = 1000
    ss = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    oracle = np.zeros((repeat, len(ss)))
    clean_lr = np.zeros((repeat, len(ss)))
    noisy_lr = np.zeros((repeat, len(ss)))
    clean_tabpfn = np.zeros((repeat, len(ss)))
    noisy_tabpfn = np.zeros((repeat, len(ss)))

    for i, n in enumerate(ss):
        for j in range(repeat):
            X, y = generate_data(pi, n + 10000, model_option, j)
            X_train, y_train = X[:n], y[:n]
            X_test, y_test = X[n:], y[n:]

            # oracle bayes classifier
            ypred = oracle_classifier(X_test, model_option, pi)
            oracle[j, i] = np.mean(y_test == ypred)

            # clean label
            if model_option == 1:
                ml = LogisticRegression(penalty=None,
                                        tol=1e-8,
                                        fit_intercept=True,
                                        max_iter=10000)
                ml.fit(X_train, y_train)

            elif model_option == 2:  # use LDA
                ml = LDA()
                ml.fit(X_train, y_train)
            ypred = ml.predict(X_test)
            clean_lr[j, i] = np.mean(y_test == ypred)

            ml = TabPFNClassifier(model_path="./tabpfn-v2-classifier.ckpt")
            ml.fit(X_train, y_train)
            ypred = ml.predict(X_test)
            clean_tabpfn[j, i] = np.mean(y_test == ypred)

            # noisy label
            noisy_y = generate_noisy_label(y_train, noise, j)
            if model_option == 1:
                ml = LogisticRegression(penalty=None,
                                        tol=1e-8,
                                        fit_intercept=True,
                                        max_iter=10000)
                ml.fit(X_train, noisy_y)
            elif model_option == 2:  # use LDA
                ml = LDA()
                ml.fit(X_train, noisy_y)

            ypred = ml.predict(X_test)
            noisy_lr[j, i] = np.mean(y_test == ypred)

            ml = TabPFNClassifier(model_path="./tabpfn-v2-classifier.ckpt")
            ml.fit(X_train, noisy_y)
            ypred = ml.predict(X_test)
            noisy_tabpfn[j, i] = np.mean(y_test == ypred)

    output = {
        'oracle': oracle,
        'clean_lr': clean_lr,
        'noisy_lr': noisy_lr,
        'clean_tabpfn': clean_tabpfn,
        'noisy_tabpfn': noisy_tabpfn
    }

    save_file = os.path.join(
        save_folder,
        "model_option_" + str(model_option) + "_pi_" + str(pi) + "_noise_" +
        str(noise) + "_lr.pickle",
    )

    f = open(save_file, "wb")
    pickle.dump(output, f)
    f.close()

    return


save_folder = os.path.join("./output")
if not os.path.exists(save_folder):
    os.makedirs(save_folder, exist_ok=True)

for noise in [0.1, 0.2, 0.3, 0.4]:
    main(1, noise, save_folder, pi=0.9)
    main(2, noise, save_folder, pi=0.5)
