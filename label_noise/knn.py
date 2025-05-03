import os
import copy
import pickle
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


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
    clean = np.zeros((repeat, len(ss)))
    noisy = np.zeros((repeat, len(ss)))

    for i, n in enumerate(ss):
        for j in range(repeat):
            X, y = generate_data(pi, n + 10000, model_option, j)
            X_train, y_train = X[:n], y[:n]
            X_test, y_test = X[n:], y[n:]

            # oracle bayes classifier
            ypred = oracle_classifier(X_test, model_option, pi)
            oracle[j, i] = np.mean(y_test == ypred)

            # clean label
            # Define the parameter values to test
            param_grid = {
                'n_neighbors':
                np.floor(np.linspace(n**(1 / 4), n**(3 / 4),
                                     10)).astype(int)[:2]
            }  # Testing k values from 1 to 30
            # Create KNN classifier
            knn = KNeighborsClassifier()
            # Use GridSearchCV with 5-fold cross-validation
            grid_search = GridSearchCV(knn,
                                       param_grid,
                                       cv=5,
                                       scoring='accuracy')
            grid_search.fit(X_train, y_train)
            # print(grid_search.best_params_['n_neighbors'])

            ypred = grid_search.predict(X_test)
            clean[j, i] = np.mean(y_test == ypred)

            # noisy label
            noisy_y = generate_noisy_label(y_train, noise, j)
            # Define the parameter values to test
            param_grid = {
                'n_neighbors':
                np.floor(np.linspace(n**(1 / 4), n**(3 / 4),
                                     10)).astype(int)[:3]
            }  # Testing k values from 1 to 30
            # Create KNN classifier
            knn = KNeighborsClassifier()
            # Use GridSearchCV with 5-fold cross-validation
            grid_search = GridSearchCV(knn,
                                       param_grid,
                                       cv=5,
                                       scoring='accuracy')
            grid_search.fit(X_train, noisy_y)

            ypred = grid_search.predict(X_test)
            # print(grid_search.best_params_['n_neighbors'])
            noisy[j, i] = np.mean(y_test == ypred)

    output = {
        'oracle': oracle,
        'clean': clean,
        'noisy': noisy,
    }

    save_file = os.path.join(
        save_folder,
        "model_option_" + str(model_option) + "_pi_" + str(pi) + "_noise_" +
        str(noise) + "_knn.pickle",
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
