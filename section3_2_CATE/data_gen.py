import numpy as np


########################################################################
# Code adapted from
# https://github.com/syrgkanislab/orthogonal_learning/tree/main
########################################################################
def get_data_generator(setup):
    if setup == 'A':
        dist = 'centered_uniform'
        def base_fn(X):
            return np.sin(
                np.pi * X[:, 0] *
                X[:, 1]) + 2 * (X[:, 2] - .5)**2 + X[:, 3] + .5 * X[:, 4]

        def prop_fn(X):
            return np.clip(np.sin(np.pi * X[:, 0] * X[:, 1]), .2, .8)

        def tau_fn(X):
            return .2 + (X[:, 0] + X[:, 1]) / 2
    elif setup == 'B':
        dist = 'centered_uniform'
        def base_fn(X):
            return np.maximum(0, np.maximum(X[:, 0] + X[:, 1],
                                            X[:, 2])) + np.maximum(
                                                X[:, 3] + X[:, 4], 0)

        def prop_fn(X):
            return .5 * np.ones(X.shape[0])

        def tau_fn(X):
            return X[:, 0] + np.log(1 + np.exp(X[:, 1]))
    elif setup == 'C':
        dist = 'centered_uniform'

        def base_fn(X):
            return 2 * np.log(1 + np.exp(np.sum(X[:, :3], axis=1)))

        def prop_fn(X):
            return 1 / (1 + np.exp(X[:, 1] + X[:, 2]))

        def tau_fn(X):
            return np.ones(X.shape[0])
    elif setup == 'D':
        dist = 'centered_uniform'

        def base_fn(X):
            return .5 * (np.maximum(0, np.sum(X[:, :3], axis=1)) +
                         np.maximum(0, X[:, 3] + X[:, 4]))

        def prop_fn(X):
            return 1 / (1 + np.exp(-X[:, 0]) + np.exp(-X[:, 1]))

        def tau_fn(X):
            return np.maximum(0, np.sum(X[:, :3], axis=1)) - np.maximum(
                0, X[:, 3] + X[:, 4])
    elif setup == 'E':
        dist = 'centered_uniform'

        def base_fn(X):
            return 5 * np.maximum(0, X[:, 0] + X[:, 1])

        def prop_fn(X):
            return 1 / (1 + np.exp(3 * X[:, 1] + 3 * X[:, 2]))

        def tau_fn(X):
            return 2 * ((X[:, 0] > 0.1) | (X[:, 1] > 0.1)) - 1
    elif setup == 'F':
        dist = 'centered_uniform'

        def base_fn(X):
            return 5 * np.maximum(0, X[:, 0] + X[:, 1])

        def prop_fn(X):
            return 1 / (1 + np.exp(3 * X[:, 1] + 3 * X[:, 2]))

        def tau_fn(X):
            return X[:, 0] + np.log(1 + np.exp(X[:, 1]))
    # elif setup == 'G':
    #     dist = 'centered_uniform'

    #     def base_fn(X):
    #         return 5 * np.maximum(0, X[:, 0] + X[:, 1])

    #     def prop_fn(X):
    #         return 0.01 * np.ones(X.shape[0])

    #     def tau_fn(X):
    #         return X[:, 0] + np.log(1 + np.exp(X[:, 1]))
    # elif setup == 'H':
    #     dist = 'centered_uniform'

    #     def base_fn(X):
    #         return 0 * np.ones(X.shape[0])

    #     def prop_fn(X):
    #         return 1 / (1 + np.exp(3 * X[:, 1] + 3 * X[:, 2]))

    #     def tau_fn(X):
    #         return np.sin(X[:, 0] * X[:, 1] + 2 * (X[:, 2] - .5)**2 + X[:, 3] +
    #                       .5 * X[:, 4])
    else:
        raise AttributeError(f"Invalid parameter setup={setup}")

    return base_fn, tau_fn, prop_fn, dist


def gen_data(n,
             d,
             base_fn,
             tau_fn,
             prop_fn,
             noise_variance,
             dist,
             random_state,
             reference_seed=None):
    if noise_variance < 0:
        raise ValueError("noise_variance must be nonnegative")
    if reference_seed is None:
        rng = np.random.default_rng(seed=random_state)
        number_of_draws = 1
    else:
        if random_state < 1:
            raise ValueError("random_state must be a positive repetition index")
        rng = np.random.RandomState(reference_seed)
        number_of_draws = random_state
    for _ in range(number_of_draws):
        if dist == 'uniform':
            X = rng.uniform(0, 1, size=(n, d))
            Xtest = rng.uniform(0, 1, size=(10000, d))
        elif dist == 'normal':
            X = rng.normal(0, 1, size=(n, d))
            Xtest = rng.normal(0, 1, size=(10000, d))
        elif dist == 'centered_uniform':
            X = rng.uniform(-.5, .5, size=(n, d))
            Xtest = rng.uniform(-.5, .5, size=(10000, d))
        else:
            raise ValueError(f"Unknown covariate distribution: {dist}")
        T = rng.binomial(1, prop_fn(X))
        y = (T - .5) * tau_fn(X) + base_fn(X) + \
            np.sqrt(noise_variance) * rng.normal(0, 1, size=(n,))
    print(X.shape, T.shape, y.shape, base_fn(X).shape, tau_fn(X).shape)

    return y, T, X, Xtest
