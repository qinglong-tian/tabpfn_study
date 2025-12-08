import os
import arff
import pickle
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from tabpfn import TabPFNRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# path to your file and the corresponding name of the response
# Dataset: TabArena on OpenML
# Selection criteria: # of samples >=5K, # features < 1K
reg_dataset_dict = {
    'data/reg/diamonds.arff': 'price',
    'data/reg/fifa.arff': 'wage_eur',
    'data/reg/Food_Delivery_Time.arff': 'Time_taken(min)',
    'data/reg/miami_housing.arff': 'SALE_PRC',
    # 'data/reg/physiochemical_protein.arff': 'ResidualSize',
    'data/reg/superconductivity.arff': 'critical_temp',
    'data/reg/houses.arff': 'LnMedianHouseValue',
    'data/reg/wine_quality.arff': 'median_wine_quality',
    'data/reg/weather_izmir.arff': 'Mean_temperature',
    'data/reg/cmc.arff': 'class',
    'data/reg/physiochemical_protein.arff': 'RMSD',
    'data/reg/nyc-taxi-green-dec-2016.arff': 'tipamount',
    'data/reg/BNG_stock.arff': 'company10',
    'data/reg/puma8NH.arff':'thetadd3',
    'data/reg/house_8L.arff':'price',
    'data/reg/kin8nm.arff':'y'
}


def load_arff_file(file_path):
    """Load ARFF file with proper encoding handling"""
    with open(file_path, 'r') as f:
        data = arff.load(f)
    # convert to pandas DataFrame
    df = pd.DataFrame(data['data'],
                      columns=[attr[0] for attr in data['attributes']])

    # Convert bytes columns to strings
    for col in df.select_dtypes([object]):
        if df[col].apply(lambda x: isinstance(x, bytes)).any():
            df[col] = df[col].str.decode('utf-8')

    return df


def preprocess_dataset(df, target_col):
    """Preprocess dataset"""
    # Step 1: Remove rows with missing values
    initial_shape = df.shape
    df_clean = df.dropna()
    print(
        f"Removed {initial_shape[0] - df_clean.shape[0]} rows with missing values"
    )

    # Step 2: Separate features and target
    if target_col not in df_clean.columns:
        available_cols = df_clean.columns.tolist()
        raise ResourceWarning(
            f"Target column '{target_col}' not found. Available columns: {available_cols}"
        )

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    # 对目标变量进行编码（如果是字符串）
    if y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index, name=y.name)

    # Step 3: Identify column types
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(categorical_cols)
    print(numerical_cols)

    # Step 4: Create dummy variables for categorical features
    if categorical_cols:
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
    else:
        X_encoded = X.copy()

    scaler = StandardScaler()
    if numerical_cols:
        X_encoded[numerical_cols] = scaler.fit_transform(
            X_encoded[numerical_cols])

    y_normalized = scaler.fit_transform(y.values.reshape(-1, 1))
    y = pd.Series(y_normalized.flatten(), index=y.index, name=y.name)

    print(f"Final processed shape: {X_encoded.shape}")

    return X_encoded, X_encoded, y


def combine_xy(X1, Y1, X2=None, Y2=None):
    Xs = [X1]
    ys = [pd.Series(Y1)]
    if X2 is not None and Y2 is not None:
        Xs.append(X2)
        ys.append(pd.Series(Y2))
    X = pd.concat(Xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True).rename("target")
    return pd.concat([y, X], axis=1)


def setup_r_env(R_HOME, R_SCRIPT1, R_SCRIPT2, tau_0, option):
    os.environ["R_HOME"] = R_HOME
    os.environ["R_USER"] = os.environ["R_HOME"]

    wd = pathlib.Path.cwd().as_posix()
    ro.r(f'setwd("{wd}")')

    ro.r('suppressMessages(library(MASS))')
    ro.r('suppressMessages(try(library(caret), silent=TRUE))')
    # ro.r('suppressMessages(library(quantreg))')
    ro.r(f'source("{R_SCRIPT1}")')
    ro.r(f'source("{R_SCRIPT2}")')

    if option in ["i", "W1", "S1"]:
        r_type = "linear"
    elif option in ["ii", "W2", "S2"]:
        r_type = "logistic"
    else:
        r_type = "quantile"

    ro.r(f'tau_0 <- {tau_0}')
    ro.r(f'option <- "{option}"')
    ro.r(f'type <- "{r_type}"')
    return ro


def to_r_matrix(ro, df: pd.DataFrame, name_in_r: str):
    """Send a pandas DataFrame to R as a numeric matrix."""
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv[name_in_r] = df.reset_index(
            drop=True).infer_objects().astype(float)
    ro.r(f'{name_in_r} <- as.matrix({name_in_r})')


def run_r_estimations(ro, df_labeled, df_imputed, df_train_imputed,
                      df_all_imputed, X_unlabeled):
    # Push to R and run estimators
    to_r_matrix(ro, df_labeled, "data_labeled")
    ro.r(
        'estimation_sup <- SupervisedEst(data_labeled, tau=tau_0, option=option); theta_sup <- estimation_sup$Est.coef'
    )
    to_r_matrix(ro, df_imputed, "data_imputed")
    ro.r(
        'estimation_imputed <- SupervisedEst(data_imputed, tau=tau_0, option=option); theta_imputed <- estimation_imputed$Est.coef'
    )

    to_r_matrix(ro, df_train_imputed, "data_train_imputed")
    ro.r(
        'estimation_train_imputed <- SupervisedEst(data_train_imputed, tau=tau_0, option=option); theta_train_imputed <- estimation_train_imputed$Est.coef'
    )

    to_r_matrix(ro, df_all_imputed, "data_all_imputed")
    ro.r(
        'estimation_all_imputed <- SupervisedEst(data_all_imputed, tau=tau_0, option=option); theta_all_imputed <- estimation_all_imputed$Est.coef'
    )

    to_r_matrix(ro, X_unlabeled, "data_unlabeled")
    ro.r(
        'estimation_SLZ <- PSSE(data_labeled, data_unlabeled, type=type, tau=tau_0); theta_SLZ <- estimation_SLZ$Hattheta'
    )

    # Fetch back to Python
    theta_supervised = np.asarray(ro.r('theta_sup'), dtype=float)
    theta_imputed = np.asarray(ro.r('theta_imputed'), dtype=float)
    theta_all_imputed = np.asarray(ro.r('theta_all_imputed'), dtype=float)
    theta_train_imputed = np.asarray(ro.r('theta_train_imputed'), dtype=float)
    theta_SLZ = np.asarray(ro.r('theta_SLZ'), dtype=float)

    # Debias
    theta_bias = theta_supervised - theta_train_imputed
    theta_debiased = theta_all_imputed + theta_bias

    return {
        "supervised": theta_supervised,
        "imputed": theta_imputed,
        "debiased": theta_debiased,
        "SLZ": theta_SLZ,
    }


def comparison(dataset, label_size=10):
    file_path = os.path.join('data/reg/', dataset + '.arff')
    df = load_arff_file(file_path)
    X, X_encoded, y = preprocess_dataset(df, reg_dataset_dict[file_path])

    results = {}

    # Set up R once
    ro = setup_r_env(R_HOME, R_SCRIPT1, R_SCRIPT2, 0.5,
                     "i")  # Logistic regression

    # Oracle estimator
    df_full = combine_xy(X_encoded, y)  # train only
    to_r_matrix(ro, df_full, "data_labelled_full")
    ro.r(
        'estimation_oracle <- SupervisedEst(data_labelled_full, tau=tau_0, option=option); theta_oracle <- estimation_oracle$Est.coef'
    )
    theta_oracle = np.asarray(ro.r('theta_oracle'), dtype=float)
    results["truth"] = theta_oracle

    for seed in tqdm(range(50)):
        # print(f"\n--- Experiment {seed} ---")
        np.random.seed(seed)

        # Use processed data shape
        n_samples = X_encoded.shape[0]
        labeled_sample_size = min(5000, int(label_size*n_samples/100))
        labeled_idx = np.random.choice(n_samples, labeled_sample_size, replace=False)
        # results["labeled_idx" + str(seed)] = labeled_idx

        unlabeled_idx = np.array(
            [i for i in range(n_samples) if i not in labeled_idx])
        # print(
        #     f"Unlabeled samples: {unlabeled_idx.shape[0]}, Total samples: {n_samples}"
        # )

        X_train, y_train = X_encoded.iloc[labeled_idx], y.iloc[labeled_idx]
        X_test = X_encoded.iloc[unlabeled_idx]

        #-------------------------------------------
        # TabPFN imputation
        #-------------------------------------------
        classifier = TabPFNRegressor(model_path="./tabpfn-v2-regressor.ckpt")
        classifier.fit(X.iloc[labeled_idx], y.iloc[labeled_idx])
        preds_train = classifier.predict(X.iloc[labeled_idx])

        # Batch prediction
        batch_size = 400
        all_predictions = []
        total_batches = (len(unlabeled_idx) - 1) // batch_size + 1

        for i in range(0, len(unlabeled_idx), batch_size):
            batch_indices = unlabeled_idx[i:i + batch_size]
            batch_X = X.iloc[batch_indices]

            # Predict current batch
            batch_pred = classifier.predict(batch_X)
            all_predictions.append(batch_pred)
            # if i % 50 == 0:
            #     print(f"Completed batch {i//batch_size + 1}/{total_batches}")

        # Combine all prediction results
        preds_test = np.concatenate(all_predictions)

        # Build datasets for different estimators
        df_labeled = combine_xy(X_train, y_train)  # train only
        df_imputed = combine_xy(X_train, y_train, X_test,
                                preds_test)  # test imputed
        df_train_imputed = combine_xy(X_train, preds_train)  # train imputed
        df_all_imputed = combine_xy(X_train, preds_train, X_test,
                                    preds_test)  # all imputed

        # Run R estimations
        try:
            thetas = run_r_estimations(ro,
                                       df_labeled=df_labeled,
                                       df_imputed=df_imputed,
                                       df_train_imputed=df_train_imputed,
                                       df_all_imputed=df_all_imputed,
                                       X_unlabeled=X_test)
    
            results["vanilla_" + str(seed)] = (thetas["supervised"] - results["truth"])**2
            results["TabPFNI_" + str(seed)] = (thetas["imputed"] - results["truth"])**2
            results["TabPFND_" + str(seed)] = (thetas["debiased"]- results["truth"])**2
            results["SLZ_" + str(seed)] = (thetas["SLZ"]- results["truth"])**2
        except:
            continue

        print(np.mean((thetas["supervised"] - results["truth"])**2), np.mean((thetas["imputed"] - results["truth"])**2), np.mean((thetas["debiased"] - results["truth"])**2), np.mean((thetas["SLZ"] - results["truth"])**2))
        
        

        # print(theta_oracle.shape, thetas["supervised"].shape, thetas["imputed"].shape, thetas["debiased"].shape, thetas["SLZ"].shape)

    # for key, val in results.items():
    #     if key != "truth":
    #         print(key, np.mean((val - results["truth"])**2))

    # Save results
    save_path = os.path.join('reg', 'n_' + str(label_size))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, dataset + '.pickle'), "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    # Configuration
    # Open R console and type "R.home()", copy the output in the next line
    R_HOME=r"/root/miniconda3/envs/FedDRM/lib/R"
    # Make sure these two files are in the same folder
    R_SCRIPT1 = "./semi_supervised_methods.R"
    R_SCRIPT2 = "./SupervisedEstimation.R"

    # comparison('weather_izmir')
    # comparison('cmc')
    # comparison('BNG_stock')
    # comparison('puma8NH')
    # comparison('house_8L')
    comparison('kin8nm')
