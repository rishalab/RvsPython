"""
ml_regression_tasks_dataset2_rerun.py

Dataset2 (UCI Drug Review) regression energy measurement for the R vs Python IST revision.

Corrected rerun. Structurally identical to ml_regression_tasks_dataset3.py so that
all three scales are measured under the same protocol. Every departure from the
original script is marked [DEV-n] inline and listed at the end of this file.

Do not edit the protocol constants without telling Ch. They are shared across
all six scripts and changing one breaks the cross-scale comparison.
"""

import pandas as pd
import numpy as np
import time
import random

from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------------
# Configuration. Identical across all six scripts.
# ---------------------------------------------------------------------------

# [FIX-DATA] The original path is another account's home directory and does
# not exist here. Fixed to the actual UCI Drug Review release (train+test,
# tab-separated) -- 161,297 + 53,766 = 215,063 rows, an exact match to the
# paper's Section 5 figure. See ml_classification_tasks_dataset2_rerun.py and
# FIXES.md Section 1 for how this was obtained.
DATA_PATH_TRAIN = "/home/ug/RvsPython/ver/drugsComTrain_raw.tsv"
DATA_PATH_TEST = "/home/ug/RvsPython/ver/drugsComTest_raw.tsv"
OUTPUT_CSV = "output_ml_regression_drug_rerun.csv"
RANDOM_STATE = 42
N_REPETITIONS = 10
SLEEP_SECONDS = 30

# Kernel methods only. Fixed at 20000 because that is the value Dataset3 was
# run at, and the cap has to be identical at every scale for the scaling
# analysis to mean anything. State it in Section 5.
KERNEL_TRAIN_CAP = 20000

# [FIX-OOM] Capping training alone is not enough. GaussianProcessRegressor and
# SVR predict by building a dense (n_query x n_train_kernel) kernel matrix
# with no chunking. At Dataset3 scale, scoring or measuring inference against
# the full ~20% held-out slice against a 20,000-row kernel training set is
# tens of GB, and this is what was crashing the Dataset3 regression run with
# an OOM kill. The same cap value is reused here, applied to the *query* side
# for kernel models only, and applied identically across all three datasets.
KERNEL_PRED_CAP = 20000

csv_handler = CSVHandler(OUTPUT_CSV)


def sleep():
    time.sleep(SLEEP_SECONDS)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
# Column selection is unchanged from the original script. Missing values are
# left as an encoded category rather than dropped, also as in the original, so
# that the reported row count for this dataset stays correct.

dataframe = pd.concat(
    [pd.read_csv(DATA_PATH_TRAIN, sep="\t"), pd.read_csv(DATA_PATH_TEST, sep="\t")],
    ignore_index=True,
)
dataframe.replace("?", np.nan, inplace=True)

le_drugName = LabelEncoder()
le_condition = LabelEncoder()
le_review = LabelEncoder()
le_date = LabelEncoder()
le_rating = LabelEncoder()
dataframe['drugName_n'] = le_drugName.fit_transform(dataframe['drugName'])
dataframe['condition_n'] = le_condition.fit_transform(dataframe['condition'])
dataframe['review_n'] = le_review.fit_transform(dataframe['review'])
dataframe['date_n'] = le_date.fit_transform(dataframe['date'])
dataframe['rating_n'] = le_rating.fit_transform(dataframe['rating'])

# [DEV-1] The duplicate entry in the original feature list is removed. It named
# the same column twice, which fed the identical vector to the model twice.
training_features = ['drugName_n', 'condition_n', 'review_n', 'date_n',
                     'usefulCount']
target = ['rating_n']

X, X_test, Y, Y_test = train_test_split(dataframe[training_features],
                                        dataframe[target],
                                        test_size=0.2,
                                        random_state=RANDOM_STATE)

X_train, X_pred, Y_train, Y_pred = train_test_split(X, Y,
                                                    test_size=0.25,
                                                    random_state=RANDOM_STATE)

# [DEV-2] All five algorithms fit on the same standardised matrix. The original
# scripts fit Linear, Gaussian and Decision Tree on raw X_train while SVR and
# MLP used the scaled X1, and Dataset2 moved Gaussian onto X1, so the Gaussian
# comparison across scales was not like for like.
#
# [DEV-3] The scaled target is not cast to int. The Dataset2 regression script
# applied Y1 = Y1.astype(int) after scaling, truncating the target for Gaussian,
# SVR and MLP.
sc_X = StandardScaler()
sc_Y = StandardScaler()
X1 = sc_X.fit_transform(X_train)
Y1 = sc_Y.fit_transform(Y_train)
X_pred_s = sc_X.transform(X_pred)
X_test_s = sc_X.transform(X_test)

if KERNEL_TRAIN_CAP is not None and len(X1) > KERNEL_TRAIN_CAP:
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X1), KERNEL_TRAIN_CAP, replace=False)
    X1_kernel, Y1_kernel = X1[idx], Y1[idx]
    print(f"[note] kernel methods trained on {KERNEL_TRAIN_CAP} of {len(X1)} rows")
else:
    X1_kernel, Y1_kernel = X1, Y1


def cap_for_kernel(X, cap=KERNEL_PRED_CAP, seed=RANDOM_STATE):
    if cap is not None and len(X) > cap:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), cap, replace=False)
        return X[idx], idx
    return X, np.arange(len(X))


# [FIX-OOM] Capped query-side inputs for the kernel models (Gaussian, SVR)
# only. Linear, Decision Tree and MLP keep predicting against the full
# X_pred_s / X_test_s.
X_pred_kernel_s, _pred_kernel_idx = cap_for_kernel(X_pred_s)
X_test_kernel_s, _test_kernel_idx = cap_for_kernel(X_test_s)
Y_test_kernel = Y_test.values[_test_kernel_idx]
if len(X_test_kernel_s) < len(X_test_s):
    print(f"[note] kernel methods scored on {len(X_test_kernel_s)} of {len(X_test_s)} test rows")


def report(name, model, kernel=False):
    # [DEV-4] Every model is scored on the same standardised test matrix. The
    # original scored some models on raw X_test after fitting on scaled data.
    #
    # [FIX-OOM] kernel=True scores against the capped query set (see
    # KERNEL_PRED_CAP above) instead of the full test set.
    print(f"The below details are for {name}..")
    X_eval = X_test_kernel_s if kernel else X_test_s
    Y_eval = Y_test_kernel if kernel else Y_test
    predicted = model.predict(X_eval)
    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)
    predicted = sc_Y.inverse_transform(predicted)
    mse = mean_squared_error(Y_eval, predicted)
    print("r2 value= ", r2_score(Y_eval, predicted))
    print("MAE value= ", mean_absolute_error(Y_eval, predicted))
    print("MSE value= ", mse)
    print("RMSE value= ", np.sqrt(mse))


# ---------------------------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------------------------

def linear_regression():
    return LinearRegression().fit(X1, Y1)


@measure_energy(handler=csv_handler)
def test_linear_regression():
    linear_regression()


linear_regression_model = linear_regression()


@measure_energy(handler=csv_handler)
def test_linear_regression_inference():
    return linear_regression_model.predict(X_pred_s)


# ---------------------------------------------------------------------------
# Gaussian Process Regression
# ---------------------------------------------------------------------------

def gaussian_regression():
    return GaussianProcessRegressor().fit(X1_kernel, Y1_kernel.ravel())


@measure_energy(handler=csv_handler)
def test_gaussian_regression():
    gaussian_regression()


gaussian_regression_model = gaussian_regression()


@measure_energy(handler=csv_handler)
def test_gaussian_regression_inference():
    # [FIX-OOM] Capped query set; see KERNEL_PRED_CAP above.
    return gaussian_regression_model.predict(X_pred_kernel_s)


# ---------------------------------------------------------------------------
# Decision Tree Regression
# ---------------------------------------------------------------------------

def decision_tree_regression():
    return DecisionTreeRegressor(random_state=RANDOM_STATE).fit(X1, Y1)


@measure_energy(handler=csv_handler)
def test_decision_tree_regression():
    decision_tree_regression()


decision_tree_regression_model = decision_tree_regression()


@measure_energy(handler=csv_handler)
def test_decision_tree_regression_inference():
    return decision_tree_regression_model.predict(X_pred_s)


# ---------------------------------------------------------------------------
# Support Vector Regression
# ---------------------------------------------------------------------------

def support_vector_regression():
    return SVR().fit(X1_kernel, Y1_kernel.ravel())


@measure_energy(handler=csv_handler)
def test_support_vector_regression():
    support_vector_regression()


svr_regression_model = support_vector_regression()


@measure_energy(handler=csv_handler)
def test_support_vector_regression_inference():
    # [FIX-OOM] Capped query set; see KERNEL_PRED_CAP above.
    return svr_regression_model.predict(X_pred_kernel_s)


# ---------------------------------------------------------------------------
# Neural Network Regression
# ---------------------------------------------------------------------------

def neural_network_regression():
    return MLPRegressor(random_state=RANDOM_STATE).fit(X1, Y1.ravel())


@measure_energy(handler=csv_handler)
def test_neural_network_regression():
    neural_network_regression()


neural_network_model = neural_network_regression()


@measure_energy(handler=csv_handler)
def test_neural_network_regression_inference():
    return neural_network_model.predict(X_pred_s)


# ---------------------------------------------------------------------------
# Accuracy reporting, then the measurement loop
# ---------------------------------------------------------------------------

report("Linear Regression", linear_regression_model)
report("Gaussian Regression", gaussian_regression_model, kernel=True)
report("Decision tree Regression", decision_tree_regression_model)
report("Support vector Regression", svr_regression_model, kernel=True)
# [DEV-5] The neural network is scored with the neural network model. The
# Dataset1 script called svr_regression_model here, so the MLP accuracy
# reported for Dataset1 was in fact SVR's.
report("Neural Network Regression", neural_network_model)

# [DEV-6] One shuffled block, each function present exactly once, so every task
# gets exactly N_REPETITIONS measurements. The originals produced counts of 10,
# 11, 20, 21, 22 and 30 against a stated protocol of ten, through duplicated
# list entries and extra pre-loops.
function_list = [
    test_linear_regression,
    test_gaussian_regression,
    test_decision_tree_regression,
    test_support_vector_regression,
    test_neural_network_regression,
    test_linear_regression_inference,
    test_gaussian_regression_inference,
    test_decision_tree_regression_inference,
    test_support_vector_regression_inference,
    test_neural_network_regression_inference,
]

for i in range(N_REPETITIONS):
    print("This is iteration no:", i)
    random.shuffle(function_list)
    for j in range(len(function_list)):
        sleep()
        function_list[j]()

print("Process complete")
csv_handler.save_data()

# ---------------------------------------------------------------------------
# [DEV-8] random_state is fixed on the splits, the trees, the MLP and the
#         kernel subsample, so the run is reproducible.
# [DEV-9] The output filename is distinct. Both original classification scripts
#         wrote to output_ml_classification_drug.csv, including the one reading
#         adult.csv.
# [DEV-10] The unused second read_csv of a derived inference file is dropped.
#          The Dataset1 regression script loaded adult_infer1.csv and never used
#          it, which blocks any rerun when that file is missing.
# ---------------------------------------------------------------------------
