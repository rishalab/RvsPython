"""
ml_regression_tasks_dataset3.py

Dataset3 (NYC Taxi Trip Duration) regression energy measurement.
Built on the structure of ml_regression_tasks_dataset1.py and
ml_regression_tasks_dataset2.py.

Deviations from the Dataset1 / Dataset2 template are marked [DEV-n] inline
and listed at the bottom of this file. They exist because the template as
written does not produce what Section 5 of the paper describes. If Dataset3
is run with these corrections, Dataset1 and Dataset2 must be rerun with them
too, otherwise the three scales are not comparable.
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
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = "/home/ug/RvsPython/ver/RvsPython/D3.csv"
OUTPUT_CSV = "output_ml_regression_taxi.csv"      # [DEV-5]
RANDOM_STATE = 42                                  # [DEV-4]
N_REPETITIONS = 10
SLEEP_SECONDS = 30

# [DECISION-A] Kernel methods do not scale to Dataset3.
# GaussianProcessRegressor holds an n x n kernel matrix. At 60 percent of
# 1,458,644 rows that is roughly 5.7 TB against 128 GB of RAM. SVR is
# O(n^2) in kernel evaluations and would run for days.
# Set KERNEL_TRAIN_CAP to an integer to subsample the training set for
# Gaussian and SVR only, or to None to attempt the full set (will not
# complete). Whatever is chosen must be stated in Section 5 and applied
# identically at Dataset1 and Dataset2 for the scaling analysis to hold.
KERNEL_TRAIN_CAP = 20000

# [FIX-OOM] Capping training alone is not enough. GaussianProcessRegressor.predict
# and SVR.predict build a (n_query x n_train_kernel) kernel evaluation against
# every query row. Scoring or measuring inference against the full ~20%
# held-out slice here (roughly 290,000 rows) against a 20,000-row kernel
# training set means GaussianProcessRegressor.predict alone materialises a
# dense ~5.8e9-cell (~46 GB) matrix with no chunking -- this is what was
# crashing this script with an OOM kill. The same cap value is reused here,
# applied to the *query* side for kernel models only, and must be applied
# identically at Dataset1 and Dataset2 (also done, see those two scripts).
KERNEL_PRED_CAP = 20000

csv_handler = CSVHandler(OUTPUT_CSV)


def sleep():
    time.sleep(SLEEP_SECONDS)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
# Feature selection follows the rule recovered from the Dataset1 and Dataset2
# scripts: every column except the identifier and the target, LabelEncoder on
# every non-numeric column, numeric columns passed through unchanged, no
# feature engineering, and one feature set shared by regression and
# classification. pickup_datetime is encoded whole rather than decomposed,
# mirroring the drug review date_n treatment.
#
# [DECISION-B] One departure. dropoff_datetime minus pickup_datetime equals
# trip_duration exactly, so the mechanical rule would leak the target. It is
# dropped. This must be stated in Section 5.

dataframe = pd.read_csv(DATA_PATH)
dataframe.replace("?", np.nan, inplace=True)

le_pickup_datetime = LabelEncoder()
le_store_and_fwd_flag = LabelEncoder()
dataframe['pickup_datetime_n'] = le_pickup_datetime.fit_transform(
    dataframe['pickup_datetime'])
dataframe['store_and_fwd_flag_n'] = le_store_and_fwd_flag.fit_transform(
    dataframe['store_and_fwd_flag'])

# vendor_id and passenger_count are already numeric, so they pass through
# unencoded, as age and usefulCount do at the earlier scales.
training_features = ['vendor_id', 'pickup_datetime_n', 'store_and_fwd_flag_n',
                     'passenger_count', 'pickup_longitude', 'pickup_latitude',
                     'dropoff_longitude', 'dropoff_latitude']
target = ['trip_duration']

# No outlier removal, matching Dataset1 and Dataset2, which filtered nothing.
dataframe = dataframe.dropna(subset=training_features + target)

# 60 / 20 / 20 train / inference / test, as in Dataset1 and Dataset2.
X, X_test, Y, Y_test = train_test_split(dataframe[training_features],
                                        dataframe[target],
                                        test_size=0.2,
                                        random_state=RANDOM_STATE)

X_train, X_pred, Y_train, Y_pred = train_test_split(X, Y,
                                                    test_size=0.25,
                                                    random_state=RANDOM_STATE)

# [DEV-1] All five algorithms fit on the same standardised matrix. The
# Dataset1 template fit Linear, Gaussian and Decision Tree on raw X_train
# while SVR and MLP used the scaled X1, and Dataset2 moved Gaussian onto X1.
# That inconsistency makes the cross-scale Gaussian comparison invalid.
# The scaled target is NOT cast to int. The Dataset2 regression script applied
# Y1 = Y1.astype(int) after scaling, so Gaussian, SVR and MLP there regressed
# against a truncated integer target while Dataset1 used the continuous one.
sc_X = StandardScaler()
sc_Y = StandardScaler()
X1 = sc_X.fit_transform(X_train)
Y1 = sc_Y.fit_transform(Y_train)
X_pred_s = sc_X.transform(X_pred)
X_test_s = sc_X.transform(X_test)

# Reduced training matrix for the kernel methods only.
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
    # [FIX-OOM] kernel=True scores against the capped query set (see
    # KERNEL_PRED_CAP above) instead of the full test set.
    print(f"The below details are for {name}..")
    X_eval = X_test_kernel_s if kernel else X_test_s
    Y_eval = Y_test_kernel if kernel else Y_test
    predicted = model.predict(X_eval)
    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)
    predicted = sc_Y.inverse_transform(predicted)
    r2 = r2_score(Y_eval, predicted)
    mae = mean_absolute_error(Y_eval, predicted)
    mse = mean_squared_error(Y_eval, predicted)
    print("r2 value= ", r2)
    print("MAE value= ", mae)
    print("MSE value= ", mse)
    print("RMSE value= ", np.sqrt(mse))


# ---------------------------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------------------------

def linear_regression():
    linear_regression_model = LinearRegression()
    return linear_regression_model.fit(X1, Y1)


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
    gaussian_regression_model = GaussianProcessRegressor()
    return gaussian_regression_model.fit(X1_kernel, Y1_kernel.ravel())


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
    decision_tree_regression_model = DecisionTreeRegressor(random_state=RANDOM_STATE)
    return decision_tree_regression_model.fit(X1, Y1)


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
    svr_regression_model = SVR()
    return svr_regression_model.fit(X1_kernel, Y1_kernel.ravel())


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
    neural_network_model = MLPRegressor(random_state=RANDOM_STATE)
    return neural_network_model.fit(X1, Y1.ravel())


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
report("support vector Regression", svr_regression_model, kernel=True)
report("Neural Network Regression", neural_network_model)

# [DEV-2] One shuffled block only, each function present exactly once, so
# every task gets exactly N_REPETITIONS measurements. The Dataset1 template
# listed linear inference twice (n=20) and the Dataset2 template ran five
# extra pre-loops, giving counts of 10, 20, 21, 22 and 30 against a stated
# protocol of 10.
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
# Deviations from the Dataset1 / Dataset2 template
#
# [DEV-1] Uniform standardisation across all five algorithms.
# [DEV-2] Exactly N_REPETITIONS measurements per task, no duplicated entries
#         in the shuffled list and no pre-loops.
# [DEV-3] Neural network inference is included in the shuffled list. The
#         Dataset2 template omitted it, giving n=10 against n=20 elsewhere.
# [DEV-4] random_state fixed on splits, tree and MLP.
# [DEV-5] Distinct output filename. Both existing classification scripts wrote
#         to output_ml_classification_drug.csv, including the one reading
#         adult.csv.
# [DEV-6] The unused adult_infer1.csv style second read is dropped. The
#         Dataset1 script loaded a file it never used, which blocks any rerun
#         when that derived file is missing.
# ---------------------------------------------------------------------------
