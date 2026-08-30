"""
ml_regression_tasks_dataset3_scratch.py

Dataset3 (NYC Taxi Trip Duration) regression energy measurement using
from-scratch (non scikit-learn) Python implementations. Structurally
identical to ../d3_cls/ml_regression_tasks_dataset3.py (same DATA_PATH file,
same 1,458,644-row dataset, same training_features/target -- including
dropping dropoff_datetime as target leakage, same as the scikit-learn
script -- same RANDOM_STATE/60-20-20 split, same tags/output-CSV shape) and
to ml_regression_tasks_dataset1_scratch.py (same algorithm implementations,
same SCRATCH_TRAIN_CAP/SCRATCH_PRED_CAP/SCRATCH_MAX_DEPTH constants -- see
FIXES_SCRATCH.md, which is shared across all six scratch scripts).

At this scale SCRATCH_TRAIN_CAP/SCRATCH_PRED_CAP matter far more than at
Dataset1/2: the natural 60% training split here is ~875,000 rows, and every
pure-Python-loop algorithm (Decision Tree, Gaussian, SVM) would be
computationally intractable on that directly -- see FIXES_SCRATCH.md Section 2
for the benchmarks the cap is based on. Linear Regression and Neural Network
are unaffected and still train on the full split, exactly as at Dataset1/2.
"""

import pandas as pd
import numpy as np
import time
import random

from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from mlfromscratch.supervised_learning.regression import LinearRegression as ScratchLinearRegression
from mlfromscratch.supervised_learning.decision_tree import RegressionTree
from mlfromscratch.deep_learning import NeuralNetwork
from mlfromscratch.deep_learning.layers import Dense, Activation
from mlfromscratch.deep_learning.loss_functions import SquareLoss
from mlfromscratch.deep_learning.optimizers import Adam

from scratch_extras import ScratchGaussianProcessRegressor, ScratchSVR

# ---------------------------------------------------------------------------
# Configuration. See ml_regression_tasks_dataset1_scratch.py for the rationale
# behind every constant below -- identical across all six scratch scripts.
# ---------------------------------------------------------------------------

DATA_PATH = "/home/ug/RvsPython/ver/RvsPython/D3.csv"
OUTPUT_CSV = "output_ml_regression_taxi_scratch.csv"
RANDOM_STATE = 42
N_REPETITIONS = 10
SLEEP_SECONDS = 30

SCRATCH_TRAIN_CAP = 500
SCRATCH_PRED_CAP = 500
SCRATCH_MAX_DEPTH = 6

NN_HIDDEN_UNITS = 32
NN_EPOCHS = 20
NN_BATCH_SIZE = 256

csv_handler = CSVHandler(OUTPUT_CSV)


def sleep():
    time.sleep(SLEEP_SECONDS)


# ---------------------------------------------------------------------------
# Data preparation -- byte-for-byte the same as
# ../d3_cls/ml_regression_tasks_dataset3.py.
# ---------------------------------------------------------------------------

dataframe = pd.read_csv(DATA_PATH)
dataframe.replace("?", np.nan, inplace=True)

le_pickup_datetime = LabelEncoder()
le_store_and_fwd_flag = LabelEncoder()
dataframe['pickup_datetime_n'] = le_pickup_datetime.fit_transform(dataframe['pickup_datetime'])
dataframe['store_and_fwd_flag_n'] = le_store_and_fwd_flag.fit_transform(dataframe['store_and_fwd_flag'])

training_features = ['vendor_id', 'pickup_datetime_n', 'store_and_fwd_flag_n',
                      'passenger_count', 'pickup_longitude', 'pickup_latitude',
                      'dropoff_longitude', 'dropoff_latitude']
target = ['trip_duration']

dataframe = dataframe.dropna(subset=training_features + target)

X, X_test, Y, Y_test = train_test_split(dataframe[training_features],
                                         dataframe[target],
                                         test_size=0.2,
                                         random_state=RANDOM_STATE)

X_train, X_pred, Y_train, Y_pred = train_test_split(X, Y,
                                                      test_size=0.25,
                                                      random_state=RANDOM_STATE)

sc_X = StandardScaler()
sc_Y = StandardScaler()
X1 = sc_X.fit_transform(X_train)
Y1 = sc_Y.fit_transform(Y_train)
X_pred_s = sc_X.transform(X_pred)
X_test_s = sc_X.transform(X_test)
n_features = X1.shape[1]


def cap_rows(*arrays, cap, seed=RANDOM_STATE):
    n = len(arrays[0])
    if n <= cap:
        return arrays
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, cap, replace=False)
    return tuple(a[idx] for a in arrays)


X1_scap, Y1_scap = cap_rows(X1, Y1, cap=SCRATCH_TRAIN_CAP)
if len(X1_scap) < len(X1):
    print(f"[note] scratch tree/gaussian/svm trained on {len(X1_scap)} of {len(X1)} rows")

X_pred_scap, = cap_rows(X_pred_s, cap=SCRATCH_PRED_CAP)
X_test_scap, Y_test_scap = cap_rows(X_test_s, Y_test.values, cap=SCRATCH_PRED_CAP,
                                     seed=RANDOM_STATE + 1)
if len(X_test_scap) < len(X_test_s):
    print(f"[note] scratch tree/gaussian/svm scored on {len(X_test_scap)} of {len(X_test_s)} test rows")


def report(name, model, capped=False):
    print(f"The below details are for {name}..")
    X_eval = X_test_scap if capped else X_test_s
    Y_eval = Y_test_scap if capped else Y_test.values
    predicted = np.asarray(model.predict(X_eval), dtype=float)
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
    model = ScratchLinearRegression(gradient_descent=False)
    model.fit(X1, Y1.ravel())
    return model


@measure_energy(handler=csv_handler)
def test_linear_regression():
    linear_regression()


linear_regression_model = linear_regression()


@measure_energy(handler=csv_handler)
def test_linear_regression_inference():
    return linear_regression_model.predict(X_pred_s)


# ---------------------------------------------------------------------------
# Gaussian Regression -- scratch_extras.ScratchGaussianProcessRegressor; see
# FIXES_SCRATCH.md Section 1.
# ---------------------------------------------------------------------------

def gaussian_regression():
    return ScratchGaussianProcessRegressor().fit(X1_scap, Y1_scap.ravel())


@measure_energy(handler=csv_handler)
def test_gaussian_regression():
    gaussian_regression()


gaussian_regression_model = gaussian_regression()


@measure_energy(handler=csv_handler)
def test_gaussian_regression_inference():
    return gaussian_regression_model.predict(X_pred_scap)


# ---------------------------------------------------------------------------
# Decision Tree Regression
# ---------------------------------------------------------------------------

def decision_tree_regression():
    model = RegressionTree(max_depth=SCRATCH_MAX_DEPTH)
    model.fit(X1_scap, Y1_scap.ravel())
    return model


@measure_energy(handler=csv_handler)
def test_decision_tree_regression():
    decision_tree_regression()


decision_tree_regression_model = decision_tree_regression()


@measure_energy(handler=csv_handler)
def test_decision_tree_regression_inference():
    return np.asarray(decision_tree_regression_model.predict(X_pred_scap))


# ---------------------------------------------------------------------------
# Support Vector Regression -- scratch_extras.ScratchSVR; see
# FIXES_SCRATCH.md Section 1.
# ---------------------------------------------------------------------------

def support_vector_regression():
    return ScratchSVR().fit(X1_scap, Y1_scap.ravel())


@measure_energy(handler=csv_handler)
def test_support_vector_regression():
    support_vector_regression()


svr_regression_model = support_vector_regression()


@measure_energy(handler=csv_handler)
def test_support_vector_regression_inference():
    return svr_regression_model.predict(X_pred_scap)


# ---------------------------------------------------------------------------
# Neural Network Regression
# ---------------------------------------------------------------------------

def build_neural_network():
    net = NeuralNetwork(optimizer=Adam(), loss=SquareLoss)
    net.add(Dense(NN_HIDDEN_UNITS, input_shape=(n_features,)))
    net.add(Activation('relu'))
    net.add(Dense(1))
    return net


def neural_network_regression():
    net = build_neural_network()
    net.fit(X1, Y1, n_epochs=NN_EPOCHS, batch_size=NN_BATCH_SIZE)
    return net


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
report("Gaussian Regression", gaussian_regression_model, capped=True)
report("Decision tree Regression", decision_tree_regression_model, capped=True)
report("Support vector Regression", svr_regression_model, capped=True)
report("Neural Network Regression", neural_network_model)

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
