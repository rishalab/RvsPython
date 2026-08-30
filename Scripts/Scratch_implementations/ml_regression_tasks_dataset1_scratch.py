"""
ml_regression_tasks_dataset1_scratch.py

Dataset1 (Adult Census Income) regression energy measurement using from-scratch
(non scikit-learn) Python implementations -- the Section 7.2 / Table 8 scratch
validation, re-run under the same fixed protocol as ../d3_cls.

Structurally identical to ../d3_cls/ml_regression_tasks_dataset1_rerun.py: same
DATA_PATH/DATA_PATH_TEST files, same 48,842-row combined dataset, same
training_features/target, same RANDOM_STATE/60-20-20 split, same tag names, same
output CSV shape (timestamp;tag;duration;package_0;...). The only thing that
changes is which Python implementation of each algorithm is measured -- no
scikit-learn model class is used anywhere below (LabelEncoder, StandardScaler,
train_test_split and the sklearn.metrics scoring functions are preprocessing/
scoring utilities, not models, and the original Table 8/9 methodology text
already used sklearn for exactly these two roles).

Two algorithms have no ready-made equivalent in the vendored mlfromscratch/
package (eriklindernoren/ML-From-Scratch) and are implemented in
scratch_extras.py instead; four algorithms (Decision Tree, Gaussian, SVM, and
classification's Random Forest) also need a SCRATCH_TRAIN_CAP/SCRATCH_PRED_CAP
tractability cap that the scikit-learn scripts don't need. See
FIXES_SCRATCH.md for the full rationale and the benchmarks the cap/max_depth/
n_estimators values below are based on -- do not change them without reading
that file first, since every other one of these six scripts depends on the
same constants for the cross-scale comparison to mean anything.
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
# Configuration. Identical across all six scratch scripts (and matching the
# corresponding ../d3_cls script's DATA_PATH / RANDOM_STATE / N_REPETITIONS /
# SLEEP_SECONDS exactly).
# ---------------------------------------------------------------------------

DATA_PATH = "/home/ug/RvsPython/ver/adult.data"
DATA_PATH_TEST = "/home/ug/RvsPython/ver/adult.test"
OUTPUT_CSV = "output_ml_regression_adult_scratch.csv"
RANDOM_STATE = 42
N_REPETITIONS = 10
SLEEP_SECONDS = 30

# [SCRATCH-CAP] Unlike ../d3_cls (where only the two kernel methods need
# capping), every pure-Python-loop algorithm here is far slower than its
# scikit-learn counterpart at real dataset scale. Decision Tree, Gaussian
# (kernel Cholesky), SVM (cvxopt QP over an n x n kernel matrix) are capped to
# this many training/query rows -- benchmarked in FIXES_SCRATCH.md Section 2 to
# keep a single fit/predict call in the single-digit seconds at worst. Linear
# Regression (closed-form), Logistic Regression / Naive Bayes (classification
# script) and Neural Network are NOT capped -- benchmarked as tractable on the
# full 60% training split even at Dataset3 scale.
SCRATCH_TRAIN_CAP = 500
SCRATCH_PRED_CAP = 500

# [SCRATCH-CAP] From-scratch trees have no complexity pruning; an unbounded
# max_depth on continuous features degenerates towards one sample per leaf and
# is computationally explosive (see FIXES_SCRATCH.md Section 2). Applied to
# every RegressionTree/ClassificationTree/RandomForest below.
SCRATCH_MAX_DEPTH = 6

# Neural Network architecture/training budget -- see FIXES_SCRATCH.md Section
# 4 for why a small hand-built Dense(32)-ReLU-Dense(1) network, not a port of
# sklearn's MLPRegressor, is what "from scratch" means here.
NN_HIDDEN_UNITS = 32
NN_EPOCHS = 20
NN_BATCH_SIZE = 256

csv_handler = CSVHandler(OUTPUT_CSV)


def sleep():
    time.sleep(SLEEP_SECONDS)


# ---------------------------------------------------------------------------
# Data preparation -- byte-for-byte the same as
# ../d3_cls/ml_regression_tasks_dataset1_rerun.py.
# ---------------------------------------------------------------------------

columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education.num',
    'marital.status', 'occupation', 'relationship', 'race', 'sex',
    'capital.gain', 'capital.loss', 'hours.per.week', 'native.country',
    'income',
]

train_df = pd.read_csv(DATA_PATH, names=columns, skipinitialspace=True)
test_df = pd.read_csv(DATA_PATH_TEST, names=columns, skipinitialspace=True,
                       skiprows=1)
test_df['income'] = test_df['income'].str.rstrip('.')
dataframe = pd.concat([train_df, test_df], ignore_index=True)
dataframe.replace("?", np.nan, inplace=True)

le_income = LabelEncoder()
le_sex = LabelEncoder()
le_occupation = LabelEncoder()
le_marital_status = LabelEncoder()
le_workclass = LabelEncoder()
le_education = LabelEncoder()
dataframe['income_n'] = le_income.fit_transform(dataframe['income'])
dataframe['sex_n'] = le_sex.fit_transform(dataframe['sex'])
dataframe['occupation_n'] = le_occupation.fit_transform(dataframe['occupation'])
dataframe['marital.status_n'] = le_marital_status.fit_transform(dataframe['marital.status'])
dataframe['workclass_n'] = le_workclass.fit_transform(dataframe['workclass'])
dataframe['education_n'] = le_education.fit_transform(dataframe['education'])

training_features = ['workclass_n', 'sex_n', 'fnlwgt', 'occupation_n',
                      'marital.status_n', 'education_n', 'education.num',
                      'capital.gain', 'hours.per.week', 'age', 'capital.loss']
target = ['income_n']

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


# [SCRATCH-CAP] Decision Tree, Gaussian and SVM train on this capped subset;
# Linear Regression and Neural Network train on the full X1/Y1.
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
    # [SCRATCH-NOTE] Not every mlfromscratch predict() returns a numpy array
    # (e.g. RegressionTree returns a plain list) -- normalise before use.
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
# Linear Regression -- mlfromscratch's closed-form (SVD/pseudo-inverse) fit,
# the scratch analogue of sklearn's LinearRegression.fit. Uncapped: O(n*d^2),
# cheap even at the full training split (FIXES_SCRATCH.md Section 2).
# ---------------------------------------------------------------------------

def linear_regression():
    # [SCRATCH-NOTE] mlfromscratch's fit() methods mutate self and do not
    # return it (unlike sklearn's .fit() chaining convention) -- construct,
    # fit, then return the object explicitly. Applies to every algorithm
    # below except the two in scratch_extras.py, which do return self.
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
# Gaussian Regression -- ML-From-Scratch has no Gaussian Process. FIXES_SCRATCH.md
# Section 1 documents the judgement call: a from-scratch RBF-kernel GP
# (scratch_extras.ScratchGaussianProcessRegressor), the same algorithm
# GaussianProcessRegressor is, rather than a different model family.
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
# Decision Tree Regression -- mlfromscratch RegressionTree, depth-bounded and
# row-capped; see SCRATCH_MAX_DEPTH / SCRATCH_TRAIN_CAP above.
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
# Support Vector Regression -- ML-From-Scratch's SupportVectorMachine is a
# classifier only (no regression variant). FIXES_SCRATCH.md Section 1
# documents scratch_extras.ScratchSVR: the standard epsilon-insensitive dual
# SVR QP, solved with cvxopt exactly as the repo's own classifier solves its
# dual, reusing the same RBF-kernel mechanism.
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
# Neural Network Regression -- mlfromscratch's own NeuralNetwork (Dense +
# Activation layers, backprop, Adam), not a port of sklearn's MLPRegressor.
# Uncapped: mini-batch training is linear in n (FIXES_SCRATCH.md Section 2).
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
# Accuracy reporting, then the measurement loop -- identical shuffled-block
# structure to every ../d3_cls script: each tag measured exactly once per
# repetition, N_REPETITIONS repetitions, SLEEP_SECONDS idle between calls.
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
