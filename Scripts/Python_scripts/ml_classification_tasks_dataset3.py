"""
ml_classification_tasks_dataset3.py

Dataset3 (NYC Taxi Trip Duration) classification energy measurement.
Built on the structure of ml_classification_tasks_dataset1.py and
ml_classification_tasks_dataset2.py.

Deviations from the template are marked [DEV-n] inline and listed at the
bottom. [DEV-1] is the important one: in both existing classification scripts
the function named decision_tree_classification instantiates SVC(), so every
published Decision Tree classification figure for Dataset1 and Dataset2 is a
second SVM run. That is corrected here, which means Dataset1 and Dataset2
classification must be rerun before the three scales can be compared.
"""

import pandas as pd
import numpy as np
import time
import random

from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = "/home/ug/RvsPython/ver/RvsPython/D3.csv"
OUTPUT_CSV = "output_ml_classification_taxi.csv"   # [DEV-5]
RANDOM_STATE = 42
N_REPETITIONS = 10
SLEEP_SECONDS = 30

# [DECISION-A] SVC is O(n^2) in kernel evaluations and will not complete on
# roughly 875,000 training rows.
# [FIX-CONSISTENCY] This was left at None here, which is inconsistent with the
# regression script and with FULL_RERUN_INSTRUCTIONS.md Section 2, both of
# which fix KERNEL_TRAIN_CAP = 20000 across all six scripts. An uncapped SVC
# fit here would also never finish, since the classification target has two
# balanced classes on ~875,000 rows. Set to None only to attempt the full set,
# and if so it must also be changed identically at Dataset1 and Dataset2.
KERNEL_TRAIN_CAP = 20000

# [FIX-OOM] Capping training alone is not enough. sklearn's kernel methods
# (SVC here; GaussianProcessRegressor and SVR in the regression scripts)
# predict by building a dense (n_query x n_train_kernel) kernel matrix with no
# chunking. At Dataset3 scale, scoring or measuring inference against the full
# ~20% held-out slice (roughly 290,000 rows) against a 20,000-row kernel
# training set is a ~5.8e9-cell matrix -- tens of GB, and this is what was
# crashing the Dataset3 regression run with an OOM kill. The same cap value is
# reused here, applied to the *query* side for kernel models only, and applied
# identically across all three datasets so no scale gets a size advantage.
KERNEL_PRED_CAP = 20000

csv_handler = CSVHandler(OUTPUT_CSV)


def sleep():
    time.sleep(SLEEP_SECONDS)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
# Feature set is identical to the Dataset3 regression script, following the
# rule recovered from Dataset1 and Dataset2, where both paradigms shared one
# feature list. dropoff_datetime is dropped as target leakage; see the
# regression script.
#
# [DECISION-B] The target is where the template cannot be inherited. Dataset1
# and Dataset2 reused the same target column for both paradigms, relying on
# StandardScaler followed by astype(int) to produce classes. That truncation
# yields two classes on Adult income and, less obviously, two classes on the
# ten-level drug rating with a threshold near rating 3.7 that is an artefact
# rather than a design choice. Applied to trip_duration, which is heavily
# right skewed, it would place almost every row in a single class.
# The class is therefore defined explicitly as a median split on
# trip_duration, giving a balanced binary task that matches the effective
# structure of both earlier scales. This must be stated in Section 5.

dataframe = pd.read_csv(DATA_PATH)
dataframe.replace("?", np.nan, inplace=True)

le_pickup_datetime = LabelEncoder()
le_store_and_fwd_flag = LabelEncoder()
dataframe['pickup_datetime_n'] = le_pickup_datetime.fit_transform(
    dataframe['pickup_datetime'])
dataframe['store_and_fwd_flag_n'] = le_store_and_fwd_flag.fit_transform(
    dataframe['store_and_fwd_flag'])

training_features = ['vendor_id', 'pickup_datetime_n', 'store_and_fwd_flag_n',
                     'passenger_count', 'pickup_longitude', 'pickup_latitude',
                     'dropoff_longitude', 'dropoff_latitude']

dataframe = dataframe.dropna(subset=training_features + ['trip_duration'])
median_duration = dataframe['trip_duration'].median()
dataframe['long_trip'] = (dataframe['trip_duration'] >= median_duration).astype(int)
target = ['long_trip']
print(f"[note] median trip_duration = {median_duration} s; "
      f"class balance = {dataframe['long_trip'].mean():.3f}")

X, X_test, Y, Y_test = train_test_split(dataframe[training_features],
                                        dataframe[target],
                                        test_size=0.2,
                                        random_state=RANDOM_STATE)

X_train, X_pred, Y_train, Y_pred = train_test_split(X, Y,
                                                    test_size=0.25,
                                                    random_state=RANDOM_STATE)

# [DEV-2] Features are standardised; the class label is not. Both existing
# classification scripts passed the label through StandardScaler and then cast
# to int, which on the ten-level drug rating collapsed the classes into
# truncated buckets rather than classifying the rating.
sc_X = StandardScaler()
X1 = sc_X.fit_transform(X_train)
X_pred_s = sc_X.transform(X_pred)
X_test_s = sc_X.transform(X_test)
Y1 = Y_train.values.ravel()

if KERNEL_TRAIN_CAP is not None and len(X1) > KERNEL_TRAIN_CAP:
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X1), KERNEL_TRAIN_CAP, replace=False)
    X1_kernel, Y1_kernel = X1[idx], Y1[idx]
    print(f"[note] SVM trained on {KERNEL_TRAIN_CAP} of {len(X1)} rows")
else:
    X1_kernel, Y1_kernel = X1, Y1


def cap_for_kernel(X, cap=KERNEL_PRED_CAP, seed=RANDOM_STATE):
    if cap is not None and len(X) > cap:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), cap, replace=False)
        return X[idx], idx
    return X, np.arange(len(X))


# [FIX-OOM] Capped query-side inputs for the kernel model (SVC) only. Every
# other classifier keeps predicting against the full X_pred_s / X_test_s.
X_pred_kernel_s, _pred_kernel_idx = cap_for_kernel(X_pred_s)
X_test_kernel_s, _test_kernel_idx = cap_for_kernel(X_test_s)
Y_test_kernel = Y_test.values.ravel()[_test_kernel_idx]
if len(X_test_kernel_s) < len(X_test_s):
    print(f"[note] SVM scored on {len(X_test_kernel_s)} of {len(X_test_s)} test rows")


def report(name, model, kernel=False):
    # [DEV-3] Metrics computed with sklearn rather than by unpacking a
    # confusion matrix into tn, fp, fn, tp. That unpacking assumes a binary
    # problem and raised on the multiclass drug target; one script also used
    # `tn, fp, fn, tp = cm` without .ravel().
    #
    # [FIX-OOM] kernel=True scores against the capped query set (see
    # KERNEL_PRED_CAP above) instead of the full test set.
    print(f"The below details are for {name}..")
    X_eval = X_test_kernel_s if kernel else X_test_s
    Y_eval = Y_test_kernel if kernel else Y_test
    predicted = model.predict(X_eval)
    print("Accuracy= ", accuracy_score(Y_eval, predicted))
    print("Recall= ", recall_score(Y_eval, predicted, average='weighted', zero_division=0))
    print("Precision= ", precision_score(Y_eval, predicted, average='weighted', zero_division=0))
    print("f1 score= ", f1_score(Y_eval, predicted, average='weighted', zero_division=0))


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def random_forest_classification():
    random_forest_model = RandomForestClassifier(random_state=RANDOM_STATE)
    return random_forest_model.fit(X1, Y1)


@measure_energy(handler=csv_handler)
def test_random_forest_classification():
    random_forest_classification()


random_forest_model = random_forest_classification()


@measure_energy(handler=csv_handler)
def test_random_forest_classification_inference():
    return random_forest_model.predict(X_pred_s)


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------
# [DEV-4] The return statement is present. In the Dataset2 script it was
# commented out, so the function returned None, the training measurement
# captured only object instantiation, and the model object was unusable.

def logistic_regression_classification():
    logistic_regression_model = LogisticRegression(max_iter=1000)
    return logistic_regression_model.fit(X1, Y1)


@measure_energy(handler=csv_handler)
def test_logistic_regression_classification():
    logistic_regression_classification()


logistic_regression_model = logistic_regression_classification()


@measure_energy(handler=csv_handler)
def test_logistic_regression_classification_inference():
    return logistic_regression_model.predict(X_pred_s)


# ---------------------------------------------------------------------------
# Naive Bayes
# ---------------------------------------------------------------------------

def gaussian_NB_classification():
    naive_bayes_model = GaussianNB()
    return naive_bayes_model.fit(X1, Y1)


@measure_energy(handler=csv_handler)
def test_gaussian_NB_classification():
    gaussian_NB_classification()


naive_bayes_model = gaussian_NB_classification()


@measure_energy(handler=csv_handler)
def test_gaussian_NB_classification_inference():
    return naive_bayes_model.predict(X_pred_s)


# ---------------------------------------------------------------------------
# Support Vector Machine
# ---------------------------------------------------------------------------

def SVM_classification():
    svm_classifier_model = SVC()
    return svm_classifier_model.fit(X1_kernel, Y1_kernel)


@measure_energy(handler=csv_handler)
def test_SVM_classification():
    SVM_classification()


svm_classifier_model = SVM_classification()


@measure_energy(handler=csv_handler)
def test_SVM_classification_inference():
    # [FIX-OOM] Capped query set; see KERNEL_PRED_CAP above.
    return svm_classifier_model.predict(X_pred_kernel_s)


# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------
# [DEV-1] DecisionTreeClassifier, not SVC.

def decision_tree_classification():
    decision_tree_classifier_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    return decision_tree_classifier_model.fit(X1, Y1)


@measure_energy(handler=csv_handler)
def test_decision_tree_classification():
    decision_tree_classification()


decision_tree_classifier_model = decision_tree_classification()


@measure_energy(handler=csv_handler)
def test_decision_tree_classification_inference():
    return decision_tree_classifier_model.predict(X_pred_s)


# ---------------------------------------------------------------------------
# Accuracy reporting, then the measurement loop
# ---------------------------------------------------------------------------

report("Random forest classification", random_forest_model)
report("Logistic Regression classification", logistic_regression_model)
report("Naive Bayes classification", naive_bayes_model)
report("SVM classification", svm_classifier_model, kernel=True)
report("Decision tree classification", decision_tree_classifier_model)

function_list = [
    test_random_forest_classification,
    test_logistic_regression_classification,
    test_gaussian_NB_classification,
    test_SVM_classification,
    test_decision_tree_classification,
    test_random_forest_classification_inference,
    test_logistic_regression_classification_inference,
    test_gaussian_NB_classification_inference,
    test_SVM_classification_inference,
    test_decision_tree_classification_inference,
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
# [DEV-1] Decision tree uses DecisionTreeClassifier, not SVC.
# [DEV-2] The class label is not standardised.
# [DEV-3] Metrics from sklearn rather than manual confusion matrix unpacking.
# [DEV-4] Logistic regression actually fits.
# [DEV-5] Distinct output filename.
# [DEV-6] Exactly N_REPETITIONS per task; no pre-loops, no duplicated entries.
# [DEV-7] random_state fixed throughout.
# ---------------------------------------------------------------------------
