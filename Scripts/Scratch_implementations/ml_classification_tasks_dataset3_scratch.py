"""
ml_classification_tasks_dataset3_scratch.py

Dataset3 (NYC Taxi Trip Duration) classification energy measurement using
from-scratch (non scikit-learn) Python implementations. Structurally
identical to ../d3_cls/ml_classification_tasks_dataset3.py (same DATA_PATH
file, same 1,458,644-row dataset, same training_features, same median-split
"long_trip" binary target, same RANDOM_STATE/60-20-20 split, same tags/
output-CSV shape) and to ml_classification_tasks_dataset1_scratch.py (same
algorithm implementations, same SCRATCH_TRAIN_CAP/SCRATCH_PRED_CAP/
SCRATCH_MAX_DEPTH/RF_N_ESTIMATORS constants -- see FIXES_SCRATCH.md, shared
across all six scratch scripts).

At this scale SCRATCH_TRAIN_CAP/SCRATCH_PRED_CAP matter far more than at
Dataset1/2: the natural 60% training split here is ~875,000 rows, and
Decision Tree / Random Forest / SVM would be computationally intractable on
that directly -- see FIXES_SCRATCH.md Section 2. Logistic Regression and
Naive Bayes are unaffected and still train on the full split.
"""

import pandas as pd
import numpy as np
import time
import random

from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mlfromscratch.supervised_learning import RandomForest, LogisticRegression, NaiveBayes, SupportVectorMachine
from mlfromscratch.supervised_learning.decision_tree import ClassificationTree

# ---------------------------------------------------------------------------
# Configuration. See ml_classification_tasks_dataset1_scratch.py for the
# rationale behind every constant below -- identical across all six scripts.
# ---------------------------------------------------------------------------

DATA_PATH = "/home/ug/RvsPython/ver/RvsPython/D3.csv"
OUTPUT_CSV = "output_ml_classification_taxi_scratch.csv"
RANDOM_STATE = 42
N_REPETITIONS = 10
SLEEP_SECONDS = 30

SCRATCH_TRAIN_CAP = 500
SCRATCH_PRED_CAP = 500
SCRATCH_MAX_DEPTH = 6
RF_N_ESTIMATORS = 10

csv_handler = CSVHandler(OUTPUT_CSV)


def sleep():
    time.sleep(SLEEP_SECONDS)


# ---------------------------------------------------------------------------
# Data preparation -- byte-for-byte the same as
# ../d3_cls/ml_classification_tasks_dataset3.py, including the median-split
# "long_trip" target (Adult/Drug Review's targets don't transfer to a heavily
# right-skewed continuous column, see that script's [DECISION-B]).
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

sc_X = StandardScaler()
X1 = sc_X.fit_transform(X_train)
X_pred_s = sc_X.transform(X_pred)
X_test_s = sc_X.transform(X_test)
Y1 = Y_train.values.ravel()
Y_test_arr = Y_test.values.ravel()


def cap_rows(*arrays, cap, seed=RANDOM_STATE):
    n = len(arrays[0])
    if n <= cap:
        return arrays
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, cap, replace=False)
    return tuple(a[idx] for a in arrays)


X1_scap, Y1_scap = cap_rows(X1, Y1, cap=SCRATCH_TRAIN_CAP)
if len(X1_scap) < len(X1):
    print(f"[note] scratch tree/forest/svm trained on {len(X1_scap)} of {len(X1)} rows")

X_pred_scap, = cap_rows(X_pred_s, cap=SCRATCH_PRED_CAP)
X_test_scap, Y_test_scap = cap_rows(X_test_s, Y_test_arr, cap=SCRATCH_PRED_CAP,
                                     seed=RANDOM_STATE + 1)
if len(X_test_scap) < len(X_test_s):
    print(f"[note] scratch tree/forest/svm scored on {len(X_test_scap)} of {len(X_test_s)} test rows")


def report(name, model, capped=False, svm=False):
    print(f"The below details are for {name}..")
    X_eval = X_test_scap if capped else X_test_s
    Y_eval = Y_test_scap if capped else Y_test_arr
    predicted = np.asarray(model.predict(X_eval))
    if svm:
        predicted = np.where(predicted <= 0, 0, 1)
    print("Accuracy= ", accuracy_score(Y_eval, predicted))
    print("Recall= ", recall_score(Y_eval, predicted, average='weighted', zero_division=0))
    print("Precision= ", precision_score(Y_eval, predicted, average='weighted', zero_division=0))
    print("f1 score= ", f1_score(Y_eval, predicted, average='weighted', zero_division=0))


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def random_forest_classification():
    model = RandomForest(n_estimators=RF_N_ESTIMATORS, max_depth=SCRATCH_MAX_DEPTH)
    model.fit(X1_scap, Y1_scap)
    return model


@measure_energy(handler=csv_handler)
def test_random_forest_classification():
    random_forest_classification()


random_forest_model = random_forest_classification()


@measure_energy(handler=csv_handler)
def test_random_forest_classification_inference():
    return np.asarray(random_forest_model.predict(X_pred_scap))


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def logistic_regression_classification():
    model = LogisticRegression()
    model.fit(X1, Y1)
    return model


@measure_energy(handler=csv_handler)
def test_logistic_regression_classification():
    logistic_regression_classification()


logistic_regression_model = logistic_regression_classification()


@measure_energy(handler=csv_handler)
def test_logistic_regression_classification_inference():
    return logistic_regression_model.predict(X_pred_s)


# ---------------------------------------------------------------------------
# Gaussian Naive Bayes -- fit() uncapped, predict() capped: see
# ml_classification_tasks_dataset1_scratch.py and FIXES_SCRATCH.md Section 2
# (NaiveBayes.predict is a pure-Python per-sample loop, benchmarked at >200s
# for a single call at this dataset's ~292,000-row test-set scale -- the
# scale that surfaced the need for this cap in the first place).
# ---------------------------------------------------------------------------

def gaussian_NB_classification():
    model = NaiveBayes()
    model.fit(X1, Y1)
    return model


@measure_energy(handler=csv_handler)
def test_gaussian_NB_classification():
    gaussian_NB_classification()


naive_bayes_model = gaussian_NB_classification()


@measure_energy(handler=csv_handler)
def test_gaussian_NB_classification_inference():
    return np.asarray(naive_bayes_model.predict(X_pred_scap))


# ---------------------------------------------------------------------------
# Support Vector Machine -- see FIXES_SCRATCH.md for the {-1,+1} label remap.
# ---------------------------------------------------------------------------

Y1_svm_scap = np.where(Y1_scap == 0, -1, 1)


def SVM_classification():
    model = SupportVectorMachine()
    model.fit(X1_scap, Y1_svm_scap)
    return model


@measure_energy(handler=csv_handler)
def test_SVM_classification():
    SVM_classification()


svm_classifier_model = SVM_classification()


@measure_energy(handler=csv_handler)
def test_SVM_classification_inference():
    return svm_classifier_model.predict(X_pred_scap)


# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------

def decision_tree_classification():
    model = ClassificationTree(max_depth=SCRATCH_MAX_DEPTH)
    model.fit(X1_scap, Y1_scap)
    return model


@measure_energy(handler=csv_handler)
def test_decision_tree_classification():
    decision_tree_classification()


decision_tree_classifier_model = decision_tree_classification()


@measure_energy(handler=csv_handler)
def test_decision_tree_classification_inference():
    return np.asarray(decision_tree_classifier_model.predict(X_pred_scap))


# ---------------------------------------------------------------------------
# Accuracy reporting, then the measurement loop
# ---------------------------------------------------------------------------

report("Random forest classification", random_forest_model, capped=True)
report("Logistic Regression classification", logistic_regression_model)
report("Naive Bayes classification", naive_bayes_model, capped=True)
report("SVM classification", svm_classifier_model, capped=True, svm=True)
report("Decision tree classification", decision_tree_classifier_model, capped=True)

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
