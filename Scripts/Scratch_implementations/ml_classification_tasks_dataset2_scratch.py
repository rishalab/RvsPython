"""
ml_classification_tasks_dataset2_scratch.py

Dataset2 (Drug Review) classification energy measurement using from-scratch
(non scikit-learn) Python implementations. Structurally identical to
../d3_cls/ml_classification_tasks_dataset2_rerun.py (same DATA_PATH_TRAIN/TEST
files, same 215,063-row combined dataset, same training_features, same
median-split "high_rating" binary target, same RANDOM_STATE/60-20-20 split,
same tags/output-CSV shape) and to ml_classification_tasks_dataset1_scratch.py
(same algorithm implementations, same SCRATCH_TRAIN_CAP/SCRATCH_PRED_CAP/
SCRATCH_MAX_DEPTH/RF_N_ESTIMATORS constants -- see FIXES_SCRATCH.md, shared
across all six scratch scripts).
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

DATA_PATH_TRAIN = "/home/ug/RvsPython/ver/drugsComTrain_raw.tsv"
DATA_PATH_TEST = "/home/ug/RvsPython/ver/drugsComTest_raw.tsv"
OUTPUT_CSV = "output_ml_classification_drug_scratch.csv"
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
# ../d3_cls/ml_classification_tasks_dataset2_rerun.py.
# ---------------------------------------------------------------------------

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

training_features = ['drugName_n', 'condition_n', 'review_n', 'date_n', 'usefulCount']

median_rating = dataframe['rating_n'].median()
dataframe['high_rating'] = (dataframe['rating_n'] >= median_rating).astype(int)
print(f"[note] median rating_n = {median_rating}; "
      f"class balance = {dataframe['high_rating'].mean():.3f}")
target = ['high_rating']

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
# for a single call at Dataset3's ~292,000-row test-set scale).
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
