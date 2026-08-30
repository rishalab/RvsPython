"""
ml_classification_tasks_dataset1_scratch.py

Dataset1 (Adult Census Income) classification energy measurement using
from-scratch (non scikit-learn) Python implementations. Structurally
identical to ../d3_cls/ml_classification_tasks_dataset1_rerun.py (same
DATA_PATH/DATA_PATH_TEST files, same 48,842-row combined dataset, same
training_features/target, same RANDOM_STATE/60-20-20 split, same tags/
output-CSV shape). See FIXES_SCRATCH.md (shared across all six scratch
scripts) for the SCRATCH_TRAIN_CAP/SCRATCH_PRED_CAP/SCRATCH_MAX_DEPTH/
n_estimators rationale and benchmarks.

No scikit-learn model class is used below; LabelEncoder, StandardScaler,
train_test_split and the sklearn.metrics scoring functions are preprocessing/
scoring utilities only, exactly as in the original Table 8/9 methodology.
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
# Configuration. Identical across all six scratch scripts (matching the
# corresponding ../d3_cls script's DATA_PATH / RANDOM_STATE / N_REPETITIONS /
# SLEEP_SECONDS exactly).
# ---------------------------------------------------------------------------

DATA_PATH = "/home/ug/RvsPython/ver/adult.data"
DATA_PATH_TEST = "/home/ug/RvsPython/ver/adult.test"
OUTPUT_CSV = "output_ml_classification_adult_scratch.csv"
RANDOM_STATE = 42
N_REPETITIONS = 10
SLEEP_SECONDS = 30

# [SCRATCH-CAP] Decision Tree, Random Forest (built from the same from-scratch
# tree) and SVM (cvxopt QP over an n x n kernel matrix) are capped on both
# training and query rows -- benchmarked in FIXES_SCRATCH.md Section 2.
# Naive Bayes' fit() is cheap and uncapped, but its predict() is a pure-Python
# per-sample loop benchmarked at >200s for one call at Dataset3's ~292,000-row
# test-set scale, so its predict side is capped too. Logistic Regression is
# fully uncapped (fit *and* predict) -- both benchmarked as tractable
# (seconds, not hours/minutes) at the full, uncapped Dataset3 scale.
SCRATCH_TRAIN_CAP = 500
SCRATCH_PRED_CAP = 500

# [SCRATCH-CAP] Same depth bound as the regression scripts, applied to
# ClassificationTree both standalone and as RandomForest's base learner.
SCRATCH_MAX_DEPTH = 6

# [SCRATCH-CAP] mlfromscratch's RandomForest defaults to n_estimators=100
# (matching the paper's "each package's own defaults" philosophy for the
# scikit-learn/CRAN comparison -- FIXES.md Section 7.3), but 100 from-scratch
# trees at any SCRATCH_TRAIN_CAP-sized sample is not tractable at
# N_REPETITIONS=10 x six scripts. Reduced to 10; disclosed deviation, see
# FIXES_SCRATCH.md Section 2.
RF_N_ESTIMATORS = 10

csv_handler = CSVHandler(OUTPUT_CSV)


def sleep():
    time.sleep(SLEEP_SECONDS)


# ---------------------------------------------------------------------------
# Data preparation -- byte-for-byte the same as
# ../d3_cls/ml_classification_tasks_dataset1_rerun.py.
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


# [SCRATCH-CAP] Decision Tree, Random Forest and SVM train on this capped
# subset; Logistic Regression and Naive Bayes train on the full X1/Y1.
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
        # mlfromscratch's SupportVectorMachine is trained on {-1,+1} labels
        # (see SVM_classification() below) and predicts in that space.
        predicted = np.where(predicted <= 0, 0, 1)
    print("Accuracy= ", accuracy_score(Y_eval, predicted))
    print("Recall= ", recall_score(Y_eval, predicted, average='weighted', zero_division=0))
    print("Precision= ", precision_score(Y_eval, predicted, average='weighted', zero_division=0))
    print("f1 score= ", f1_score(Y_eval, predicted, average='weighted', zero_division=0))


# ---------------------------------------------------------------------------
# Random Forest -- mlfromscratch RandomForest, depth- and estimator-reduced;
# see RF_N_ESTIMATORS / SCRATCH_MAX_DEPTH / SCRATCH_TRAIN_CAP above.
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
# Logistic Regression -- mlfromscratch LogisticRegression (full-batch gradient
# descent). Uncapped: vectorised, benchmarked tractable even at ~800,000 rows.
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
# Gaussian Naive Bayes -- mlfromscratch NaiveBayes is algorithmically the same
# Gaussian-likelihood model as sklearn's GaussianNB (per-class/per-feature
# mean+variance), not a substitute. fit() is uncapped (near-instant even at
# scale), but predict() is a pure-Python per-sample loop and is NOT
# vectorised like LogisticRegression's -- benchmarked at >200s for a single
# call on Dataset3's ~292,000-row test set (FIXES_SCRATCH.md Section 2), so
# its predict side is capped the same as Decision Tree/Random Forest/SVM.
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
# Support Vector Machine -- mlfromscratch SupportVectorMachine (dual QP via
# cvxopt). Requires labels in {-1,+1}; income_n/high_rating/long_trip are all
# already {0,1}, remapped here and back in report()/inference.
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
# Decision Tree -- mlfromscratch ClassificationTree, depth-bounded and
# row-capped; see SCRATCH_MAX_DEPTH / SCRATCH_TRAIN_CAP above.
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
