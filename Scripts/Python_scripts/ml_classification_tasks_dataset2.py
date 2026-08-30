"""
ml_classification_tasks_dataset2_rerun.py

Dataset2 (UCI Drug Review) classification energy measurement for the R vs Python IST revision.

Corrected rerun. Structurally identical to ml_classification_tasks_dataset3.py so that
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)

# ---------------------------------------------------------------------------
# Configuration. Identical across all six scripts.
# ---------------------------------------------------------------------------

# [FIX-DATA] The original path is another account's home directory and does
# not exist here. Originally pointed at a different, pre-cleaned Kaggle
# release of this corpus (110,811 + 46,108 = 156,919 rows, short of the
# paper's 215,063). Fixed to the actual UCI Drug Review release (train+test,
# tab-separated), downloaded directly from
# https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip
# -- 161,297 + 53,766 = 215,063 rows, an exact match to the paper's Section 5
# figure. See FIXES.md Section 1.
DATA_PATH_TRAIN = "/home/ug/RvsPython/ver/drugsComTrain_raw.tsv"
DATA_PATH_TEST = "/home/ug/RvsPython/ver/drugsComTest_raw.tsv"
OUTPUT_CSV = "output_ml_classification_drug_rerun.csv"
RANDOM_STATE = 42
N_REPETITIONS = 10
SLEEP_SECONDS = 30

# Kernel methods only. Fixed at 20000 because that is the value Dataset3 was
# run at, and the cap has to be identical at every scale for the scaling
# analysis to mean anything. State it in Section 5.
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

# [DECISION-C] The drug rating has ten levels while Dataset1 income is binary
# and the Dataset3 class is a median split. Class count was therefore not held
# constant across scales, and the effective Dataset2 threshold came from a
# truncation artefact rather than a choice. The rating is binarised at the
# median here so that all three scales run a balanced binary task. If Ch would
# rather keep the full ten level rating, set target = ['rating_n'] and drop the
# two lines above it. Whichever is chosen must be stated in Section 5.
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

# [DEV-2] Features are standardised; the class label is not. The originals
# passed the label through StandardScaler and then cast to int, which truncates.
# On Adult income that happened to preserve two classes. On the ten level drug
# rating it collapsed to two classes at a threshold near rating 3.7 that was an
# artefact rather than a design choice.
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
    # [DEV-3] Inference and scoring use the standardised matrix. In the
    # originals every classifier fitted on X1 but predicted on the raw X_pred
    # and X_test, so training and inference ran on different scales and the
    # published classification accuracy figures are not usable.
    #
    # [DEV-4] Metrics come from sklearn rather than from unpacking a confusion
    # matrix into tn, fp, fn, tp, which assumes a binary problem and raised on
    # the multiclass target.
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
    return RandomForestClassifier(random_state=RANDOM_STATE).fit(X1, Y1)


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
# [DEV-5] The return statement is present. In the Dataset2 script it was
# commented out, so the function returned None and the training measurement
# captured only object instantiation.

def logistic_regression_classification():
    return LogisticRegression(max_iter=1000).fit(X1, Y1)


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
    return GaussianNB().fit(X1, Y1)


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
    return SVC().fit(X1_kernel, Y1_kernel)


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
# [DEV-6] DecisionTreeClassifier, not SVC. In both original classification
# scripts this function instantiated SVC(), so every published Decision Tree
# classification figure for Dataset1 and Dataset2 was a second SVM run.

def decision_tree_classification():
    return DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X1, Y1)


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

# [DEV-7] One shuffled block, each function present exactly once, so every task
# gets exactly N_REPETITIONS measurements.
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
# [DEV-8] random_state is fixed on the splits, the trees, the MLP and the
#         kernel subsample, so the run is reproducible.
# [DEV-9] The output filename is distinct. Both original classification scripts
#         wrote to output_ml_classification_drug.csv, including the one reading
#         adult.csv.
# [DEV-10] The unused second read_csv of a derived inference file is dropped.
#          The Dataset1 regression script loaded adult_infer1.csv and never used
#          it, which blocks any rerun when that file is missing.
# ---------------------------------------------------------------------------
