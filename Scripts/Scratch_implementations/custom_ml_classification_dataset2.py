
import sys
sys.path.append('/home/rajrupa/cs22s504/ML-From-Scratch')
import pandas as pd
import numpy as np
import time
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mlfromscratch.supervised_learning import RandomForest, LogisticRegression, NaiveBayes, SupportVectorMachine
from mlfromscratch.supervised_learning.decision_tree import DecisionTree
import random

# Load dataset (Modify the path as per your environment)
dataframe = pd.read_csv("/home/rajrupa/cs22s504/drug_review.csv")
csv_handler = CSVHandler('myoutput_classification2.csv')

def sleep():
    time.sleep(30)

# Preprocessing
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

X = dataframe[['drugName_n', 'condition_n', 'review_n', 'date_n', 'rating_n']].values
y = dataframe['rating_n'].values

# Split the dataset into 60% train, 20% validation, 20% test with random_state for reproducibility
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Logistic Regression (with energy measurement)
@measure_energy(handler=csv_handler)
def logistic_regression_classification():
    model = LogisticRegression(learning_rate=0.01)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"Logistic Regression Validation Accuracy: {np.mean(y_pred == y_val)}")

decision_tree_model=logistic_regression_classification()
@measure_energy(handler=csv_handler)
def logistic_regression_classification_inference():
    #model = LogisticRegression(learning_rate=0.01)
    #decision_tree_model.fit(X_train, y_train)
    y_pred = decision_tree_model.predict(X_test)
    return y_pred

# Decision Tree (with energy measurement)
@measure_energy(handler=csv_handler)
def decision_tree_classification():
    model = DecisionTree(max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"Decision Tree Validation Accuracy: {np.mean(y_pred == y_val)}")

decision_tree_model=decision_tree_classification()
@measure_energy(handler=csv_handler)
def decision_tree_classification_inference():
    decision_tree_model = DecisionTree(max_depth=10)
    y_pred = decision_tree_model.predict(X_test)
    return y_pred

# Naive Bayes (with energy measurement)
@measure_energy(handler=csv_handler)
def naive_bayes_classification():
    model = NaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"Naive Bayes Validation Accuracy: {np.mean(y_pred == y_val)}")

naive_bayes_classification_model=naive_bayes_classification()
@measure_energy(handler=csv_handler)
def naive_bayes_classification_inference():
    #model = NaiveBayes()
    naive_bayes_classification_model.fit(X_train, y_train)
    y_pred = naive_bayes_classification_model.predict(X_test)
    return y_pred

# Support Vector Machine (with energy measurement)
@measure_energy(handler=csv_handler)
def svm_classification():
    model = SupportVectorMachine()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"SVM Validation Accuracy: {np.mean(y_pred == y_val)}")
    
svm_classification_model=svm_classification()
@measure_energy(handler=csv_handler)
def svm_classification_inference():
    #model = SupportVectorMachine()
    svm_classification_model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# Random Forest (with energy measurement)
@measure_energy(handler=csv_handler)
def random_forest_classification():
    model = RandomForest(n_estimators=10)
    #model = RandomForest()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"Random Forest Validation Accuracy: {np.mean(y_pred == y_val)}")

random_forest_classification_model=random_forest_classification()
@measure_energy(handler=csv_handler)
def random_forest_classification_inference():
    #model = RandomForest(n_estimators=10, max_depth=10)
    random_forest_classification_model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

#svm_classification()
#random_forest_classification()
# Shuffle and run classification and inference tests
function_list = [
    logistic_regression_classification,
    naive_bayes_classification,
    svm_classification,
    random_forest_classification,
    logistic_regression_classification_inference,
    naive_bayes_classification_inference,
    svm_classification_inference,
    random_forest_classification_inference
]
#
for i in range(10):
    print(f"This is iteration no: {i+1}")
    random.shuffle(function_list)
    for func in function_list:
        sleep()
        func()

print("Process complete")
csv_handler.save_data()
