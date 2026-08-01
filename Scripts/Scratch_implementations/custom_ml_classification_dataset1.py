import sys
sys.path.append('/home/rajrupa/cs22s504/ML-From-Scratch')
import pandas as pd
import numpy as np
import time
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mlfromscratch.supervised_learning import RandomForest
from mlfromscratch.supervised_learning import LogisticRegression
from mlfromscratch.supervised_learning import NaiveBayes
from mlfromscratch.supervised_learning import SupportVectorMachine
from mlfromscratch.supervised_learning.decision_tree import DecisionTree
import random

# Load dataset (Modify the path as per your environment)
dataframe = pd.read_csv("/home/rajrupa/cs22s504/adult.csv")
csv_handler = CSVHandler('myoutput_classification1.csv')

def sleep():
    time.sleep(30)

# Preprocessing
dataframe.replace("?", np.nan, inplace=True)
le_income = LabelEncoder()
le_sex = LabelEncoder()
le_occupation = LabelEncoder()
le_martial_status = LabelEncoder()
le_workclass = LabelEncoder()
le_education = LabelEncoder()

dataframe['income_n'] = le_income.fit_transform(dataframe['income'])
dataframe['sex_n'] = le_income.fit_transform(dataframe['sex'])
dataframe['occupation_n'] = le_occupation.fit_transform(dataframe['occupation'])
dataframe['marital.status_n'] = le_martial_status.fit_transform(dataframe['marital.status'])
dataframe['workclass_n'] = le_workclass.fit_transform(dataframe['workclass'])
dataframe['education_n'] = le_workclass.fit_transform(dataframe['education'])

X = dataframe[['age', 'sex_n', 'education_n', 'occupation_n', 'hours.per.week']].values
Y = dataframe[['income_n']].values

# Split the dataset into 60% train, 20% validation, 20% test with random_state for reproducibility
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Defining and using the custom models from ML-From-Scratch

@measure_energy(handler=csv_handler)
def random_forest_classification():
    print("Training Random Forest...")
    model = RandomForest(n_estimators=10)
    model.fit(X_train, Y_train.ravel())
    return model

@measure_energy(handler=csv_handler)
def logistic_regression_classification():
    print("Training Logistic Regression...")
    model = LogisticRegression()
    model.fit(X_train, Y_train.ravel())
    return model

@measure_energy(handler=csv_handler)
def gaussian_NB_classification():
    print("Training Gaussian Naive Bayes...")
    model = NaiveBayes()
    model.fit(X_train, Y_train.ravel())
    return model

@measure_energy(handler=csv_handler)
def decision_tree_classification():
    print("Training Decision Tree...")
    model = DecisionTree()
    model.fit(X_train, Y_train.ravel())
    return model

@measure_energy(handler=csv_handler)
def SVM_classification():
    print("Training Support Vector Machine...")
    model = SupportVectorMachine()
    model.fit(X_train, Y_train.ravel())
    return model

# Inference functions
# Inference functions
random_forest_model=random_forest_classification()
@measure_energy(handler=csv_handler)
def test_random_forest_classification_inference():
    #model = random_forest_classification()
    y_pred = random_forest_model.predict(X_test)
    return y_pred


logistic_regression_model=logistic_regression_classification()
@measure_energy(handler=csv_handler)
def test_logistic_regression_classification_inference():
    #model = logistic_regression_classification()
    y_pred = logistic_regression_model.predict(X_test)
    return y_pred

gaussian_classification_model=gaussian_NB_classification()
@measure_energy(handler=csv_handler)
def test_gaussian_NB_classification_inference():
#    model = gaussian_NB_classification()
    y_pred = gaussian_classification_model.predict(X_test)
    return y_pred
##
decision_tree_model=decision_tree_classification()
@measure_energy(handler=csv_handler)
def test_decision_tree_classification_inference():
    model = decision_tree_classification()
    y_pred = model.predict(X_test)
    return y_pred

svm_classification_model=SVM_classification()
@measure_energy(handler=csv_handler)
def test_SVM_classification_inference():
    #svm_classification_model = SVM_classification()
    y_pred = svm_classification_model.predict(X_test)
    return y_pred
    
## Define dummy versions of the problematic methods
#def dummy_impurity_calculation(self, y, y1, y2):
#    print("Ignoring impurity calculation in DecisionTree")
#    return 0  # Return a dummy value to bypass the error
#
#def dummy_build_tree(self, X, y):
#    print("Ignoring tree building in DecisionTree")
#    self.root = None  # Avoid actual tree building
#
## Monkey patch the methods at runtime
#DecisionTree._impurity_calculation = dummy_impurity_calculation
#DecisionTree._build_tree = dummy_build_tree

# Shuffle and run tests
#function_list = [
#    random_forest_classification,
#    logistic_regression_classification,
#    gaussian_NB_classification,
#    SVM_classification,
#    decision_tree_classification,
#    test_random_forest_classification_inference,
#    test_logistic_regression_classification_inference,
#    test_gaussian_NB_classification_inference,
#    test_SVM_classification_inference,
#    test_decision_tree_classification_inference
#]
function_list = [
    random_forest_classification,
    logistic_regression_classification,
    gaussian_NB_classification,
    SVM_classification,
    test_random_forest_classification_inference,
    test_logistic_regression_classification_inference,
    test_gaussian_NB_classification_inference,
    test_SVM_classification_inference,
    decision_tree_classification,
    test_decision_tree_classification_inference
]
# Shuffle and run tests
#function_list = [
#    decision_tree_classification,
#    test_decision_tree_classification_inference
#]
for i in range(10):
    print("This is iteration no:", i)
    random.shuffle(function_list)
    for j in range(len(function_list)):
        sleep()
        function_list[j]()

print("Process complete")
csv_handler.save_data()
