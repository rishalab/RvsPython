
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

import pandas as pd
import numpy as np
import time

from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

 


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import random

# Load the taxi trip dataset
dataframe = pd.read_csv("/path/to/taxi_trip_dataset.csv")

csv_handler = CSVHandler('output_ml_regression_taxi.csv')

def sleep():
    time.sleep(30)

# Handling missing values, if any
dataframe.replace("?", np.nan, inplace=True)

# Feature engineering: encoding categorical variables if necessary (like 'vendor_id', 'store_and_fwd_flag')
le_vendor = LabelEncoder()
le_store_fwd = LabelEncoder()

dataframe['vendor_id_n'] = le_vendor.fit_transform(dataframe['vendor_id'])
dataframe['store_and_fwd_flag_n'] = le_store_fwd.fit_transform(dataframe['store_and_fwd_flag'])

# Define features (X) and target (Y)
X = dataframe[['vendor_id_n', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
Y = dataframe['trip_duration']

# Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Standardizing features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Regression Models
@measure_energy(handler=csv_handler)
def linear_regression():
    model = LinearRegression()
    return model.fit(X_train, Y_train)

@measure_energy(handler=csv_handler)
def decision_tree_regression():
    model = DecisionTreeRegressor()
    return model.fit(X_train, Y_train)

@measure_energy(handler=csv_handler)
def gaussian_regression():
    model = GaussianProcessRegressor()
    return model.fit(X_train, Y_train)

@measure_energy(handler=csv_handler)
def support_vector_regression():
    model = SVR()
    return model.fit(X_train, Y_train)

@measure_energy(handler=csv_handler)
def neural_network_regression():
    model = MLPRegressor()
    return model.fit(X_train, Y_train)

@measure_energy(handler=csv_handler)
def test_linear_regression():
    linear_regression()

@measure_energy(handler=csv_handler)
def test_decision_tree_regression():
    decision_tree_regression()

@measure_energy(handler=csv_handler)
def test_gaussian_regression():
    gaussian_regression()

@measure_energy(handler=csv_handler)
def test_support_vector_regression():
    support_vector_regression()

@measure_energy(handler=csv_handler)
def test_neural_network_regression():
    neural_network_regression()

# Functions to evaluate models
def evaluate_model(model):
    Y_pred = model.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    print(f"R²: {r2}, MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

function_list = [
    test_linear_regression, test_decision_tree_regression, 
    test_gaussian_regression, test_support_vector_regression, 
    test_neural_network_regression
]

# Randomized model execution
for i in range(10):
    print("This is iteration no:", i)
    random.shuffle(function_list)
    for func in function_list:
        sleep()
        func()

print("Process complete")
csv_handler.save_data()
