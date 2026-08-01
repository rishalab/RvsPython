import sys
sys.path.append('/home/rajrupa/cs22s504/ML-From-Scratch')
import pandas as pd
import numpy as np
import time
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
from mlfromscratch.supervised_learning import LinearRegression, DecisionTreeRegressor, SupportVectorMachine, NeuralNetwork
from mlfromscratch.utils import train_test_split, mean_squared_error
import random

# Load the NYC Taxi Trip Duration dataset
data = pd.read_csv('nyc_taxi_trip_duration.csv')

# Feature engineering: extract datetime features, compute trip distance, etc.
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
data['dropoff_datetime'] = pd.to_datetime(data['dropoff_datetime'])
data['trip_duration'] = (data['dropoff_datetime'] - data['pickup_datetime']).dt.total_seconds()

# Dropping unnecessary columns and handling missing values
data = data.dropna()
X = data[['passenger_count', 'trip_distance', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]  # Feature columns
y = data['trip_duration']  # Target column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

# Initialize energy meter handler
csv_handler = CSVHandler('energy_consumption_scratch.csv')

# Function to measure energy for each model
@measure_energy(handler=csv_handler)
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Model: {model.__class__.__name__}, MSE: {mse}')

# Initialize different regression models using ML-From-Scratch library
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    SupportVectorMachine(),
    NeuralNetwork(n_hidden=[5, 5], n_iterations=500)  # Example configuration for NN
]

# Train and evaluate each model, and measure energy consumption
for model in models:
    train_and_evaluate(model, X_train, y_train, X_test, y_test)

# Save the energy consumption measurements
csv_handler.save_data()
