import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import random
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

#dataframe = pd.read_csv("/Users/rajrupachattaraj/Documents/Splash/adult.csv")
dataframe=pd.read_csv("/home/rajrupa/cs22s504/drug_review.csv")
dataframe_infer = pd.read_csv("/home/rajrupa/cs22s504/drug_review.csv")
csv_handler = CSVHandler('output_ml_regression_drug.csv')

def sleep():
    time.sleep(30)

dataframe.replace("?", np.nan, inplace = True)
#Handling categorical variables
le_drugName=LabelEncoder()
le_condition=LabelEncoder()
le_review=LabelEncoder()
le_date=LabelEncoder()
le_rating=LabelEncoder()
print("hello")
dataframe['drugName_n']=le_drugName.fit_transform(dataframe['drugName'])
dataframe['condition_n']=le_condition.fit_transform(dataframe['condition'])
dataframe['review_n']=le_review.fit_transform(dataframe['review'])
dataframe['date_n']=le_date.fit_transform(dataframe['date'])
dataframe['rating_n']=le_date.fit_transform(dataframe['rating'])
# dataframe_infer['drugName_n']=le_drugName.fit_transform(dataframe_infer['drugName'])
# dataframe_infer['condition_n']=le_condition.fit_transform(dataframe_infer['condition'])
# dataframe_infer['review_n']=le_review.fit_transform(dataframe_infer['review'])
# dataframe_infer['date_n']=le_date.fit_transform(dataframe_infer['date'])
# dataframe_infer['rating_n']=le_rating.fit_transform(dataframe_infer['rating'])

training_features = ['drugName_n','condition_n','review_n','date_n','usefulCount']
#training_features = ['drugName_n', 'condition_n']
# training_features = ['drugName_n']
target = ['rating_n']
#X_infer=dataframe_infer[training_features]
# X=dataframe[training_features]
# Y=dataframe[target]


X, X_test, Y, Y_test = train_test_split(dataframe[training_features],
                                                    dataframe[target],
                                                     test_size=0.2)

X_train, X_pred, Y_train, Y_pred = train_test_split(X,
                                                    Y,
                                                     test_size=0.25)
def linear_regression():
    linear_regression_model = LinearRegression()
    return linear_regression_model.fit(X_train,Y_train)

@measure_energy(handler=csv_handler)
def test_linear_regression():
    linear_regression()

linear_regression_model=linear_regression()

@measure_energy(handler=csv_handler)
def test_linear_regression_inference():
    y_infer=linear_regression_model.predict(X_pred)
    return y_infer

def linear_regression_prediction():
    print("The below details are for Linear Regression..")
    predicted_entropy = linear_regression_model.predict(X_test)
    r2=r2_score(Y_test,predicted_entropy)
    mae=mean_absolute_error(Y_test,predicted_entropy)
    mse=mean_squared_error(Y_test,predicted_entropy)
    rmse=np.sqrt(mse)
    print("r2 value= ",r2)
    print("MAE value= ",mae)
    print("MSE value= ",mse)
    print("RMSE value= ",rmse)

linear_regression_prediction()

for i in range(10):
    sleep()
    test_linear_regression()
    sleep()
    test_linear_regression_inference()

sc_X = StandardScaler()
sc_Y = StandardScaler()
X1 = sc_X.fit_transform(X_train)
Y1= sc_Y.fit_transform(Y_train)
Y1=Y1.astype(int)


def gaussian_regression():
    gaussian_regression_model = GaussianProcessRegressor()
    return gaussian_regression_model.fit(X1,Y1.ravel())


@measure_energy(handler=csv_handler)
def test_gaussian_regression():
    gaussian_regression()
    
gaussian_regression_model=gaussian_regression()

@measure_energy(handler=csv_handler)
def test_test_gaussian_regression_inference():
    y_infer=gaussian_regression_model.predict(X_pred)
    return y_infer

def gaussian_regression_prediction():
    print("The below details are for Gaussian Regression..")
    predicted_entropy = gaussian_regression_model.predict(X_test)
    r2=r2_score(Y_test,predicted_entropy)
    mae=mean_absolute_error(Y_test,predicted_entropy)
    mse=mean_squared_error(Y_test,predicted_entropy)
    rmse=np.sqrt(mse)
    print("r2 value= ",r2)
    print("MAE value= ",mae)
    print("MSE value= ",mse)
    print("RMSE value= ",rmse)
    
gaussian_regression_prediction()

test_gaussian_regression()

for i in range(10):
    print("Hi...")
    sleep()
    test_gaussian_regression()
    #sleep()
    #test_test_gaussian_regression_inference()

test_gaussian_regression()

def decision_tree_regression():
    decision_tree_regression_model = DecisionTreeRegressor(random_state = 0) 
    return decision_tree_regression_model.fit(X_train,Y_train)

@measure_energy(handler=csv_handler)
def test_decision_tree_regression():
    decision_tree_regression()

decision_tree_regression_model = decision_tree_regression()

@measure_energy(handler=csv_handler)
def test_decision_tree_regression_inference():
    y_infer=decision_tree_regression_model.predict(X_pred)
    return y_infer

def decision_tree_regression_prediction():
    print("The below details are for Decision tree Regression..")
    predicted_entropy = decision_tree_regression_model.predict(X_test)
    r2=r2_score(Y_test,predicted_entropy)
    mae=mean_absolute_error(Y_test,predicted_entropy)
    mse=mean_squared_error(Y_test,predicted_entropy)
    rmse=np.sqrt(mse)
    print("r2 value= ",r2)
    print("MAE value= ",mae)
    print("MSE value= ",mse)
    print("RMSE value= ",rmse)
   

for i in range(10):
    sleep()
    #decision_tree_regression_prediction()
    test_decision_tree_regression()
    #test_decision_tree_regression
    sleep()
    test_decision_tree_regression_inference()



def support_vector_regression():
    svr_regression_model= SVR()
    return svr_regression_model.fit(X1,Y1.ravel())

@measure_energy(handler=csv_handler)
def test_support_vector_regression():
    support_vector_regression()

svr_regression_model=support_vector_regression()

@measure_energy(handler=csv_handler)
def test_support_vector_regression_inference():
    y_infer=svr_regression_model.predict(X_pred)
    return y_infer

def support_vector_regression_prediction():
    print("The below details are for support vector Regression..")
    predicted_entropy = svr_regression_model.predict(X_test)
    r2=r2_score(Y_test,predicted_entropy)
    mae=mean_absolute_error(Y_test,predicted_entropy)
    mse=mean_squared_error(Y_test,predicted_entropy)
    rmse=np.sqrt(mse)
    print("r2 value= ",r2)
    print("MAE value= ",mae)
    print("MSE value= ",mse)
    print("RMSE value= ",rmse)
    

for i in range(10):
    sleep()
    #decision_tree_regression_prediction()
    test_support_vector_regression()
    #test_decision_tree_regression
    sleep()
    test_support_vector_regression_inference()


# support_vector_regression_prediction()
# test_support_vector_regression()

def neural_network_regression():
    neural_networkn_model= MLPRegressor()
    return neural_networkn_model.fit(X1,Y1.ravel())

@measure_energy(handler=csv_handler)
def test_neural_network_regression():
    neural_network_regression()

neural_networkn_model=neural_network_regression()


@measure_energy(handler=csv_handler)
def neural_network_regression_inference():
    y_infer=neural_networkn_model.predict(X_pred)
    return y_infer


def neural_network_regression_prediction():
    print("The below details are for Neural Network Regression..")
    predicted_entropy = neural_networkn_model.predict(X_test)
    r2=r2_score(Y_test,predicted_entropy)
    mae=mean_absolute_error(Y_test,predicted_entropy)
    mse=mean_squared_error(Y_test,predicted_entropy)
    rmse=np.sqrt(mse)
    print("r2 value= ",r2)
    print("MAE value= ",mae)
    print("MSE value= ",mse)
    print("RMSE value= ",rmse)

for i in range(10):
    sleep()
    test_neural_network_regression()
    sleep()
    neural_network_regression_inference()

neural_network_regression_prediction()
test_neural_network_regression()
    


linear_regression_prediction()
gaussian_regression_prediction()
decision_tree_regression_prediction()
support_vector_regression_prediction()
neural_network_regression_prediction()

#ridge_regression_prediction()

function_list=[test_decision_tree_regression,test_gaussian_regression,test_linear_regression,test_neural_network_regression,test_support_vector_regression,test_decision_tree_regression_inference,test_linear_regression_inference,test_test_gaussian_regression_inference,test_support_vector_regression_inference,test_linear_regression_inference]

for i in range(10):
    print("This is iteration no:",i)
    random.shuffle(function_list)
    for j in range(len(function_list)):
        sleep()
        function_list[j]()
    



print("Process complete")
csv_handler.save_data()