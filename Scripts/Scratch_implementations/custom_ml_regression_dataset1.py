
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

 


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import random
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

#dataframe = pd.read_csv("/Users/rajrupachattaraj/Documents/Splash/adult.csv")
dataframe=pd.read_csv("/home/rajrupa/cs22s504/adult.csv")
dataframe_infer = pd.read_csv("/home/rajrupa/cs22s504/adult_infer1.csv")
csv_handler = CSVHandler('output_ml_regression_adult.csv')

def sleep():
    time.sleep(30)

dataframe.replace("?", np.nan, inplace = True)
#Handling categorical variables
le_income=LabelEncoder()
le_sex=LabelEncoder()
le_occupation=LabelEncoder()
le_martial_status=LabelEncoder()
le_workclass=LabelEncoder()
le_education=LabelEncoder()
dataframe['income_n']=le_income.fit_transform(dataframe['income'])
dataframe['sex_n']=le_income.fit_transform(dataframe['sex'])
dataframe['occupation_n']=le_occupation.fit_transform(dataframe['occupation'])
dataframe['marital.status_n']=le_martial_status.fit_transform(dataframe['marital.status'])
dataframe['workclass_n']=le_workclass.fit_transform(dataframe['workclass'])
dataframe['education_n']=le_workclass.fit_transform(dataframe['education'])
# dataframe_infer['income_n']=le_income.fit_transform(dataframe_infer['income'])
# dataframe_infer['sex_n']=le_income.fit_transform(dataframe_infer['sex'])
# dataframe_infer['occupation_n']=le_occupation.fit_transform(dataframe_infer['occupation'])
# dataframe_infer['marital.status_n']=le_martial_status.fit_transform(dataframe_infer['marital.status'])
# dataframe_infer['workclass_n']=le_workclass.fit_transform(dataframe_infer['workclass'])
# dataframe_infer['education_n']=le_workclass.fit_transform(dataframe_infer['education'])

training_features = ['fnlwgt','sex_n','hours.per.week','occupation_n','marital.status_n','education_n','education.num','capital.gain','workclass_n','age','capital.loss','education_n']
target = ['income_n']
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

# linear_regression_prediction()
# test_linear_regression()

def gaussian_regression():
    gaussian_regression_model = GaussianProcessRegressor()
    return gaussian_regression_model.fit(X_train,Y_train)


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
   


sc_X = StandardScaler()
sc_Y = StandardScaler()
X1 = sc_X.fit_transform(X_train)
Y1= sc_Y.fit_transform(Y_train)

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
    predicted_entropy = svr_regression_model.predict(X_test)
    r2=r2_score(Y_test,predicted_entropy)
    mae=mean_absolute_error(Y_test,predicted_entropy)
    mse=mean_squared_error(Y_test,predicted_entropy)
    rmse=np.sqrt(mse)
    print("r2 value= ",r2)
    print("MAE value= ",mae)
    print("MSE value= ",mse)
    print("RMSE value= ",rmse)
    
 def ridge_regression():
     ridge_regression_model= Ridge()
     return ridge_regression_model.fit(X1,Y1.ravel())

 @measure_energy(handler=csv_handler)
 def test_ridge_regression():
     ridge_regression()

 ridge_regression_model=ridge_regression()

 @measure_energy(handler=csv_handler)
 def test_ridge_regression_inference():
     y_infer=ridge_regression_model.predict(X_infer)
     return y_infer

 def ridge_regression_prediction():
     print("The below details are for Ridge Regression..")
     predicted_entropy = ridge_regression_model.predict(X_test)
     r2=r2_score(Y_test,predicted_entropy)
     mae=mean_absolute_error(Y_test,predicted_entropy)
     mse=mean_squared_error(Y_test,predicted_entropy)
     rmse=np.sqrt(mse)
     print("r2 value= ",r2)
     print("MAE value= ",mae)
     print("MSE value= ",mse)
     print("RMSE value= ",rmse)


linear_regression_prediction()
gaussian_regression_prediction()
decision_tree_regression_prediction()
support_vector_regression_prediction()
neural_network_regression_prediction()
#ridge_regression_prediction()

function_list=[test_decision_tree_regression,test_gaussian_regression,test_linear_regression,test_neural_network_regression,test_support_vector_regression,test_decision_tree_regression_inference,test_linear_regression_inference,test_test_gaussian_regression_inference,test_support_vector_regression_inference,test_linear_regression_inference,  neural_network_regression_inference]

for i in range(10):
    print("This is iteration no:",i)
    random.shuffle(function_list)
    for j in range(len(function_list)):
        sleep()
        function_list[j]()
    


print("Process complete")
csv_handler.save_data()
