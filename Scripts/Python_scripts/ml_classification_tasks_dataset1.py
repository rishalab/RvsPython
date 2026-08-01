import pandas as pd
import numpy as np
import time
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from  sklearn.metrics import confusion_matrix 
import random

#dataframe = pd.read_csv("/Users/rajrupachattaraj/Documents/Splash/adult.csv")
dataframe=pd.read_csv("/home/rajrupa/cs22s504/adult.csv")
#dataframe_infer=pd.read_csv("/home/rajrupa/cs22s504/adult_infer1.csv")
#dataframe_infer = pd.read_csv("/Users/rajrupachattaraj/Documents/Splash/adult_infer1.csv")
csv_handler = CSVHandler('output_ml_classification_drug.csv')

def sleep():
    time.sleep(30)

dataframe.replace("?", np.nan, inplace = True)
# dataframe_infer.replace("?", np.nan, inplace = True)
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

training_features = ['workclass_n','sex_n','fnlwgt','occupation_n','marital.status_n','education_n','education.num','capital.gain','hours.per.week','age','capital.loss','education_n']
target = ['income_n']
#inference_features = ['workclass_n','sex_n','fnlwgt','occupation_n','marital.status_n','education_n','education.num','capital.gain','hours.per.week','age','capital.loss','education_n']
#inference_target = ['income_n']
#X_infer=dataframe_infer[training_features]
# X=dataframe[training_features]
# Y=dataframe[target]

X, X_test, Y, Y_test = train_test_split(dataframe[training_features],
                                                    dataframe[target],
                                                     test_size=0.2)

X_train, X_pred, Y_train, Y_pred = train_test_split(X,
                                                    Y,
                                                     test_size=0.25)

sc_X = StandardScaler()
sc_Y = StandardScaler()
X1 = sc_X.fit_transform(X_train)
Y1= sc_Y.fit_transform(Y_train)
Y1=Y1.astype(int)

def random_forest_classification():
    random_forest_model = RandomForestClassifier()
    return random_forest_model.fit(X1,Y1.ravel())

random_forest_model=random_forest_classification()

@measure_energy(handler=csv_handler)
def test_random_forest_classification_inference():
    y_infer=random_forest_model.predict(X_pred)
    return y_infer

@measure_energy(handler=csv_handler)
def test_random_forest_classification():
    random_forest_classification()


def random_forest_classification_prediction():
    print("The below details are for Random forest classification..")
    score=random_forest_model.score(X_test,Y_test)
    print("Model score= ",score)
    predicted_entropy = random_forest_model.predict(X_test)
    cm = confusion_matrix(Y_test,predicted_entropy)
    tn, fp, fn, tp = cm.ravel()
    print("Accuracy= ",(tp+tn)/(tp+tn+fp+fn))
    recall = tp/(tp+fn)
    precision=tp/(tp+fp)
    print("Recall = Sensitivity = ",tp/(tp+fn))
    print("Specificity= ", tn/(tn+fp))
    print("Precision= ",tp/(tp+fp))
    f1score= 2 *(recall*precision)/(precision+recall)
    print("f1 score= ", f1score) 


test_random_forest_classification()
test_random_forest_classification_inference()

def logistic_regression_classification():
    logistic_regression_model = LogisticRegression()
    return logistic_regression_model.fit(X1,Y1.ravel())



@measure_energy(handler=csv_handler)
def test_logistic_regression_classification():
    logistic_regression_classification()

logistic_regression_model=logistic_regression_classification()

@measure_energy(handler=csv_handler)
def test_logistic_regression_classification_inference():
    y_infer=logistic_regression_model.predict(X_pred)
    return y_infer


def logistic_regression_classification_prediction():
    print("The below details are for Logistic Regression classification..")
    score=logistic_regression_model.score(X_test,Y_test)
    print("Model score= ",score)
    predicted_entropy = logistic_regression_model.predict(X_test)
    cm = confusion_matrix(Y_test,predicted_entropy)
    tn, fp, fn, tp = cm.ravel()
    print("Accuracy= ",(tp+tn)/(tp+tn+fp+fn))
    recall = tp/(tp+fn)
    precision=tp/(tp+fp)
    print("Recall = Sensitivity = ",tp/(tp+fn))
    print("Specificity= ", tn/(tn+fp))
    print("Precision= ",tp/(tp+fp))
    f1score= 2 *(recall*precision)/(precision+recall)
    print("f1 score= ", f1score) 


def gaussian_NB_classification():
    naive_bayes_model = GaussianNB()
    return naive_bayes_model.fit(X1,Y1.ravel())

@measure_energy(handler=csv_handler)
def test_gaussian_NB_classification():
    gaussian_NB_classification()

naive_bayes_model=gaussian_NB_classification()

@measure_energy(handler=csv_handler)
def test_gaussian_NB_classification_inference():
    y_infer=naive_bayes_model.predict(X_pred)
    return y_infer

def gaussian_NB_classification_prediction():  
    print("The below details are for Naive Bayes classification..")
    score=naive_bayes_model.score(X_test,Y_test)
    print("Model score= ",score)
    predicted_entropy = naive_bayes_model.predict(X_test)
    cm = confusion_matrix(Y_test,predicted_entropy)
    tn, fp, fn, tp = cm.ravel()
    print("Accuracy= ",(tp+tn)/(tp+tn+fp+fn))
    recall = tp/(tp+fn)
    precision=tp/(tp+fp)
    print("Recall = Sensitivity = ",tp/(tp+fn))
    print("Specificity= ", tn/(tn+fp))
    print("Precision= ",tp/(tp+fp))
    f1score= 2 *(recall*precision)/(precision+recall)
    print("f1 score= ", f1score) 



def SVM_classification():
    svm_classifier_model = SVC()
    return svm_classifier_model.fit(X1,Y1.ravel())

@measure_energy(handler=csv_handler)
def test_SVM_classification():
    SVM_classification()


svm_classifier_model=SVM_classification()

@measure_energy(handler=csv_handler)
def test_SVM_classification_inference():
    y_infer=svm_classifier_model.predict(X_pred)
    return y_infer

def svm_classification_prediction():  
    print("The below details are for SVM classification..")
    score=svm_classifier_model.score(X_test,Y_test)
    print("Model score= ",score)
    predicted_entropy = svm_classifier_model.predict(X_test)
    cm = confusion_matrix(Y_test,predicted_entropy)
    tn, fp, fn, tp = cm.ravel()
    print("Accuracy= ",(tp+tn)/(tp+tn+fp+fn))
    recall = tp/(tp+fn)
    precision=tp/(tp+fp)
    print("Recall = Sensitivity = ",tp/(tp+fn))
    print("Specificity= ", tn/(tn+fp))
    print("Precision= ",tp/(tp+fp))
    f1score= 2 *(recall*precision)/(precision+recall)
    print("f1 score= ", f1score) 





def decision_tree_classification():
    decision_tree_classifier_model = SVC()
    return decision_tree_classifier_model.fit(X1,Y1.ravel())

@measure_energy(handler=csv_handler)
def test_decision_tree_classification():
    decision_tree_classification()



decision_tree_classifier_model=decision_tree_classification()
@measure_energy(handler=csv_handler)
def test_decision_tree_classification_inference():
    y_infer=decision_tree_classifier_model.predict(X_pred)
    return y_infer

def decision_tree_cassification_prediction():  
    print("The below details are for Decision tree classification..")
    score=decision_tree_classifier_model.score(X_test,Y_test)
    print("Model score= ",score)
    predicted_entropy = decision_tree_classifier_model.predict(X_test)
    cm = confusion_matrix(Y_test,predicted_entropy)
    tn, fp, fn, tp = cm.ravel()
    print("Accuracy= ",(tp+tn)/(tp+tn+fp+fn))
    recall = tp/(tp+fn)
    precision=tp/(tp+fp)
    print("Recall = Sensitivity = ",tp/(tp+fn))
    print("Specificity= ", tn/(tn+fp))
    print("Precision= ",tp/(tp+fp))
    f1score= 2 *(recall*precision)/(precision+recall)
    print("f1 score= ", f1score)    





random_forest_classification_prediction()
logistic_regression_classification_prediction()
gaussian_NB_classification_prediction()
svm_classification_prediction()
# decision_tree_cassification_prediction()

function_list=[test_random_forest_classification,test_logistic_regression_classification,test_gaussian_NB_classification,test_SVM_classification,test_decision_tree_classification,test_decision_tree_classification_inference,test_gaussian_NB_classification_inference,test_logistic_regression_classification_inference,test_SVM_classification_inference,test_random_forest_classification_inference]

for i in range(10):
    print("This is iteration no:",i)
    random.shuffle(function_list)
    for j in range(len(function_list)):
        sleep()
        function_list[j]()
    


print("Process complete")
csv_handler.save_data()