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
dataframe=pd.read_csv("/home/rajrupa/cs22s504/drug_review.csv")
#dataframe_infer=pd.read_csv("/home/rajrupa/cs22s504/drug_review.csv")
#dataframe_infer = pd.read_csv("/Users/rajrupachattaraj/Documents/Splash/adult_infer1.csv")
csv_handler = CSVHandler('output_ml_classification_drug.csv')

def sleep():
    time.sleep(30)

dataframe.replace("?", np.nan, inplace = True)
#dataframe_infer.replace("?", np.nan, inplace = True)
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

print("hi")
#training_features = ['drugName_n']
training_features = ['drugName_n','condition_n','review_n','date_n','usefulCount']
target = ['rating_n']
#inference_features = ['workclass_n','sex_n','fnlwgt','occupation_n','marital.status_n','education_n','education.num','capital.gain','hours.per.week','age','capital.loss','education_n']
#inference_target = ['income_n']
# X_infer=dataframe_infer[training_features]
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
    tn, fp, fn, tp = cm
    print("Accuracy= ",(tp+tn)/(tp+tn+fp+fn))
    recall = tp/(tp+fn)
    precision=tp/(tp+fp)
    print("Recall = Sensitivity = ",tp/(tp+fn))
    print("Specificity= ", tn/(tn+fp))
    print("Precision= ",tp/(tp+fp))
    f1score= 2 *(recall*precision)/(precision+recall)
    print("f1 score= ", f1score) 



def logistic_regression_classification():
    logistic_regression_model = LogisticRegression()
#     return logistic_regression_model.fit(X1,Y1.ravel())



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


for i in range(10):
    sleep()
    test_decision_tree_classification()
    sleep()
    test_decision_tree_classification_inference()


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
decision_tree_cassification_prediction()

function_list=[test_random_forest_classification,test_logistic_regression_classification,test_gaussian_NB_classification,test_SVM_classification,test_decision_tree_classification,test_decision_tree_classification_inference,test_gaussian_NB_classification_inference,test_logistic_regression_classification_inference,test_SVM_classification_inference,test_random_forest_classification_inference]
#function_list=[test_random_forest_classification,test_logistic_regression_classification,test_gaussian_NB_classification,test_SVM_classification,test_decision_tree_classification]
for i in range(10):
    print("This is iteration no:",i)
    random.shuffle(function_list)
    for j in range(len(function_list)):
        sleep()
        function_list[j]()
    


print("Process complete")
csv_handler.save_data()