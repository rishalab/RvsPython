source("/home/rajrupa/cs22s504/energy_measurement.R")
library(naivebayes)
library(e1071)
library(rpart)
library(randomForest)

sleep<-function(){
  Sys.sleep(30)
}

dataset1<-read.csv("/home/rajrupa/cs22s504/drug_review.csv", stringsAsFactors=T)
#data_infer<- read.csv("/home/rajrupa/cs22s504/adult_infer1.csv", stringsAsFactors=T)
#train <- read.csv("/Users/rajrupachattaraj/Documents/Splash/adult.csv", stringsAsFactors=T)

dataset1[dataset1 == '?'] =  NA
dataset1[sapply(dataset1, is.infinite)] <- NA
dataset1[sapply(dataset1, is.nan)] <- NA

#
# data_infer[data_infer == '?'] =  NA
# data_infer[sapply(data_infer, is.infinite)] <- NA
# data_infer[sapply(data_infer, is.nan)] <- NA



sample <- sample(c(TRUE, FALSE), nrow(dataset1), replace=TRUE, prob=c(0.9,0.1))
train_total <- dataset1[sample, ]
test   <- dataset1[!sample, ]
sample_new <- sample(c(TRUE, FALSE), nrow(train_total), replace=TRUE, prob=c(0.75,0.25))
train <- train_total[sample_new, ]
infer <- train_total[!sample_new, ]

# #Logistic Regression
# 
test_logistic_regression<- function(){
  # logistic_regression_model<- glm(train$income ~ train$hours.per.week+train$workclass  + train$sex +  train$race + train$relationship + train$native.country + train$fnlwgt + train$occupation + train$marital.status + train$education +train$education.num + train$capital.gain +train$workclass +train$age + train$capital.loss,  data = train, family = binomial)
  train$rating<-as.factor(train$rating)
  #logistic_regression_model<- glm(train$rating ~  train$drugName, data = train, family = binomial)
  logistic_regression_model<- glm(train$rating ~  train$drugName + train$condition + train$review + train$date, data = train, family = binomial)
  return(logistic_regression_model)
}

logistic_regression_model=test_logistic_regression()
test_logistic_regression_inference<- function(){

  predict(logistic_regression_model,newdata=infer,interval = 'confidence', na.action=na.exclude)
}

for (i in 1:10) {
  sleep()
  measure_energy(test_logistic_regression)()
  sleep()
  measure_energy(test_logistic_regression_inference)()
}

test_naivebayes_classification<- function(){
  train$rating<-as.factor(train$rating)
  
  naivebayes_classification_model<- gaussian_naive_bayes(train$rating ~  train$drugName + train$condition + train$review + train$date,  data = train, type="class")
  #naivebayes_classification_model<- naive_bayes(train$rating ~  train$drugName,  data = train, family = binomial)
  return(naivebayes_classification_model)
}

naivebayes_classification_model=test_naivebayes_classification()
test_naivebayes_classification_inference<-function(){
  predict(naivebayes_classification_model,newdata=infer,interval = 'confidence', na.action=na.exclude)
}
#
# for (i in 1:10) {
#   measure_energy(test_naivebayes_classification)()
#   measure_energy(test_naivebayes_classification_inference)()
# }

test_decision_tree_classification<- function(){
  #decision_tree_classifier<-rpart(train$income ~ train$hours.per.week+ train$workclass + train$sex +  train$race + train$relationship + train$native.country + train$fnlwgt + train$occupation + train$marital.status + train$education +train$education.num + train$capital.gain +train$workclass +train$age + train$capital.loss,  data = train)
  #decision_tree_classifier<-rpart(train$rating ~ train$drugName, data=train)
  decision_tree_classifier<-rpart(train$rating ~  train$drugName + train$condition + train$review + train$date, data=train)
  
  return(decision_tree_classifier)
}

decision_tree_classifier=test_decision_tree_classification()
test_decision_tree_classification_inference<-function(){
  predict(decision_tree_classifier,newdata=infer,interval = 'confidence', na.action=na.exclude)

}
#
# for (i in 1:10) {
#   measure_energy(test_decision_tree_classification)()
#   measure_energy(test_decision_tree_classification_inference)()
# }

test_svm_classification<- function(){
  #svm_classifier<-svm(train$income ~ train$hours.per.week+ train$workclass + train$sex +  train$race + train$relationship + train$native.country + train$fnlwgt + train$occupation + train$marital.status + train$education +train$education.num + train$capital.gain +train$workclass +train$age + train$capital.loss,  data = train, type= 'C-classification')
  #svm_classifier<-svm(train$rating ~ train$drugName, data = train, type= 'C-classification')
  svm_classifier<-svm(train$rating ~  train$drugName + train$condition + train$review + train$date, data = train, type= 'C-classification')
  
  return(svm_classifier)
}

svm_classifier=test_svm_classification()

test_svm_classification_inference<-function(){
  predict(svm_classifier,newdata=infer,interval = 'confidence', na.action=na.exclude)

}

for (i in 1:10) {
  measure_energy(test_svm_classification)()
  measure_energy(test_svm_classification_inference)()
}

#train$rating<-droplevels(train$rating)
test_random_forest_classification<-function(){
  
  train$rating<-as.factor(train$rating)
  train$rating<-as.numeric(as.character(train$rating))
  #train$rating<-droplevels(train$rating)
  train$drugName<-as.factor(train$drugName)
  train$drugName<-as.numeric(as.character(train$drugName))
  #train$drugName<-droplevels(train$drugName)
  
  #random_forest_classifier<-randomForest(rating ~ drugName, data=train,   na.action = na.roughfix)
  random_forest_classifier<-randomForest(rtrain$rating ~  train$drugName + train$condition + train$review + train$date, data=train,   na.action = na.roughfix)
  #random_forest_classifier<-randomForest(train$income ~  train$hours.per.week, data=train,   na.action = na.roughfix)
  #+ train$workclass + train$sex +  train$race + train$relationship + train$native.country + train$fnlwgt + train$occupation + train$marital.status + train$education +train$education.num + train$capital.gain +train$workclass +train$age + train$capital.loss, data=train,   na.action = na.exclude)
  return(random_forest_classifier)
}
random_forest_classifier=test_random_forest_classification()
test_random_forest_classification_inference<-function(){
  predict(random_forest_classifier,newdata=infer,interval = 'confidence', na.action=na.roughfix, type="prob")
}

for (i in 1:10) {
  sleep()
  measure_energy(test_random_forest_classification)()
  sleep()
  measure_energy(test_random_forest_classification_inference)()
}



function_list<-list()
function_list<-append(function_list,test_random_forest_classification)
function_list<-append(function_list,test_logistic_regression)
function_list<-append(function_list,test_naivebayes_classification)
function_list<-append(function_list,test_svm_classification)
function_list<-append(function_list,test_decision_tree_classification)
function_list<-append(function_list,test_random_forest_classification_inference)
function_list<-append(function_list,test_logistic_regression_inference)
function_list<-append(function_list,test_naivebayes_classification_inference)
function_list<-append(function_list,test_svm_classification_inference)
function_list<-append(function_list,test_decision_tree_classification_inference)

#
# shuffled_list<-sample(function_list)
# for (j in 1:length(shuffled_list)){
#   sleep()
#   print(shuffled_list[[j]])
#   measure_energy(shuffled_list[[j]])()
#
# }

for (i in 1:10) {

  shuffled_list<-sample(function_list)
  for (j in 1:length(shuffled_list)){
    sleep()
    print(shuffled_list[[j]])
    measure_energy(shuffled_list[[j]])()

  }

}
print("Process Done..")


