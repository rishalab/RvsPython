source("/home/rajrupa/cs22s504/energy_measurement.R")
library(tree)
library(glmnet)
library(e1071)
library(MASS)
#library(lmridge)
library(neuralnet)



sleep<-function(){
  Sys.sleep(30)
}
#dataset1 <- read.csv("/Users/rajrupachattaraj/Documents/Splash/adult.csv")
#data_infer<- read.csv("/Users/rajrupachattaraj/Documents/Splash/adult_infer1.csv")
#summary(train)
dataset1<-read.csv("/home/rajrupa/cs22s504/drug_review.csv", stringsAsFactors=T)
# data_infer<- read.csv("/home/rajrupa/cs22s504/drug_review.csv")


dataset1[dataset1 == '?'] =  NA
dataset1[sapply(dataset1, is.infinite)] <- NA
dataset1[sapply(dataset1, is.nan)] <- NA

# data_infer[data_infer == '?'] =  NA
# data_infer[sapply(data_infer, is.infinite)] <- NA
# data_infer[sapply(data_infer, is.nan)] <- NA



sample <- sample(c(TRUE, FALSE), nrow(dataset1), replace=TRUE, prob=c(0.8,0.2))
train_total  <- dataset1[sample, ]
test   <- dataset1[!sample, ]
sample_new <- sample(c(TRUE, FALSE), nrow(train_total), replace=TRUE, prob=c(0.75,0.25))
train <- train_total[sample_new, ]
infer <- train_total[!sample_new, ]


#Linear regression
test_linear_regression <- function(){
  #train$income <- as.numeric(gsub("\\.", "", train$income))
  #train$hours.per.week <- as.numeric(gsub("\\.", "", train$hours.per.week))
  #ataset1$income<-droplevels(train$income)
  #linear_regression_model<- lm(train$rating ~  train$drugName + train$condition , data = train, na.action = na.exclude)
  linear_regression_model<- lm(train$rating ~  train$drugName + train$condition + train$review + train$date , data = train, na.action = na.exclude)

  return(linear_regression_model)
}

linear_regression_model=test_linear_regression()

test_linear_regression_inference<- function(){

  predict(linear_regression_model,newdata=infer,interval = 'confidence', na.action=na.exclude)
}

for (i in 1:10) {
  measure_energy(test_linear_regression)()
  measure_energy(test_linear_regression_inference)()
}


# 



#Gaussian Regression

test_gaussian_regression<-function(){
  train$rating<- as.numeric(train$rating)

  #gaussian_regression_model<- glm(train$rating ~  train$drugName, data=train, family=gaussian, na.action = na.exclude)
  gaussian_regression_model<- glm(train$rating ~  train$drugName + train$condition + train$review + train$date, family=gaussian, na.action = na.exclude)
  return(gaussian_regression_model)
}



gaussian_regression_model=test_gaussian_regression()
test_gaussian_regression_inference<- function(){

  predict(gaussian_regression_model,newdata=infer,interval = 'confidence', na.action=na.exclude)
}

for (i in 1:10) {
  measure_energy(test_gaussian_regression)()
  measure_energy(test_gaussian_regression_inference)()
}




#Neural network Regression
test_neural_network_regression<-function(){
  #neural_network_model<- neuralnet(train$rating ~  train$drugName, data = train, linear.output = TRUE)
  neural_network_model<- neuralnet(train$rating ~  train$drugName + train$condition + train$review + train$date, data = train, linear.output = TRUE) 
  return(neural_network_model)

}

neural_network_model=test_neural_network_regression()

test_neural_network_regression_inference<- function(){
  # y.pred <- as.matrix(cbind(1,train))
  # coef(ridge_regression_model)

  predict(neural_network_model,newdata=infer,interval = 'confidence')
  #predict(ridge_regression_model,newdata=data_infer,interval = 'confidence')
}

for (i in 1:10) {
  sleep()
  measure_energy(test_neural_network_regression)()
  sleep()
  measure_energy(test_neural_network_regression_inference)()
}


# 
# 
# 
# 
# # data_infer$income<- as.numeric(data_infer$income)
# # data_infer$age<- as.numeric(data_infer$age)
# # data_infer$workclass<- as.numeric(data_infer$workclass)
# # data_infer$fnlwgt<- as.numeric(data_infer$fnlwgt)
# # data_infer$education<- as.numeric(data_infer$education)
# # data_infer$education.num<- as.numeric(data_infer$education.num)
# # data_infer$marital.status<- as.numeric(data_infer$marital.status)
# # data_infer$occupation<- as.numeric(data_infer$occupation)
# # data_infer$relationship<as.numeric(data_infer$relationship)
# # data_infer$race<as.numeric(data_infer$race)
# # data_infer$sex<as.numeric(data_infer$sex)
# # data_infer$capital.gain<as.numeric(data_infer$capital.gain)
# # data_infer$capital.loss<as.numeric(data_infer$capital.loss)
# # data_infer$hours.per.week<as.numeric(data_infer$hours.per.week)
# # data_infer$native.country<as.numeric(data_infer$native.country)
# # data_infer$income<as.numeric(data_infer$income)
# 
# 
#Decision tree regression
test_decision_tree<- function(){
  decision_tree_model<- tree(train$rating ~  train$drugName + train$condition + train$review + train$date, data=train)
  return(decision_tree_model)
}
# 
decision_tree_model=test_decision_tree()

test_decision_tree_inference<- function(){

  predict(decision_tree_model,newdata=infer,interval = 'confidence', na.action=na.exclude)
}
# 
# for (i in 1:10) {
#   sleep()
#   measure_energy(test_decision_tree)()
#   # sleep()
#   # measure_energy(test_decision_tree_inference)()
# }

# 
# 
# 
#SVM regression

test_svm_regression<- function(){
  svm_regressio_model<- svm(train$rating ~  train$drugName)
  return(svm_regressio_model)
}

svm_regressio_model=test_svm_regression()

test_svm_regression_inference<- function(){

  predict(svm_regressio_model,newdata=infer,interval = 'confidence', na.action=na.exclude)
}

for (i in 1:10) {
  sleep()
  measure_energy(test_svm_regression)()
  sleep()
  measure_energy(test_svm_regression_inference)()
}


function_list<-list()
function_list<-append(function_list,test_svm_regression)
function_list<-append(function_list,test_decision_tree)
#function_list<-append(function_list,test_neural_network_regression)
function_list<-append(function_list,test_gaussian_regression)
function_list<-append(function_list,test_linear_regression)
function_list<-append(function_list,test_svm_regression_inference)
function_list<-append(function_list,test_decision_tree_inference)
#function_list<-append(function_list,test_neural_network_regression_inference)
function_list<-append(function_list,test_gaussian_regression_inference)
function_list<-append(function_list,test_linear_regression_inference)

shuffled_list<-sample(function_list)
for (j in 1:length(shuffled_list)){
  sleep()
  print(shuffled_list[[j]])
  measure_energy(shuffled_list[[j]])()
}

for (i in 1:10) {
  shuffled_list<-sample(function_list)
  for (j in 1:length(shuffled_list)){
    sleep()
    print(shuffled_list[[j]])
    measure_energy(shuffled_list[[j]])()
  }

}
print("Process Done..")


