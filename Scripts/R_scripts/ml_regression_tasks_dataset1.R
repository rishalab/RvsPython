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
#infer<- read.csv("/Users/rajrupachattaraj/Documents/Splash/adult_infer1.csv")
#summary(train)
dataset1<-read.csv("/home/rajrupa/cs22s504/adult.csv", stringsAsFactors=T)
# infer<- read.csv("/home/rajrupa/cs22s504/adult_infer1.csv")


dataset1[dataset1 == '?'] =  NA
dataset1[sapply(dataset1, is.infinite)] <- NA
dataset1[sapply(dataset1, is.nan)] <- NA

# infer[infer == '?'] =  NA
# infer[sapply(infer, is.infinite)] <- NA
# infer[sapply(infer, is.nan)] <- NA



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
  linear_regression_model<- lm(train$income ~  train$hours.per.week + train$sex +  train$race + train$relationship + train$native.country + train$fnlwgt + train$occupation + train$marital.status + train$education +train$education.num + train$capital.gain +train$workclass +train$age + train$capital.loss, data = train, na.action = na.exclude)

  return(linear_regression_model)
}

linear_regression_model=test_linear_regression()
test_linear_regression_inference<- function(){

  predict(linear_regression_model,newdata=infer,interval = 'confidence', na.action=na.exclude)
}



#Gaussian Regression
test_gaussian_regression<-function(){
  train$income<- as.numeric(train$income)

  gaussian_regression_model<- glm(train$income ~ train$hours.per.week + train$sex +  train$race + train$relationship + train$native.country + train$fnlwgt + train$occupation + train$marital.status + train$education +train$education.num + train$capital.gain +train$workclass +train$age + train$capital.loss, data=train, family=gaussian, na.action = na.exclude)
  return(gaussian_regression_model)
}

gaussian_regression_model=test_gaussian_regression()
test_gaussian_regression_inference<- function(){

  predict(gaussian_regression_model,newdata=infer,interval = 'confidence', na.action=na.exclude)
}







#Neural network Regression
test_neural_network_regression<-function(){
  neural_network_model<- neuralnet(income ~ hours.per.week + age + fnlwgt + education.num + capital.gain + capital.loss, data = train, linear.output = TRUE)
  return(neural_network_model)

}

neural_network_model=test_neural_network_regression()

test_neural_network_regression_inference<- function(){
  # y.pred <- as.matrix(cbind(1,train))
  # coef(ridge_regression_model)

  predict(neural_network_model,newdata=infer,interval = 'confidence')
  #predict(ridge_regression_model,newdata=infer,interval = 'confidence')
}



#
# # measure_energy(test_neural_network_regression)()
# # measure_energy(test_neural_network_regression_inference)()
#
#
#
#
# # infer$income<- as.numeric(infer$income)
# # infer$age<- as.numeric(infer$age)
# # infer$workclass<- as.numeric(infer$workclass)
# # infer$fnlwgt<- as.numeric(infer$fnlwgt)
# # infer$education<- as.numeric(infer$education)
# # infer$education.num<- as.numeric(infer$education.num)
# # infer$marital.status<- as.numeric(infer$marital.status)
# # infer$occupation<- as.numeric(infer$occupation)
# # infer$relationship<as.numeric(infer$relationship)
# # infer$race<as.numeric(infer$race)
# # infer$sex<as.numeric(infer$sex)
# # infer$capital.gain<as.numeric(infer$capital.gain)
# # infer$capital.loss<as.numeric(infer$capital.loss)
# # infer$hours.per.week<as.numeric(infer$hours.per.week)
# # infer$native.country<as.numeric(infer$native.country)
# # infer$income<as.numeric(infer$income)
#
#Decision tree regression
test_decision_tree<- function(){
  decision_tree_model<- tree(train$income ~ train$hours.per.week + train$sex +train$race + train$relationship, data=train)
  return(decision_tree_model)
  }

decision_tree_model=test_decision_tree()

test_decision_tree_inference<- function(){

  predict(decision_tree_model,newdata=infer,interval = 'confidence', na.action=na.exclude)
}

for (i in 1:10) {
  measure_energy(test_decision_tree)()
  measure_energy(test_decision_tree_inference)()
}
#
#
#
#SVM regression

test_svm_regression<- function(){
  svm_regressio_model<- svm(train$income ~ train$hours.per.week + train$sex +  train$race + train$relationship + train$native.country + train$fnlwgt + train$occupation + train$marital.status + train$education +train$education.num + train$capital.gain +train$workclass +train$age + train$capital.loss)
  return(svm_regressio_model)
}

svm_regressio_model=test_decision_tree()

test_svm_regression_inference<- function(){

  predict(svm_regressio_model,newdata=infer,interval = 'confidence', na.action=na.exclude)
}

for (i in 1:10) {
  measure_energy(test_svm_regression)()
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
#
# # shuffled_list<-sample(function_list)
# # for (j in 1:length(shuffled_list)){
# #   sleep()
# #   print(shuffled_list[[j]])
# #   measure_energy(shuffled_list[[j]])()
# # }
#
for (i in 1:10) {
  shuffled_list<-sample(function_list)
  for (j in 1:length(shuffled_list)){
    sleep()
    print(shuffled_list[[j]])
     measure_energy(shuffled_list[[j]])()
 }

}
print("Process Done..")


