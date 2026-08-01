install.packages("rpart",  dependencies = TRUE, "~/Rlibs")
install.packages("e1071", dependencies = TRUE, "~/Rlibs")
install.packages("adabag", dependencies = TRUE, "~/Rlibs")
install.packages("ipred", dependencies = TRUE, "~/Rlibs")
library(e1071)
library(rpart)
library(adabag)
library(ipred)




source("PATH/energy_measurement.R")

dataset1<-read.csv("PATH/spam.csv", stringsAsFactors=T, check.names=F)

sleep<-function(){
  Sys.sleep(30)
}


dataset1[dataset1 == '?'] =  NA
dataset1[sapply(dataset1, is.infinite)] <- NA
dataset1[sapply(dataset1, is.nan)] <- NA

sample <- sample(c(TRUE, FALSE), nrow(dataset1), replace=TRUE, prob=c(0.9,0.1))
train  <- dataset1[sample, ]
test   <- dataset1[!sample, ]


test_ada_boost_classification<- function(){
  model_adaboost <- boosting(v1~v2, data=train, boos=TRUE, mfinal=50)
}


for (i in 1:10){
  sleep()
  measure_energy(test_ada_boost_classification)()
}
#measure_energy(test_ada_boost_classification)()



test_bagging_classification<-function(){
  ames_bag <- bagging(
    formula = v1 ~ v2,
    data = train,
    nbagg = 100,
    coob = TRUE,
    control = rpart.control(minsplit = 2, cp = 0)
  )
}

for (i in 1:10){
  sleep()
  measure_energy(test_bagging_classification)()
}

# #measure_energy(test_bagging_classification)()
#
#

test_decision_tree_classification<- function(){
  decision_tree_classifier<-rpart(train$v1 ~ train$v2, data = train)
  return(decision_tree_classifier)
}

for (i in 1:10){
  sleep()
  measure_energy(test_decision_tree_classification)()
}

#measure_energy(test_decision_tree_classification)()


test_svm_classification<- function(){
  svm_classifier<-svm(train$v1 ~ train$v2, data = train, type= 'C-classification')
  return(svm_classifier)
}


for (i in 1:10){
  sleep()
  measure_energy(test_svm_classification)()
}
#measure_energy(test_svm_classification)()






