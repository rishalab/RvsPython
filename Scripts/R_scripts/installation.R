urlPackage<-"https://cran.r-project.org/src/contrib/Archive/randomForest/randomForest_4.6-12.tar.gz"
install.packages(urlPackage,repos=NULL, type="source",dependencies = TRUE ,"~/Rlibs")
install.packages("sgd",dependencies = TRUE ,"~/Rlibs")
install.packages("naivebayes",dependencies = TRUE ,"~/Rlibs")
install.packages("neuralnet",dependencies = TRUE ,"~/Rlibs")
install.packages("tree", dependencies =TRUE ,"~/Rlibs")
install.packages("e1071", dependencies = TRUE, "~/Rlibs")
install.packages("glmnet",dependencies =TRUE, "~/Rlibs")
install.packages("rpart",  dependencies = TRUE, "~/Rlibs")
install.packages("lmridge",  dependencies = TRUE, "~/Rlibs")
print("Done..")

