

#Importing/sourcing RJoules

source("PATH/energy_measurement.R")


#Write the code snippet for which energy consumption measurement needs to be done
test_foo<-function(){
  
  print("This is my foo function..")
  df <- data.frame("Age" = c(12, 21, 15, 5, 25),
                   "Name" = c("Johnny", "Glen", "Alfie",
                              "Jack", "Finch"))

  newdf <- df[order(df$Age), ]
  print(newdf)
}

#Call measure_energy function defined in RJoules with test function name as parameter
measure_energy(test_foo)()