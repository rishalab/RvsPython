# Please follow the below steps before executing the evaluation script: test_models.R
#### Kindly note Step2 to Step4 are to be performed due to the library dependencies of machine learning algorithms, these are not required if energy measurements are being done for simple code snippets(for example, please refer testfile1.R in <a href="https://github.com/rishalab/RJoules/tree/main/Test_files">test files</a>) where no external library is required.

### Step1
Mention the correct absolute path for below two lines in test_models.R(Importing RJoules and dataset(spam.csv)) 
```source("PATH/energy_measurement.R")``` 
```dataset1<-read.csv("PATH/spam.csv", stringsAsFactors=T, check.names=F)```
### Step2: 
Save the libpath.sh in the same directory as test_models.R It is responsible for making a directory where we’ll install required libraries for machine learning algorithms 
### Step3: 
Making libpath.sh executable.  ```chmod u+x libPath.sh```
### Step4: 
Run it with command ```./libPath.sh```
### Step5: 
Run test_models.R to measure energy for machine learning algorithms. ```R<test_models.R --no-save```
### Step6: 
Check the output ```vi output.csv```

## Note: 
Since RJoules access intel-RAPL directories in host machine you might encounter permission error. 
It can be resolved with ```sudo chmod -R a+r /sys/class/powercap/intel-rapl``` or with the similar command ```sudo chown -R energy /sys/class/powercap/intel-rapl```

You might get error stating a particular library(for ML example: “rpart library not found”) not found even after installing it. Please rerun ```./libPath.sh``` using step4 and follow from step5 again.
