# RJoules
Towards Developing Energy Efficient R Programming
# Description
RJoules measures the energy consumption of R code snippeton Intel processor and provides domain-level granular results. This tool can be serves as a valuable resource for developers and practitioners seeking to develop energy-aware applications using the R languaguage. 

# System Requirements
1. Intel processors from the Celeron, Pentium, Core i3, i7, i9, and Xeon series, featuring the Sandy Bridge or newer architecture.</li>
2. Linux distribution Debian based operating system. We have tested on Ubuntu 20.04 and 22.04 Versions(recommended).</li>
3. Have <a href="https://cran.r-project.org">R</a> installed in your system.</li>

# Steps to run
1. Clone the repository using ```git clone https://github.com/rishalab/RJoules.git```
2. Navigate into the RJoules_main directory```cd RJoules_main``` and copy the path ```pwd```
3. For testing we have provided few <a href="https://github.com/rishalab/RJoules/tree/main/Test_files">test files</a>. Take any of the test file and add the copied path in source.
4. For checking the energy consumption due to the code snippet present in the test file, run the below command
 ```TESTFILENAME.R```
4. Check the energy measurement output ```vi output.csv```
5. This tool can be used to measure energy consumption of any code snippet written in R. For details please visit the Important links given below.


# Usage
1. Measuring energy consumption due to R code snippets.
2. Check energy consumption of machine learning models developed in R 


# Important Links
1. [Demonstration Video](https://www.youtube.com/watch?v=hP4pWJ4AKxE)
2. [Tool Website](https://rishalab.github.io/RJoules/)
