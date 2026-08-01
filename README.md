# RvsPython: Energy Consumption Analysis of Machine Learning Algorithms


## Overview
In this project, we conducted an emprical analysis of the energy consumption associated with different machine learning algorithms using the standard libraries of Python and R programming languages throughout the processes of model training and inference. 


## Tools Used
- **Python**
  - Libraries: scikit-learn libraries (specific libraries and version details provided in the draft)
  - Energy Consumption Measurement: pyJoules
- **R**
  - Libraries: CRAN repository (specific libraries and version details provided in the draft)
  - Energy Consumption Measurement: RJoules


## Methodology
1. **Data Collection**: Gathered datasets suitable for various machine learning tasks.
2. **Algorithm Implementation**: Implemented machine learning algorithms using Python and R.
3. **Energy Consumption Measurement**:
   - Utilized pyJoules for measuring energy consumption in Python scripts.
   - Employed RJoules for measuring energy consumption in R scripts.
4. **Analysis**: Compared energy consumption across different algorithms, libraries, and programming languages.


## Datasets Used
1. **[Adult Dataset](https://archive.ics.uci.edu/ml/datasets/Adult)**: Available from the UCI Machine Learning Repository.
2. **[Drug Review Dataset](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29)**: Also available from the UCI Machine Learning Repository.
3. **[New York City Taxi Trip Duration Dataset](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)**: Available from Kaggle.


## Files Included
- `Scripts/`: Contains all the scripts used during the experiments..
- `Results/`: Stores the results of energy consumption measurements, inside the `cumulative_results/` folder final results(mean values) are available 
- `README.md`: This file, providing an overview of the project.

## How to Replicate
1. Clone this repository to your local machine.
2. Navigate to the `Python_scripts/` or `R_scripts/` directory depending on the language you want to analyze.
3. Run the scripts using the respective language interpreters.
4. Refer to the `Results/` directory for energy consumption measurements and analysis outcomes.

