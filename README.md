# Data-Science-Projects
### This repository serves both educational and research purposes related to different industries
#### Please Check the Notebook(.ipynb) files for step-by-step analysis and python (.py) for modularized scripts for code.

# Involving 7 different modularized algorithms for different Use-cases and problem solutions
### All the machine learning models follow same Directory structure and flow while everything else, from dataset to algorithm selection being different. 
#### Machine learning Models used:
`1. House Price Prediction: **Linear Regression, Decision Tree and Random Forest**`

`2. Loan Eligibility Verification: **Logistic Regression and Random Forest**`

`3. Mall Customers Segregration: **Kmeans Clustering**`

`4. Predicting Chances of Admission at UCLA: **Multi-Layer Perceptron (Neural Network)**`

`5. Employee Attrition Prediction: **Support Vector Machine using Different Kernels**`

`6. Deep Learning with Tensorflow: **Neural networks using Sequential API from Tensorflow, experimentation with Different activation functions**`

`7. Apple Stock Price Prediction: **TIME SERIES FORECASTING with ARIMA and ARIMAX, Additionally XGBOOST used**`

## First step is to Clone the repository
```python
git clone https://github.com/Vasant19/Data-Science-Projects.git
```
## Second step is to Install the requirements.txt
```python
pip install requirements.txt
```

## Third step is to Change the directory in the terminal you want to run or edit hyperparameters of a model
#### For example: 
```python
cd "House Price Prediction"
```

## Finally Run the main file containing modularized code
```python
python MLMAIN.py
```
### For relative imports code below is used (check explaination.ipynb) for more information)
```python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
```
