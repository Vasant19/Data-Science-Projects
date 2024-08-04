import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.lemsI import *
from utils._2_Cleaning.lemsC import *

def train_LogisticRegression(xtrain_scaled,ytrain):
    """
     Train Logistic Regression model on training data. This is a wrapper for Lasso's fit function
     
     @param xtrain_scaled - Numpy array with training data
     @param ytrain - Numpy array with training labels ( 0 - indexed
    """
    lrmodel = LogisticRegression().fit(xtrain_scaled, ytrain)
    return lrmodel

def train_RandomForest(xtrain,ytrain):
    """
     Train a Random Forest classifier. This is a wrapper around the : class : RFClassifier class to allow training of random forests
     
     @param xtrain - features of training data.
     @param ytrain - labels of training data. Should be same shape as xtrain
     
     @return a : class : RFClassifier instance with trained random forests. See Also train_Bayesian
    """
    rfmodel = RandomForestClassifier(n_estimators=100, 
                                 min_samples_leaf=5, 
                                 max_features='sqrt')
    rfmodel.fit(xtrain, ytrain)
    return rfmodel 

if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/credit.csv'
                                                                                                                                                                   
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    check_data(df)
    df = cleanandprep_data(df)
    xtrain,xtest,xtrain_scaled, xtest_scaled, ytrain, ytest = splitdata_and_Scale(df)
    lrmodel = train_LogisticRegression(xtrain_scaled, ytrain)
    rfmodel = train_RandomForest(xtrain, ytrain)
    if lrmodel:
        print(f" Logistic Regression Model: Successfully trained ")
        ypred = lrmodel.predict(xtest_scaled)
        print(ypred)
    if rfmodel:
        print(f" Random forest : Successfully trained ")
        ypred = rfmodel.predict(xtest)
        print(ypred)


   
    
        
        