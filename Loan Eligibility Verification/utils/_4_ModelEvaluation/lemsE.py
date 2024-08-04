import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.lemsI import *
from utils._2_Cleaning.lemsC import *
from utils._3_ModelTraining.lemsT import *

def evaluate_model_LogisticRegression(lrmodel,xtest_scaled,ytest): 
    """
     Evaluate logistic regression on xtest_scaled and print accuracy and confusion matrix
     
     @param lrmodel - a LRModel to be evaluated
     @param xtest_scaled - a numpy array of size ( nbitems )
     @param ytest - a numpy array of size ( nbitems
    """
    ypred = lrmodel.predict(xtest_scaled)
    # print(ypred)
    print(accuracy_score(ypred, ytest))
    print(confusion_matrix(ytest, ypred))
    
def evaluate_model_RandomForest(rfmodel,xtest,ytest):
    """
     Evaluates a random forest model. Prints accuracy and confusion matrix. This is a wrapper for predict () and predict_and_predict ()
     
     @param rfmodel - The RF model to be evaluated
     @param xtest - The test data used for predicting the model
     @param ytest - The test data used for predicting the
    """
    ypred = rfmodel.predict(xtest)
    # print(ypred)
    print(accuracy_score(ypred, ytest))
    print(confusion_matrix(ytest, ypred))

def kfolds_evaluations_lr(lrmodel,rfmodel,xtrain_scaled,ytrain):
    """
     Evaluates logistic regression and random forest on xtrain. Prints accuracy and standard deviations
     
     @param lrmodel - A callable that takes a numpy array of shape ( nb_samples n_features ) and returns a numpy array of shape ( nb_samples n_classes )
     @param rfmodel - A callable that takes a numpy array of shape ( nb_samples n_classes ) and returns a numpy array of shape ( nb_samples n
     @param xtrain_scaled
     @param ytrain
    """
    kfold = KFold(n_splits=5)
    lr_scores = cross_val_score(lrmodel, xtrain_scaled, ytrain, cv=kfold)
    rf_scores = cross_val_score(rfmodel, xtrain_scaled, ytrain, cv=kfold)

    print("Accuracy scores for Logistic regression :", lr_scores)
    print("Mean accuracy for Logistic regression :", lr_scores.mean())
    print("Standard deviation for Logistic regression :", lr_scores.std())
    print("Accuracy scores for random forest :", rf_scores)
    print("Mean accuracy for random forest :", rf_scores.mean())
    print("Standard deviation for random forest :", rf_scores.std())

    
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
        evaluate_model_LogisticRegression(lrmodel,xtest_scaled,ytest)
        kfolds_evaluations_lr(lrmodel,rfmodel,xtrain_scaled,ytrain)
    if rfmodel:
        print(f" Random forest : Successfully trained ")
        evaluate_model_RandomForest(rfmodel,xtest,ytest)

