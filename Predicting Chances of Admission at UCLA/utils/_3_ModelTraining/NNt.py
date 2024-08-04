import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.NNi import *
from utils._2_Cleaning.NNc import *

def train_MultiLayerPerceptron(data,xtrain_scaled,ytrain):
    """
     Train multi - layer perceptron neural network on data. This is a wrapper for MLPClassifier
     
     @param data - Data to train the model on
     @param xtrain_scaled - Scaled x - axis data
     @param ytrain - Target y - axis data ( labels )
     
     @return MLPClassifier with trained parameters and training / validation data on it's own. It is used to train the
    """
    print(f"Data shape before training the model: ", data.shape)
    mlp = MLPClassifier(hidden_layer_sizes=(3), batch_size=50, max_iter=100, random_state=123)
    mlp.fit(xtrain_scaled,ytrain)
    print("Successfully trained Multi-Layer-Perceptron Neural Network")
    return mlp

def train_and_fit_using_GridSearchCV(mlp,x,y):
    """
     Trains and fits a MLP using GridSearchCV. This is a wrapper around the : class : ` GridSearchCV ` class
     
     @param mlp - The MLP to be trained
     @param x - The data used for training the model. Should be a Numpy array
     @param y - The target data used for training the model. Should be a Numpy array
     
     @return A grid search object with the training and validation data as well as the accuracy of the model's
    """
    params = {'batch_size':[20, 30, 40, 50],
          'hidden_layer_sizes':[(2,),(3,),(3,2)],
         'max_iter':[50, 70, 100]}
    grid = GridSearchCV(mlp, params, cv=10, scoring='accuracy')
    grid.fit(x, y)
    print("Grid Search Training complete")
    return grid
    
if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/Admission.csv'
    # Test Execution: Load the data and check it
    data = load_data(file_path)
    data = pre_prepdata(data)
    check_data(data)
    x,y,data,xtrain,xtest,ytrain,ytest,xtrain_scaled,xtest_scaled= cleanprep_and_split(data)
    mlp = train_MultiLayerPerceptron(data,xtrain_scaled,ytrain)
    grid = train_and_fit_using_GridSearchCV(mlp,x,y)
    
    