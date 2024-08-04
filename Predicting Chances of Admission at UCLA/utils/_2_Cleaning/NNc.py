import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.NNi import *

def load_data(file_path):
    """
     Loads data from file. This is a convenience function to be used in testing
     
     @param file_path - Path to the file to load
     
     @return DataFrame with the data from the file. The columns are the names of
    """
    data = pd.read_csv(file_path)
    return data



def pre_prepdata(data):
    """
     Prepares and sanitizes data before processing. This is a helper function to be used by : py : func : ` prepare `.
     
     @param data - The data to be sanitized. Should be a dictionary where keys are column names and values are numpy arrays of length equal to the number of columns in the
    """
    # Converting the target variable into a categorical variable
    data['Admit_Chance']=(data['Admit_Chance'] >=0.8).astype(int)
    data = data.drop(['Serial_No'], axis=1)
    return data
    
def check_data(data): 
    """
     Check the data and visualize it. This is a function to be called from test_data.
     
     @param data - The data to check. Should be a pysynphot
    """
    print(data.head())
    print(data.shape)
    print(data.info())
    print(data.describe().T)
    plt.figure(figsize=(15,8))
    sns.scatterplot(data=data, 
           x='GRE_Score', 
           y='TOEFL_Score', 
           hue='Admit_Chance');
    plt.title("Visualize the dataset to identify some patterns")
    plt.show()
    
def cleanprep_and_split(data):
    """
     Split and clean - prep data for training and testing. This is a helper function to do the following : 1. Remove data columns from the data that are not used in the training set. 2
     
     @param data
    """
    data = pd.get_dummies(data, columns=['University_Rating','Research'])
    x = data.drop(['Admit_Chance'], axis=1)
    y = data['Admit_Chance']
    xtrain, xtest, ytrain, ytest =  train_test_split(x, y, test_size=0.2, random_state=123)
    # fit calculates the mean and standard deviation
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    # Now transform xtrain and xtest

    xtrain_scaled = scaler.transform(xtrain)
    xtest_scaled = scaler.transform(xtest)
    print("Data split successfully and transformed properly")
    return x,y,data,xtrain,xtest,ytrain,ytest,xtrain_scaled,xtest_scaled
    
if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/Admission.csv'
    # Test Execution: Load the data and check it
    data = load_data(file_path)
    data = pre_prepdata(data)
    check_data(data)
    data = cleanprep_and_split(data)