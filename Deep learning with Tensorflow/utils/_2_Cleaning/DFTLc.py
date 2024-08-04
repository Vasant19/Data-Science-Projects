import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.DFTLi import *

def load_data(file_path):
    """
     Loads data from a CSV file. This is a convenience function for use in testing
     
     @param file_path - Path to the CSV file
     
     @return DataFrame with the data from the CSV file as a
    """
    df = pd.read_csv(file_path)
    return df

def prepare_and_splitdata(df):
    """
     Prepares and splits data to train and test. This is a helper function for : func : ` split_data `
     
     @param df - Dataframe containing data to be split
     
     @return X_scaled : pandas. DataFrame Dataframe containing transformed data for training x_train : pandas. DataFrame Transformed data for training x_test : pandas
    """
    Y= df.Attrition
    X= df.drop(columns = ['Attrition'])
    sc=StandardScaler()
    X_scaled=sc.fit_transform(X)
    X_scaled=pd.DataFrame(X_scaled, columns=X.columns)
    x_train,x_test,y_train,y_test=train_test_split(X_scaled,Y,test_size=0.2,random_state=1,stratify=Y)
    print("TRAIN TEST SPLIT SUCCESSFUL")
    return X,Y,X_scaled,x_train,x_test,y_train,y_test

if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/employee_attrition.csv'
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    X,Y,X_scaled,x_train,x_test,y_train,y_test = prepare_and_splitdata(df)
