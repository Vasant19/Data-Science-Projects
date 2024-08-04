import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.lemsI import *

#Load the data using the file path mentioned
def load_data(file_path):
    """
     Loads data from a CSV file. This is a convenience function for use in testing. It will take a path to a CSV file and load it into a pandas dataframe
     
     @param file_path - Path to the CSV file
     
     @return DataFrame with the data from the CSV file as a column and the columns as a row. Example :
    """
    df = pd.read_csv(file_path)
    return df

#To Get Information on the Dataframe (Optional)
def check_data(df):
    """
     Check data and plot. This is a function to run when you want to check the data in a pandas DataFrame
     
     @param df - pandas DataFrame with Loan
    """
    print(df.head(5))
    print(df.shape)
    df['Loan_Status'].value_counts().plot.bar()
    print(df.isnull().sum())
    print(df.dtypes)
    print(df['Dependents'].mode()[0])
    sns.displot(df['LoanAmount'])
    plt.show()
    
# Impute missing values, Dropping Unneccesary Columns and Getting dummies for Categorical columns 
def cleanandprep_data(df):
    """
     Cleans and preps data to be used in tests.
     
     @param df - Dataframe containing the data to be cleaned and prepped
     
     @return Dataframe containing cleaned and prepped
    """
    df['Gender'].fillna('Male', inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    
    df = df.drop('Loan_ID', axis=1)
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
    #Debugging
    # print("Data types after get_dummies:")
    # print(df.dtypes)
    # print("Checking for null values after data cleaning:")
    # print(df.isnull().sum())
    
    return df

#Split Data into Train-Test and Scale
def splitdata_and_Scale(df):
    """
     Split data into training and test sets and scale them. This is a helper function for test_split and train_test
     
     @param df - dataframe with Loan_Status columns
     
     @return tuple of data for training and test set xtrain_scaled : numpy array of training data xtest_scaled : numpy array of test
    """
    x = df.drop('Loan_Status',axis=1)
    y = df['Loan_Status']
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=123)   
    # print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)   #OPTIONAL TO CHECK THE SHAPE
    scale = MinMaxScaler()
    xtrain_scaled = scale.fit_transform(xtrain)
    xtest_scaled = scale.transform(xtest)
    return xtrain,xtest,xtrain_scaled, xtest_scaled, ytrain, ytest

    
#TEST
if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/credit.csv'
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    check_data(df)
    df = cleanandprep_data(df)
    xtrain_scaled, xtest_scaled, ytrain, ytest = splitdata_and_Scale(df)
    print(xtrain_scaled,xtest_scaled)
