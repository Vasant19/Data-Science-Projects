import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.uscI import *

def load_data(file_path):
    """
     Loads data from a CSV file. This is a convenience function for use in testing. It will take a path to a CSV file and load it into a pandas dataframe
     
     @param file_path - Path to the CSV file
     
     @return DataFrame with the data from the CSV file as a column and the columns as a row. Example :
    """
    df = pd.read_csv(file_path)
    return df

def check_data(df): #OPTIONAL
    """
     Check data and plot. This is a function to run when you want to check the data in a pandas DataFrame
     
     @param df - pandas DataFrame
    """
    df.head()
    df.describe()
    df.shape
    sns.pairplot(df[['Age','Annual_Income','Spending_Score']])
    plt.show()
    
if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/mall_customers.csv'
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    check_data(df)
   