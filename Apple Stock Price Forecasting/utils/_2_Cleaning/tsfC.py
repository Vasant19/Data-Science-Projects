import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.tsfI import *

def load_data(file_path):
    """
     Loads data from file. This is a convenience function to be used in testing
     
     @param file_path - Path to the file to load
     
     @return DataFrame with the data from the file. The columns are the names of
    """
    data = pd.read_csv(file_path)
    return data

def pre_prep_data(data):
    """
     Prepare data for plotting. This is a helper function to make it easier to use in the plot_data function
     
     @param data - pandas dataframe with data to plot
     
     @return pandas dataframe with data ready to plot in the plot_
    """
    print(f"Datatypes Original: ", data.dtypes)
    data['Date'] = pd.to_datetime(data['Date'])
    df = data.iloc[:-2,0:2]
    df = df.set_index('Date')
    return df
    
def check_data(df): #OPTIONAL
    """
    Check data and show seaborn lineplot if data is stationary.
    
    @param df - dataframe with AAPL column as key and data
    """

    print(f"Datatypes After Change: ", df.dtypes)
    #create seaborn lineplot
    plot = sns.lineplot(df['AAPL'])

    #rotate x-axis labels
    plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
    plt.show()
    # if p value < 0.05 the series is stationary
    results = adfuller(df['AAPL'])
    print('p-value:', results[1]) # adf, pvalue, usedlag_, nobs_, critical_values_, icbest_

def making_data_stationery(df):
    """
     This function plots 1st order differencing of AAPL. The plot is based on adfuller function
     
     @param df - pandas dataframe with data
    """
    # 1st order differencing
    v1 = df['AAPL'].diff().dropna()
    # adf test on the new series. if p value < 0.05 the series is stationary
    results1 = adfuller(v1)
    print('p-value:', results1[1]) # adf, pvalue, usedlag_, nobs_, critical_values_, icbest_
    plt.plot(v1)
    plt.title('1st order differenced series')
    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.ylabel('Price (USD)')
    plt.show()

def Bivariate_using_ExogenousVariable(data):
    """
     This function takes data from Bivariate_using_ExogenousVariable and converts it to a DataFrame
     
     @param data - pandas DataFrame with data to be converted
     
     @return dfx DataFrame with data converted to a DataFrame with columns 0 and
    """
    dfx = data.iloc[0:-2,0:3]
    return dfx

def yfinance_dataset_API():
    """
     Download and make Yfinance dataset using API. This is a function to be used in development and testing
     
     
     @return dataYF ['Target'] = True if target is greater
    """
    dataYF = yf.download("AAPL", start="2000-01-01", end="2022-05-31")
    dataYF['Next_day'] = dataYF['Close'].shift(-1)
    dataYF['Target'] = (dataYF['Next_day'] > dataYF['Close']).astype(int)
    dataYF.head(5)
    from matplotlib import pyplot as plt
    dataYF['Close'].plot(kind='line', figsize=(8, 4), title='line Plot')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.show()
    print("Yfinance dataset make success")
    return dataYF

if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/AAPL.csv'
    # Test Execution: Load the data and check it
    data = load_data(file_path)
    df = pre_prep_data(data)
    check_data(df)
    making_data_stationery(df)
    dfx = Bivariate_using_ExogenousVariable(data)
    dataYF = yfinance_dataset_API()