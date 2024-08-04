import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.tsfI import *
from utils._2_Cleaning.tsfC import *

def decompisition_analysis(df):
    """
     Decompose and plot AAPL trend and residual. This is a function to be used in conjunction with plot_harmonics
     
     @param df - dataframe with data to
    """
    decomposed = seasonal_decompose(df['AAPL'])
    trend = decomposed.trend
    seasonal = decomposed.seasonal
    residual = decomposed.resid
    plt.figure(figsize=(12,8))
    plt.subplot(411)
    plt.plot(df['AAPL'], label='Original', color='black')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(trend, label='Trend', color='red')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonal', color='blue')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(residual, label='Residual', color='black')
    plt.legend(loc='upper left')
    plt.rcParams.update({'figure.figsize':(7,4), 'figure.dpi':80})
    # plot ACF
    plot_acf(df['AAPL'].dropna());
    # plot PACF
    plot_pacf(df['AAPL'].dropna(), lags=11);
    plt.show()

def train_arimamodel(df):
    """
     Train ARIMamodel on data. This is a wrapper around ARIMA
     
     @param df - dataframe with data to train
     
     @return A fitted ARIMamodel object with data from df
    """
    arima = ARIMA(df.AAPL, order=(1,1,1))
    ar_model = arima.fit()
    print(ar_model.summary())
    print("Arima model training complete")
    return ar_model

def train_arimamodel_bivariate(dfx):
    """
     Train arimamodel bivariate model. ARIMA ( dfx AAPL order = ( 1 1 1 ))
     
     @param dfx - Dataframe containing data to train model on
     
     @return A fitted arimamodel model as a : class : ` ~gensim. models. base_model. BaseModel
    """
    model2 = ARIMA(dfx.AAPL, order=(1,1,1))
    arimax = model2.fit()
    print(arimax.summary())
    print("Arima Bivariate model training complete")
    print(dfx.head(5))
    return arimax

def train_XGBOOST(dataYF):
    """
     Train XGBoost on data. Train the baseline model and make predictions
     
     @param dataYF - pandas dataframe with training data
     
     @return train test model1_preds features : list
    """
    # Train test split. Note, this is a time series data.
    train = dataYF.iloc[:-30]
    test = dataYF.iloc[-30:]
    # Be carefull not to use the next_day feature
    features = ['Open', 'High', 'Low', 'Close', 'Volume']

    model1 = XGBClassifier(max_depth=3, n_estimators=100, random_state=42)
    # Train the baseline model
    model1.fit(train[features], train['Target'])
    # Make predictions
    model1_preds = model1.predict(test[features])
    # Convert numpy array to pandas series
    model1_preds = pd.Series(model1_preds, index=test.index)
    print("Training XGBOOST Sucess")
    return train,test,model1,model1_preds,features
    
    
if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/AAPL.csv'
    # Test Execution: Load the data and check it
    data = load_data(file_path)
    df = pre_prep_data(data)
    check_data(df)
    decompisition_analysis(df)
    dfx = Bivariate_using_ExogenousVariable(data)
    train_arimamodel_bivariate(data)