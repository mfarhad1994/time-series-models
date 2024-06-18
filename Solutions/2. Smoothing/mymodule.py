import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def check_df(dataframe, head=3):

    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Duplicate Values #####################")
    print(dataframe.duplicated().sum())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Unique Values #####################")
    print(dataframe.nunique())

#----------------------------------------------------------------------------------------------
def find_categorical_columns(df, max_unique_values=7):
    categorical_cols = []
    for col in df.columns:
        dtype = str(df[col].dtypes)
        num_unique_values = df[col].nunique(dropna=False)

        # Check if the column is a categorical type or has a limited number of unique values
        if dtype in ["category", "object", "bool"] or (num_unique_values <= max_unique_values):
            categorical_cols.append(col)

    return categorical_cols


#-------------------------------------------------------------------------------------------
def binary_columns_find(df):
    binary_cols = []
    for col in df.columns:
      
        unique_values = df[col].dropna().unique()
 
        num_unique_non_null = len(unique_values)
        
        has_nan = df[col].isnull().any()

        if (num_unique_non_null == 1 and has_nan) or num_unique_non_null == 2:
            binary_cols.append(col)

    return binary_cols

#----------------------------------------------------------------------------------
def find_outlier_columns(df, q1_value, q2_value):
    """
    Identify columns in the DataFrame that contain outliers.
    :param df: A pandas DataFrame.
    :return: A list of column names that have outliers.
    """
    outlier_columns = []
    numerical_cols = df.select_dtypes(include=['number']).columns
    # q1_value=0.25
    # q2_value=0.75
    for col in numerical_cols:
        Q1 = df[col].quantile(q1_value)
        Q3 = df[col].quantile(q2_value)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        if ((df[col] < lower_bound) | (df[col] > upper_bound)).any():
            outlier_columns.append(col)
    return outlier_columns


        
#-----------------------------------------------------------------------------------------------------------------------------
def transform_and_test_stationarity(series, diff_lag=1):
    time_series_log_diff = series.diff(periods=diff_lag).dropna()
    print(f"\nAfter Logarithmic Transformation and Differencing (lag={diff_lag}):")
    check_stationarity(time_series_log_diff)
    return time_series_log_diff.tail()
#------------------------------------------------------------------------------------------------------------------------------------
def check_stationarity(df):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(df)
    test_statistic = result[0]
    p_value = result[1]

    print('ADF Statistic: %f' % test_statistic)
    print('p-value: %f' % p_value)

    if p_value < 0.05:
        print('The time series is stationary.')
    else:
        print('The time series is not stationary.')  
#-----------------------------------------------------------------------------------------------------------------------------------------

def find_best_parameters(train, test, seasonal_periods_options=[ 4, 7, 12], damped_trend_options=[True, False], trend_option='add'):
    from statsmodels.tsa.api import ExponentialSmoothing
    from sklearn.metrics import mean_squared_error
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    import numpy as np
    import warnings
    best_rmse = np.inf
    best_params = {
        'smoothing_level': None, 
        'smoothing_slope': None, 
        'smoothing_seasonal': None,
        'seasonal_periods': None,
        'damped_trend': None
    }
    smoothing_level_options = np.linspace(0.01, 1, 10)
    smoothing_slope_options = np.linspace(0.01, 1, 10)
    smoothing_seasonal_options = np.linspace(0.01, 1, 10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for seasonal_periods in seasonal_periods_options:
            for damped_trend in damped_trend_options:
                for alpha in smoothing_level_options:
                    for beta in smoothing_slope_options:
                        for gamma in smoothing_seasonal_options:
                            try:
                                model = ExponentialSmoothing(
                                    train, 
                                    trend=trend_option, 
                                    seasonal='add', 
                                    seasonal_periods=seasonal_periods,  
                                    damped_trend=damped_trend
                                ).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)

                                predictions = model.forecast(len(test))
                                rmse = np.sqrt(mean_squared_error(test, predictions))

                                if rmse < best_rmse:
                                    best_rmse = rmse
                                    best_params = {
                                        'smoothing_level': alpha, 
                                        'smoothing_slope': beta, 
                                        'smoothing_seasonal': gamma,
                                        'seasonal_periods': seasonal_periods,
                                        'damped_trend': damped_trend,
                                        'Best RMSE': best_rmse
                                    }
                            except Exception as e:
                                continue
    return best_params 