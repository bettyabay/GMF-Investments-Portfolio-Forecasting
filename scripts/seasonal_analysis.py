import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

class SeasonalAnalysis:
    """
    This class is designed to perform seasonal analysis on time series data, 
    such as stock prices. For example, let's consider analyzing Tesla's stock 
    price (TSLA). The class provides methods to test for stationarity, 
    difference the series if necessary, and decompose the time series into 
    its trend, seasonal, and residual components.
    """
    
    def __init__(self, logger=None):
        self.logger = logger
    
    def _log_error(self, message):
        """Logs error messages to the log file."""
        if self.logger:
            self.logger.error(message)
        print(f"Error: {message}")
    # The following code explains the decomposition of the time series into its components.
    # The decomposition is performed using the seasonal_decompose function from statsmodels.
    
    # The 'decomposition' object contains three main components:
    # 1. Trend: This component represents the long-term progression of the series.
    # 2. Seasonal: This component captures the repeating patterns or cycles in the data.
    # 3. Residual: This component is the remainder of the series after removing the trend and seasonal components.
    
    # We will print and plot each of these components for better understanding.
    
    def adf_test(self, series):
        """Perform ADF test for stationarity."""
        adf_result = adfuller(series.dropna())
        return adf_result[1]  # Return p-value of ADF test
    
    def difference_series(self, series):
        """Apply differencing to the series to make it stationary."""
        return series.diff().dropna()
    
    def decompose_series(self, series, model='addictive'):
        """Decompose the time series into trend, seasonal, and residual components."""
        decomposition = seasonal_decompose(series.dropna(), model=model, period=252)  # Assuming daily data, 252 trading days in a year
        return decomposition
    
    def analyze_trends_and_seasonality(self, data_dict, threshold=0.05):
        """Analyze seasonality and trends of stock price by decomposing it."""
        sns.set(style="whitegrid")

        for symbol, df in data_dict.items():
            try:
                if df is None or df.empty:
                    self._log_error(f"DataFrame for {symbol} is empty.")
                    continue

                # Perform ADF test for stationarity
                p_value = self.adf_test(df['Close'])
                print(f"ADF test p-value for {symbol}: {p_value}")

                # If non-stationary, apply differencing
                if p_value > threshold:
                    print(f"{symbol} series is non-stationary. Differencing the series.")
                    df['Close'] = self.difference_series(df['Close'])
                
                # Ensure series has at least 504 points post-differencing
                if len(df['Close'].dropna()) < 504:
                    self._log_error(f"Series length insufficient for decomposition; requires at least 504 observations.")
                    continue

                # Decompose the series into trend, seasonal, and residual components
                decomposition = self.decompose_series(df['Close'])
                
                # Plot decomposition
                plt.figure(figsize=(12, 8))
                plt.subplot(411)
                plt.plot(df['Close'], label=f'{symbol} Closing Price')
                plt.title(f'{symbol} Closing Price')
                plt.legend(loc='best')

                plt.subplot(412)
                plt.plot(decomposition.trend, label=f'{symbol} Trend', color='orange')
                plt.title(f'{symbol} Trend')
                plt.legend(loc='best')

                plt.subplot(413)
                plt.plot(decomposition.seasonal, label=f'{symbol} Seasonal', color='green')
                plt.title(f'{symbol} Seasonal')
                plt.legend(loc='best')

                plt.subplot(414)
                plt.plot(decomposition.resid, label=f'{symbol} Residual', color='red')
                plt.title(f'{symbol} Residual')
                plt.legend(loc='best')

                plt.tight_layout()
                plt.show()

            except Exception as e:
                self._log_error(f"Error analyzing {symbol}: {str(e)}")
