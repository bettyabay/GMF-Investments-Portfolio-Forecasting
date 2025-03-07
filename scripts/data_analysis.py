import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# DataAnalysis class defined below handles data analysis and visualization tasks
class DataAnalysis:
    # - Initializing with data directory and logger
    def __init__(self, data_dir='../data/', logger=None):
        """
        Initializes the DataAnalysis class.

        Parameters:
        - data_dir (str): Directory containing the data files.
        - logger (logging.Logger, optional): Logger instance to log errors.
        """
        self.data_dir = data_dir
        self.logger = logger
    # - Error logging functionality
    def _log_error(self, message):
        """Logs error messages to the log file."""
        if self.logger:
            self.logger.error(message)
        print(f"Error: {message}")

    
    # - Plotting percentage changes in closing prices
    #   Example: For a stock ticker like 'AAPL':
    #   If AAPL closes at $100 on day 1 and $105 on day 2,
    #   The percentage change would be: ((105-100)/100) * 100 = 5%
    #   This method plots these daily % changes over time to show price volatility
    def plot_percentage_change(self, data_dict):
        """Plots the daily percentage change in the closing price."""
        for symbol, df in data_dict.items():
            
            if df is None or df.empty:
                self._log_error(f"DataFrame for {symbol} is empty.")
                return
            try:
                df['Pct_Change'] = df['Close'].pct_change() * 100
                plt.figure(figsize=(10, 6))
                plt.plot(df.index, df['Pct_Change'], label=f'{symbol} Daily Percentage Change')
                plt.title(f"{symbol} Daily Percentage Change Over Time")
                plt.xlabel("Date")
                plt.ylabel("Percentage Change (%)")
                plt.legend()
                plt.grid(True)
                plt.show()
            except Exception as e:
                self._log_error(f"Error plotting percentage change for {symbol}: {str(e)}")
                
    # - Analyzing price trends with rolling means and volatility
    def analyze_price_trend(self, data_dict, window_size=30):
        """Plots the closing price, rolling mean, and volatility (rolling std) over time for multiple symbols."""

        sns.set(style="whitegrid")  # Set Seaborn style for improved aesthetics

        for symbol, df in data_dict.items():
            try:
                # Data validation checks
                if df is None or df.empty:  # If AAPL's data is missing/empty
                    self._log_error(f"DataFrame for {symbol} is empty.")
                    continue

                if 'Close' not in df.columns:  # If closing price column is missing
                    self._log_error(f"'Close' column not found in DataFrame for {symbol}.")
                    continue

                # Calculate metrics:
                # For AAPL example:
                # - Rolling_Mean: Average of last 30 days' closing prices
                # - Rolling_Std: Standard deviation of last 30 days' prices (measures volatility)
                df['Rolling_Mean'] = df['Close'].rolling(window=window_size).mean()
                df['Rolling_Std'] = df['Close'].rolling(window=window_size).std()

                # Plotting each symbol in a separate figure for clarity
                plt.figure(figsize=(12, 6))
                
                # Plot three lines:
                # 1. Blue solid line: AAPL's actual closing prices
                sns.lineplot(data=df, x=df.index, y='Close', label=f'{symbol} Closing Price', color="blue", linestyle='solid')
                # 2. Orange dashed line: 30-day moving average (smoother trend line) - Rolling mean line
                sns.lineplot(data=df, x=df.index, y='Rolling_Mean', label=f'{symbol} {window_size}-day Rolling Mean', color="orange", linestyle="--")
                # 3. Green dotted line: 30-day volatility - Rolling standard deviation (volatility) line
                sns.lineplot(data=df, x=df.index, y='Rolling_Std', label=f'{symbol} {window_size}-day Rolling Volatility', color="green", linestyle=":")

                # Add title and labels to the plot
                plt.title(f"Closing Price Trend, Rolling Mean and Volatility of {symbol} Over Time", fontsize=16)
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Value", fontsize=12)
                plt.legend(title="Symbols")
                plt.grid(True)

                # Make y-axis more readable by setting tick marks every $50
                y_max = int(plt.ylim()[1])
                plt.yticks(range(0, y_max + 50, 50))  # Adjusts y-ticks by 50 increments

                # Tight layout for clarity
                plt.tight_layout()
                plt.show()  # Display the final plot

            except Exception as e:  # Handle any errors during plotting
                self._log_error(f"Error plotting data for {symbol}: {str(e)}")
        

    def plot_unusual_daily_return(self, data_dict, threshold=2.5):
        """
        Calculates and plots daily returns with highlights on unusually high or low return days for each symbol.

        Parameters:
        - data_dict (dict): Dictionary with stock symbols as keys and their DataFrames as values.
        - threshold (float): Threshold (in terms of standard deviations) to define unusual returns.
                           Default 2.5 means returns > ±2.5 standard deviations from mean are unusual
        """
        # Set plot style to have white background with grid
        sns.set(style="whitegrid")

        for symbol, df in data_dict.items():
            try:
                # Check if we have valid data for the symbol (e.g., AAPL)
                if df is None or df.empty:
                    print(f"DataFrame for {symbol} is empty.")
                    continue

                # Calculate daily percentage returns
                # For AAPL example: If price goes from $100 to $105, return = ((105-100)/100) * 100 = 5%
                df['Daily_Return'] = df['Close'].pct_change() * 100

                     # Determine unusual returns using the standard deviation threshold
                # Calculate statistics for unusual returns
                # Example for AAPL:
                # If mean return = 0.1% and std_dev = 1%
                # With threshold = 2.5, unusual returns would be any returns below -2.4% or above 2.6%
                mean_return = df['Daily_Return'].mean()  # e.g., 0.1%
                std_dev = df['Daily_Return'].std()       # e.g., 1%
                unusual_returns = df[(df['Daily_Return'] > mean_return + threshold * std_dev) |
                                     (df['Daily_Return'] < mean_return - threshold * std_dev)]

                # Plot daily returns
                plt.figure(figsize=(12, 6))
                
                # Plot all daily returns as a continuous line (e.g., AAPL returns over time)
                sns.lineplot(x=df.index, y=df['Daily_Return'], label=f'{symbol} Daily Return', color='skyblue')

                # Highlight unusual return days with red dots
                # For AAPL example: If there was a -3% return day, it would get a red dot
                plt.scatter(unusual_returns.index, unusual_returns['Daily_Return'], color='red', 
                            label=f"Unusual Returns (±{threshold}σ)", s=50, marker='o')

                # Plot styling
                plt.title(f"Daily Returns with Unusual Days Highlighted - {symbol}", fontsize=16)
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Daily Return (%)", fontsize=12)
                
                # Add horizontal line at 0% for reference
                plt.axhline(0, color='grey', linestyle='--')
                
                # Add legend, grid and adjust layout
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"Error plotting unusual daily returns for {symbol}: {str(e)}")
 

    