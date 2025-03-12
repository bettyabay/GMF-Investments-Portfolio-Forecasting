import pandas as pd  # Imports the pandas library for data manipulation and analysis.
import numpy as np  # Imports the NumPy library for numerical operations.
import joblib  # Imports joblib for saving and loading Python objects efficiently.
from tensorflow.keras.models import load_model  # Imports load_model to load Keras models.
from sklearn.preprocessing import MinMaxScaler  # Imports MinMaxScaler for scaling data.
import matplotlib.pyplot as plt  # Imports matplotlib for plotting graphs.
import matplotlib.dates as mdates  # Imports date handling functions for matplotlib.
import seaborn as sns  # Imports seaborn for enhanced data visualization.

class ModelForecaster:  # Defines the ModelForecaster class.
    def __init__(self, historical_csv, forecast_csv, logger=None, column='Close'):
        """
        Initialize the model forecaster.

        Parameters:
        historical_csv (str): Path to the historical data CSV file.
        forecast_csv (str): Path to the forecast data CSV file.
        logger (logging.Logger): Logger instance for logging information.
        column (str): The column to forecast.
        """
        self.historical_data = historical_csv  # Stores the path or DataFrame for historical data.
        self.forecast_csv = forecast_csv  # Stores the path for the forecast data.
        self.column = column  # Stores the name of the column to forecast.
        self.logger = logger  # Initializes the logger instance.
        self.predictions = {}  # Initializes an empty dictionary to store predictions.

        # Load historical and forecast data
        self._load_data()  # Calls the method to load data from CSV files.

    def _load_data(self):
        """
        Load historical and forecast data from CSV files.
        """
        try:
            # Load historical data from a CSV file if the input is a string
            if isinstance(self.historical_data, str):
                self.historical_data = pd.read_csv(self.historical_data, index_col=0, parse_dates=True)

            # Ensure the historical data contains the specified column
            if self.column not in self.historical_data.columns:
                raise ValueError(f"Historical data must have a '{self.column}' column.")

            # Load forecast data from the provided CSV file
            self.forecast_data = pd.read_csv(self.forecast_csv, index_col=0, parse_dates=True)

            # Ensure the forecast data contains a 'forecast' column
            if 'forecast' not in self.forecast_data.columns:
                raise ValueError("Forecast CSV must have a 'forecast' column.")

            # Extract forecast values into the predictions dictionary
            self.predictions['forecast'] = self.forecast_data['forecast'].values
            self.forecast_dates = self.forecast_data.index  # Store the dates of the forecast data.

            self.logger.info("Historical and forecast data loaded successfully.")  # Log successful data loading.
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")  # Log any errors encountered during data loading.
            raise ValueError("Error loading data")  # Raise an error if data loading fails.

    def plot_forecast(self):
        """
        Plot the historical data alongside the forecast data with confidence intervals.
        """
        try:
            # Extract historical and forecast dates
            historical_dates = self.historical_data.index
            forecast_dates = self.forecast_dates

            # Set up the plot dimensions
            plt.figure(figsize=(15, 8))

            # Plot historical data
            plt.plot(historical_dates, self.historical_data[self.column], label='Actual', color='blue', linewidth=2)

            # Plot forecast data
            forecast = self.predictions['forecast']  # Retrieve forecast values for plotting.
            plt.plot(forecast_dates, forecast, label='Forecast', linestyle='--', color='red')

            # Plot confidence intervals
            plt.fill_between(
                forecast_dates,
                self.forecast_data['conf_lower'],  # Lower bound of confidence interval.
                self.forecast_data['conf_upper'],  # Upper bound of confidence interval.
                color='red', alpha=0.25, label='95% Confidence Interval'  # Fill color and transparency.
            )

            # Set up labels and title for the plot
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability.
            plt.title("Historical vs. Forecast Data with Confidence Intervals", fontsize=16)  # Set plot title.
            plt.xlabel("Date", fontsize=14)  # Set x-axis label.
            plt.ylabel(self.column, fontsize=14)  # Set y-axis label based on the forecasted column.
            plt.legend(loc='best')  # Display legend in the best location.
            sns.set(style="whitegrid")  # Set seaborn style for the plot.
            plt.tight_layout()  # Automatically adjust subplot parameters for better fit.
            plt.show()  # Display the plot.

        except Exception as e:
            self.logger.error(f"Error in plotting forecasts: {e}")  # Log any errors encountered during plotting.
            raise ValueError("Error plotting forecasts")  # Raise an error if plotting fails.

    def analyze_forecast(self, threshold=0.05):
        """
        Analyze and interpret the forecast results, including trend, volatility, and market opportunities/risk.
        """
        analysis_results = {}  # Initialize a dictionary to store analysis results.

        self.logger.info("Starting forecast analysis.")  # Log the start of the analysis.
        
        for model_name, forecast in self.predictions.items():  # Iterate over each model's forecast.
            # Trend Analysis
            trend = "upward" if np.mean(np.diff(forecast)) > 0 else "downward"  # Determine trend direction.
            trend_magnitude = np.max(np.diff(forecast))  # Calculate the magnitude of the trend.
            self.logger.info(f"{model_name} forecast shows a {trend} trend.")  # Log the trend information.

            # Volatility and Risk Analysis
            volatility = np.std(forecast)  # Calculate the standard deviation (volatility) of the forecast.
            volatility_level = "High" if volatility > threshold else "Low"  # Set volatility level based on threshold.
            max_price = np.max(forecast)  # Find the maximum forecasted price.
            min_price = np.min(forecast)  # Find the minimum forecasted price.
            price_range = max_price - min_price  # Calculate the price range.

            # Perform additional volatility and risk analysis
            volatility_analysis = self._volatility_risk_analysis(forecast, threshold)

            # Assess market opportunities and risks
            opportunities_risks = self._market_opportunities_risks(trend, volatility_level)
            
            # Store results in the analysis dictionary
            analysis_results[model_name] = {
                'Trend': trend,
                'Trend_Magnitude': trend_magnitude,
                'Volatility': volatility,
                'Volatility_Level': volatility_level,
                'Max_Price': max_price,
                'Min_Price': min_price,
                'Price_Range': price_range
            }
            print(f"  Volatility and Risk: {volatility_analysis}")  # Print volatility analysis results.
            print(f"  Market Opportunities/Risks: {opportunities_risks}")  # Print market opportunities and risks.
            
            # Log the detailed analysis
            self.logger.info(f"{model_name} Analysis Results:")
            self.logger.info(f"  Trend: {trend}")  # Log the trend.
            self.logger.info(f"  Trend Magnitude: {trend_magnitude:.2f}")  # Log trend magnitude.
            self.logger.info(f"  Volatility: {volatility:.2f}")  # Log volatility.
            self.logger.info(f"  Volatility Level: {volatility_level}")  # Log volatility level.
            self.logger.info(f"  Max Price: {max_price:.2f}")  # Log maximum price.
            self.logger.info(f"  Min Price: {min_price:.2f}")  # Log minimum price.
            self.logger.info(f"  Price Range: {price_range:.2f}")  # Log price range.
            self.logger.info(f"  Volatility and Risk: {volatility_analysis}")  # Log volatility analysis.
            self.logger.info(f"  Market Opportunities/Risks: {opportunities_risks}")  # Log opportunities and risks.
        
        # Return the results in a DataFrame for easy viewing
        analysis_df = pd.DataFrame(analysis_results).T  # Convert analysis results to a DataFrame and transpose.
        return analysis_df  # Return the analysis DataFrame.

    def _volatility_risk_analysis(self, forecast, threshold):
        """
        Analyze the volatility and risk based on forecast data.
        """
        volatility = np.std(forecast)  # Calculate the standard deviation of the forecast data.
        volatility_level = "High" if volatility > threshold else "Low"  # Determine the volatility level.

        # Highlight periods of increasing volatility
        increasing_volatility = any(np.diff(forecast) > np.mean(np.diff(forecast)))  # Check for increasing volatility.
        
        if increasing_volatility:
            return f"Potential increase in volatility, which could lead to market risk."  # Return risk assessment.
        else:
            return f"Stable volatility, lower risk."  # Return stability assessment.

    def _market_opportunities_risks(self, trend, volatility_level):
        """
        Identify market opportunities or risks based on forecast trends and volatility.
        """
        if trend == "upward":  # If the trend is upward
            if volatility_level == "High":
                return "Opportunity with high risk due to increased volatility."  # High risk opportunity.
            else:
                return "Opportunity with moderate risk due to stable volatility."  # Moderate risk opportunity.
        elif trend == "downward":  # If the trend is downward
            if volatility_level == "High":
                return "Risk of decline with high uncertainty."  # High uncertainty risk.
            else:
                return "Moderate risk of decline with low volatility."  # Moderate risk with stability.
        else:
            return "Stable market, with minimal risks."  # Return a stable market assessment.