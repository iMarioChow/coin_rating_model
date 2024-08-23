import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import os

class BuyAndHold(Strategy):
    def init(self):
        pass
    
    def next(self):
        if not self.position:
            self.buy()

def clean_data(df):
    # Ensure we have all necessary columns
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Missing one or more required columns (Date, Open, High, Low, Close)")

    # Convert Date to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Drop rows with NaT in Date column
    df = df.dropna(subset=['Date'])

    # Set Date as index
    df.set_index('Date', inplace=True)

    # Sort index
    df.sort_index(inplace=True)

    # Forward fill NaN values in OHLC columns
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].ffill()

    # Drop any remaining rows with NaN values in OHLC
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    # Ensure High is the highest value
    df.loc[:, 'High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)

    # Ensure Low is the lowest value
    df.loc[:, 'Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)

    # Remove any rows where all OHLC values are identical (which can cause issues)
    df = df[~(df[['Open', 'High', 'Low', 'Close']] == df[['Open', 'High', 'Low', 'Close']].iloc[0]).all(axis=1)]

    return df

# Load existing backtest results
results_file = "/Users/cyh/Desktop/coin_rating_model/data/backtest_results.csv"
results_df = pd.read_csv(results_file)

# Directory containing individual CSV files for each symbol
data_dir = "/Users/cyh/Desktop/coin_rating_model/data/data"

# Function to run backtest and return all required metrics
def get_backtest_metrics(symbol):
    file_path = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(file_path):
        print(f"File not found for symbol: {symbol}")
        return {
            'Equity Peak [$]': np.nan,
            'Return [%]': np.nan,
            'Volatility (Ann.) [%]': np.nan,
            'Sharpe Ratio': np.nan,
            'Duration': np.nan,
            'Calmar Ratio': np.nan,
            'Max. Drawdown [%]': np.nan,
            'Initial Price': np.nan,
            'Equity Final [$]': np.nan
        }
    
    df = pd.read_csv(file_path)
    df = clean_data(df)
    
    if len(df) < 2:
        print(f"Not enough data points after cleaning for symbol: {symbol}")
        return {
            'Equity Peak [$]': np.nan,
            'Return [%]': np.nan,
            'Volatility (Ann.) [%]': np.nan,
            'Sharpe Ratio': np.nan,
            'Duration': np.nan,
            'Calmar Ratio': np.nan,
            'Max. Drawdown [%]': np.nan,
            'Initial Price': np.nan,
            'Equity Final [$]': np.nan
        }
    
    bt = Backtest(df, BuyAndHold, cash=100000)
    stats = bt.run()
    
    return {
        'Equity Peak [$]': stats['Equity Peak [$]'],
        'Return [%]': stats['Return [%]'],
        'Volatility (Ann.) [%]': stats['Volatility (Ann.) [%]'],
        'Sharpe Ratio': stats['Sharpe Ratio'],
        'Duration': (stats['End'] - stats['Start']).days,
        'Calmar Ratio': stats['Calmar Ratio'],
        'Max. Drawdown [%]': stats['Max. Drawdown [%]'],
        'Initial Price': df['Close'].iloc[0],
        'Equity Final [$]': stats['Equity Final [$]']
    }

# Apply the function to each symbol and update the DataFrame
metrics = results_df['Symbol'].apply(get_backtest_metrics)
metrics_df = pd.DataFrame(list(metrics))

# Update the original DataFrame with the new metrics
updated_results_df = pd.concat([results_df[['Symbol']], metrics_df], axis=1)

# Save updated results
updated_results_df.to_csv(results_file, index=False)

print(f"Updated {results_file} with required columns")
print(updated_results_df.head())
