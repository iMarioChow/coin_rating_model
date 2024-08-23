import os
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

class BuyAndHold(Strategy):
    def init(self):
        pass
    
    def next(self):
        if not self.position:
            self.buy()

def geometric_mean(returns: pd.Series) -> float:
    returns = returns.fillna(0) + 1
    if np.any(returns <= 0):
        return 0
    return np.exp(np.log(returns).sum() / (len(returns) or np.nan)) - 1

def compute_drawdown_duration_peaks(dd: pd.Series):
    iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])
    iloc = pd.Series(iloc, index=dd.index[iloc])
    df = iloc.to_frame('iloc').assign(prev=iloc.shift())
    df = df[df['iloc'] > df['prev'] + 1].astype(int)

    if not len(df):
        return (dd.replace(0, np.nan),) * 2

    df['duration'] = df['iloc'].map(dd.index.__getitem__) - df['prev'].map(dd.index.__getitem__)
    df['peak_dd'] = df.apply(lambda row: dd.iloc[row['prev']:row['iloc'] + 1].max(), axis=1)
    df = df.reindex(dd.index)
    return df['duration'], df['peak_dd']

def compute_stats(equity: np.ndarray, ohlc_data: pd.DataFrame, risk_free_rate: float = 0) -> dict:
    index = ohlc_data.index
    dd = 1 - equity / np.maximum.accumulate(equity)
    dd_dur, dd_peaks = compute_drawdown_duration_peaks(pd.Series(dd, index=index))
    
    equity_df = pd.DataFrame({
        'Equity': equity,
        'DrawdownPct': dd,
        'DrawdownDuration': dd_dur},
        index=index)
    
    day_returns = equity_df['Equity'].resample('D').last().dropna().pct_change()
    gmean_day_return = geometric_mean(day_returns)
    
    annual_trading_days = 365  # For cryptocurrency trading
    
    # Annualized return calculation
    annualized_return = (1 + gmean_day_return)**annual_trading_days - 1
    
    # Improved volatility calculation
    day_returns_var = day_returns.var(ddof=1)
    annualized_volatility = np.sqrt((day_returns_var + (1 + gmean_day_return)**2)**annual_trading_days - (1 + gmean_day_return)**(2*annual_trading_days))
    
    # Sharpe Ratio calculation
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
    
    # Sortino Ratio calculation
    negative_returns = day_returns[day_returns < 0]
    sortino_ratio = (annualized_return - risk_free_rate) / (np.sqrt(np.mean(negative_returns**2)) * np.sqrt(annual_trading_days))
    
    # Calmar Ratio calculation
    max_dd = dd.max()
    calmar_ratio = annualized_return / max_dd if max_dd != 0 else np.inf
    
    return {
        'Equity Peak [$]': equity.max(),
        'Return [%]': (equity[-1] - equity[0]) / equity[0] * 100,
        'Return (Ann.) [%]': annualized_return * 100,
        'Volatility (Ann.) [%]': annualized_volatility * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Max. Drawdown [%]': max_dd * 100,
        'Max. Drawdown Duration': dd_dur.max(),
        'Equity Final [$]': equity[-1]
    }

def run_backtest(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Handle NaN values by interpolating, then forward filling any remaining NaNs
    df.interpolate(method='linear', inplace=True)
    df.ffill(inplace=True)  # Forward fill remaining NaN values
    
    # Drop any rows where critical columns are still NaN after filling
    df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    
    if len(df) < 2:
        raise ValueError("Not enough data points after cleaning")
    
    bt = Backtest(df, BuyAndHold, cash=100000)
    stats = bt.run()
    
    custom_stats = compute_stats(stats._equity, df)
    
    return {
        'Symbol': os.path.basename(file_path).split('.')[0],
        'Equity Peak [$]': custom_stats['Equity Peak [$]'],
        'Return [%]': custom_stats['Return [%]'],
        'Volatility (Ann.) [%]': custom_stats['Volatility (Ann.) [%]'],
        'Sharpe Ratio': custom_stats['Sharpe Ratio'],
        'Duration': (stats['End'] - stats['Start']).days,  # Duration in days
        'Calmar Ratio': custom_stats['Calmar Ratio'],
        'Max. Drawdown [%]': custom_stats['Max. Drawdown [%]'],
        'Initial Price': '',  # Empty string for Initial Price
        'Equity Final [$]': custom_stats['Equity Final [$]']
    }

# Load existing backtest results
existing_results = pd.read_csv('/Users/cyh/Desktop/coin_rating_model/data/backtest_results.csv')
existing_symbols = set(existing_results['Symbol'])

# Get list of all CSV files in the data folder
data_folder = '/Users/cyh/Desktop/coin_rating_model/data/data'
all_csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
all_symbols = set([f.split('.')[0] for f in all_csv_files])

# Find symbols that need to be backtested
symbols_to_backtest = all_symbols - existing_symbols

# Run backtests for missing symbols
new_results = []
failed_symbols = []  # To track symbols that fail

for symbol in symbols_to_backtest:
    file_path = os.path.join(data_folder, f"{symbol}.csv")
    try:
        result = run_backtest(file_path)
        new_results.append(result)
        print(f"Backtest completed for {symbol}")
    except Exception as e:
        print(f"Error backtesting {symbol}: {str(e)}")
        failed_symbols.append(symbol)

# Convert new results to DataFrame
new_results_df = pd.DataFrame(new_results)

# Combine existing and new results
all_results = pd.concat([existing_results, new_results_df], ignore_index=True)

# Ensure 'Duration' is in numerical format for all results
all_results['Duration'] = pd.to_numeric(all_results['Duration'], errors='coerce')

# Save updated results
output_file = '/Users/cyh/Desktop/coin_rating_model/data/updated_backtest_results.csv'
all_results.to_csv(output_file, index=False)

# Optionally, save the list of failed symbols for review
if failed_symbols:
    failed_output_file = '/Users/cyh/Desktop/coin_rating_model/data/failed_symbols.csv'
    pd.DataFrame(failed_symbols, columns=['Symbol']).to_csv(failed_output_file, index=False)
    print(f"List of failed symbols saved to {failed_output_file}")

print(f"\nBacktesting complete. Updated results saved to {output_file}")
