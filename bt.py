import os
import pandas as pd
from backtesting import Backtest, Strategy

class BuyAndHold(Strategy):
    def init(self):
        pass
    
    def next(self):
        if not self.position:
            self.buy()

# Load existing backtest results
file_path = '/Users/cyh/Desktop/coin_rating_model/data/updated_backtest_results.csv'
existing_results = pd.read_csv(file_path)

# Get list of all CSV files in the data folder
data_folder = '/Users/cyh/Desktop/coin_rating_model/data/data'
all_csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
all_symbols = set([f.split('.')[0] for f in all_csv_files])

# Function to run the backtest
def run_backtest(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    bt = Backtest(df, BuyAndHold, cash=100000)
    stats = bt.run()
    
    return stats['Return [%]']

# Update the "Return [%]" column
for symbol in all_symbols:
    csv_file_path = os.path.join(data_folder, f"{symbol}.csv")
    try:
        new_return = run_backtest(csv_file_path)
        existing_results.loc[existing_results['Symbol'] == symbol, 'Return [%]'] = new_return
        print(f"Updated 'Return [%]' for {symbol}")
    except Exception as e:
        print(f"Error backtesting {symbol}: {str(e)}")

# Save the updated results to the CSV file
existing_results.to_csv(file_path, index=False)

print(f"\nUpdated 'Return [%]' in {file_path}")
