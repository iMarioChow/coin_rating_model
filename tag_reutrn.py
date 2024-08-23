import pandas as pd
import os
from collections import defaultdict

# Load daily tag returns
daily_tag_returns = pd.read_csv('/Users/cyh/Desktop/coin_rating_model/daily_tag_returns.csv', index_col='Date', parse_dates=True)

# Load symbol tags
symbol_tags = pd.read_csv('/Users/cyh/Desktop/coin_rating_model/data/coin_rating_final_adjusted_tags.csv')
symbol_tags_dict = dict(zip(symbol_tags['Symbol'], symbol_tags['tags'].str.split(',')))

# Directory containing symbol data
data_dir = '/Users/cyh/Desktop/coin_rating_model/data/data'

def calculate_tag_outperformance(symbol_data, tags):
    outperformance = {tag: 0 for tag in tags}
    total_days = {tag: 0 for tag in tags}
    
    for date, row in symbol_data.iterrows():
        if date in daily_tag_returns.index:
            for tag in tags:
                if tag in daily_tag_returns.columns:
                    tag_return = daily_tag_returns.loc[date, tag]
                    if pd.notna(tag_return) and pd.notna(row['Daily Return']):
                        outperformance[tag] += (row['Daily Return'] > tag_return)
                        total_days[tag] += 1
    
    return {tag: (count / total_days[tag] * 100 if total_days[tag] > 0 else 0) 
            for tag, count in outperformance.items()}

results = {}

for symbol, tags in symbol_tags_dict.items():
    file_path = os.path.join(data_dir, f"{symbol}.csv")
    if os.path.exists(file_path):
        symbol_data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        if 'Daily Return' not in symbol_data.columns:
            symbol_data['Daily Return'] = (symbol_data['Close'] - symbol_data['Open']) / symbol_data['Open']
        
        outperformance = calculate_tag_outperformance(symbol_data, tags)
        results[symbol] = outperformance
        print(f"Processed {symbol}")
    else:
        print(f"File not found for {symbol}")

# Prepare results for output
output_data = []
for symbol, tag_performance in results.items():
    for tag, percentage in tag_performance.items():
        output_data.append({
            'Symbol': symbol,
            'Tag': tag,
            'Outperformance Percentage': percentage
        })

output_df = pd.DataFrame(output_data)

# Calculate average outperformance for each tag
tag_avg_performance = output_df.groupby('Tag')['Outperformance Percentage'].mean().sort_values(ascending=False)

print("\nAverage Tag Outperformance:")
print(tag_avg_performance)

# Save detailed results
output_df.to_csv('/Users/cyh/Desktop/coin_rating_model/symbol_specific_tag_performance.csv', index=False)
print("\nDetailed results saved to symbol_specific_tag_performance.csv")

# Print top performing symbols for each tag
print("\nTop 5 Performing Symbols for Each Tag:")
for tag in tag_avg_performance.index:
    top_symbols = output_df[output_df['Tag'] == tag].nlargest(5, 'Outperformance Percentage')
    print(f"\n{tag}:")
    for _, row in top_symbols.iterrows():
        print(f"  {row['Symbol']}: {row['Outperformance Percentage']:.2f}%")

# Print symbols with highest overall outperformance
print("\nTop 10 Symbols with Highest Average Outperformance Across Their Tags:")
symbol_avg_performance = output_df.groupby('Symbol')['Outperformance Percentage'].mean().sort_values(ascending=False)
for symbol, avg_performance in symbol_avg_performance.head(10).items():
    tags = symbol_tags_dict[symbol]
    print(f"{symbol} (Tags: {', '.join(tags)}): {avg_performance:.2f}%")