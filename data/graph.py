import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file
df = pd.read_csv('/Users/cyh/Desktop/HMM/buy_and_hold_stats.csv')

# Find extreme Sharpe ratio values
max_sharpe = df.loc[df['Sharpe Ratio'].idxmax()]
min_sharpe = df.loc[df['Sharpe Ratio'].idxmin()]

print("Cryptocurrency with highest Sharpe ratio:")
print(f"Symbol: {max_sharpe['Symbol']}, Sharpe Ratio: {max_sharpe['Sharpe Ratio']:.2f}")

print("\nCryptocurrency with lowest Sharpe ratio:")
print(f"Symbol: {min_sharpe['Symbol']}, Sharpe Ratio: {min_sharpe['Sharpe Ratio']:.2f}")

# Remove custom columns
columns_to_remove = ['Custom Sharpe Ratio', 'Custom Sortino Ratio', 'Custom Calmar Ratio']
df_cleaned = df.drop(columns=columns_to_remove)

# Function to identify outliers
def identify_outliers(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return data[(data < lower_bound) | (data > upper_bound)]

# Create individual graphs and identify outliers for each numeric column
numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns

for column in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_cleaned[column])
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
    plt.show()
    
    # Identify and print outliers
    outliers = identify_outliers(df_cleaned[column])
    if not outliers.empty:
        print(f"\nOutliers for {column}:")
        for symbol, value in zip(df_cleaned.loc[outliers.index, 'Symbol'], outliers):
            print(f"{symbol}: {value:.2f}")
    else:
        print(f"\nNo outliers found for {column}")