import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('/Users/cyh/Desktop/portfolio_manager/tokeninsight/modified_coin_rating.csv')

# Remove rows where token_performance is empty
df = df.dropna(subset=['token_performance'])

# Convert percentage strings to floats
def percentage_to_float(value):
    if isinstance(value, str) and '%' in value:
        return float(value.strip('%'))  # Remove '%' but don't divide by 100
    return value

# Apply conversion to relevant columns
columns_to_convert = ['token_performance', 'team_partners_investors', 'token_economics', 'roadmap_progress']
for col in columns_to_convert:
    df[col] = df[col].apply(percentage_to_float)

# Function to calculate new rating score
def calculate_rating_score(row):
    return (1/4) * (row['token_performance'] + 
                    row['team_partners_investors'] + 
                    row['token_economics'] + 
                    row['roadmap_progress'])

# Apply the new rating score calculation
df['rating_score'] = df.apply(calculate_rating_score, axis=1)

# Save the updated DataFrame to a new CSV file
df.to_csv('/Users/cyh/Desktop/portfolio_manager/tokeninsight/updated_coin_rating.csv', index=False)

print(df.head())
print(f"Processed {len(df)} rows.")