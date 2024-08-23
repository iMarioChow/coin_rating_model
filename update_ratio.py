import pandas as pd

# Load the CSV file
file_path = '/Users/cyh/Desktop/portfolio_manager/tokeninsight/modified_coin_rating.csv'
df = pd.read_csv(file_path)

# Step 0: Remove rows where token_performance is empty and rating_level is 'CCC'
df = df[~((df['token_performance'].isna()) & (df['rating_level'] == 'CCC'))]

# Step 1: Replace empty token_performance with 'medium'
df['token_performance'].fillna('medium', inplace=True)


# Calculate the new rating_score
df['rating_score'] = df[['token_performance', 'team_partners_investors', 'token_economics', 'roadmap_progress']].mean(axis=1)


# Save the updated DataFrame back to the CSV file
df.to_csv(file_path, index=False)

print(f"CSV file updated and saved to {file_path}")
