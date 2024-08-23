import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Load and preprocess data
df = pd.read_csv('/Users/cyh/Desktop/coin_rating_model/data/backtest_results.csv')

# Calculate High/Current ratio using Equity Peak [$] and Equity Final [$]
df['High_Current_Ratio'] = df['Equity Peak [$]'] / df['Equity Final [$]']

# Invert High_Current_Ratio so that higher is better (closer to ATH)
df['Inverted_High_Current_Ratio'] = 1 / df['High_Current_Ratio']

# Custom scaling function for Return [%]
def scale_return(x):
    if x <= 0:
        return 0
    elif x <= 100:
        return x / 100
    elif x <= 1000:
        return 1 + (x - 100) / 900
    else:
        return 2 + np.log10(x / 1000)

# Custom scaling function for Sharpe Ratio
def scale_sharpe(x):
    if x <= 0:
        return 0
    elif x <= 1:
        return x
    elif x <= 3:
        return 1 + (x - 1) / 2
    else:
        return 2 + np.log10(x / 3)


# Apply custom scaling
df['Scaled_Return'] = df['Return [%]'].apply(scale_return)
df['Scaled_Sharpe'] = df['Sharpe Ratio'].apply(scale_sharpe)

# Define features, using scaled versions
features = ['Scaled_Return', 'Scaled_Sharpe', 'Max. Drawdown [%]', 'Duration', 'Inverted_High_Current_Ratio']

# Print unnormalized statistics
print("\nUnnormalized Statistics:")
print(df[features].describe())

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df[features]), columns=features)

# Use StandardScaler for other features
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df_imputed), columns=features)

print("\nStandardized Statistics:")
print(df_standardized.describe())

# Perform PCA on standardized features
pca = PCA()
pca.fit(df_standardized)

# Get feature weights from the first principal component
feature_weights = np.abs(pca.components_[0])
feature_weights = feature_weights / np.sum(feature_weights)  # Normalize to sum to 1

# Adjust weights to focus more on performance metrics and less on duration
performance_boost = 0.15
duration_reduction = 0.3

for i, feature in enumerate(features):
    if feature == 'Duration':
        feature_weights[i] -= duration_reduction
    elif feature in ['Scaled_Return', 'Scaled_Sharpe']:
        feature_weights[i] += performance_boost / 3
    elif feature == 'Inverted_High_Current_Ratio':
        feature_weights[i] += 0.05  # Slight boost to Inverted_High_Current_Ratio

# Ensure minimum weight of 5%
min_weight = 0.05
excess_weight = np.sum(np.maximum(feature_weights - min_weight, 0))
feature_weights = np.maximum(feature_weights, min_weight)
feature_weights += excess_weight / len(feature_weights)
feature_weights = feature_weights / np.sum(feature_weights)  # Renormalize to sum to 1

# Create a dictionary of feature weights
weight_dict = dict(zip(features, feature_weights))

print("\nFeature weights after adjustment:")
for feature, weight in weight_dict.items():
    print(f"{feature}: {weight:.4f}")

# Calculate weighted composite score using adjusted weights
df_standardized['Composite_Score'] = np.sum(df_standardized * feature_weights, axis=1)

# Transform composite score to a uniform distribution
def to_uniform_distribution(scores, min_rating=20, max_rating=100):
    percentile_ranks = pd.Series(scores).rank(pct=True)
    return percentile_ranks * (max_rating - min_rating) + min_rating

df_standardized['Rating'] = to_uniform_distribution(df_standardized['Composite_Score'])

# Add Risk Indicator based on Volatility
df_standardized['Risk'] = pd.qcut(df['Volatility (Ann.) [%]'], q=[0, 0.4, 0.7, 1], labels=['Low', 'Medium', 'High'])

# Add original Symbol and Return [%] columns back to df_standardized
df_standardized['Symbol'] = df['Symbol']
df_standardized['Original Return [%]'] = df['Return [%]']

# Sort by Rating
df_sorted = df_standardized.sort_values('Rating', ascending=False)

print("\nTop Rated Tokens:")
print(df_sorted[['Symbol', 'Rating', 'Risk', 'Composite_Score', 'Original Return [%]'] + features].head(20))

print("\nRating Distribution:")
print(df_sorted['Rating'].describe())

print("\nRisk Distribution:")
print(df_sorted['Risk'].value_counts(normalize=True))

# Save the results
df_sorted[['Symbol', 'Rating', 'Risk', 'Composite_Score', 'Original Return [%]'] + features].to_csv('token_ratings_extreme_return_handled.csv', index=False)
print("\nResults saved to 'token_ratings_extreme_return_handled.csv'")

print("\nNew Rating Distribution:")
print(df_sorted['Rating'].describe())

# Print the number of tokens in each rating range
print("\nNumber of tokens in each rating range:")
print(pd.cut(df_sorted['Rating'], bins=[20, 40, 60, 80, 100], labels=['20-40', '40-60', '60-80', '80-100']).value_counts().sort_index())