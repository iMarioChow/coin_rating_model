import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Load the refined CSV file
file_path = '/Users/cyh/Desktop/coin_rating_model/data/coin_rating_final_adjusted_tags.csv'
df = pd.read_csv(file_path)

def adjust_tags(tags):
    if pd.isna(tags):
        return tags
    
    tag_list = tags.split(',')
    new_tags = []
    
    for tag in tag_list:
        tag = tag.strip()
        if tag not in ['Curve', 'Sharding', 'Synthetics', 'Metaverse']:
            if tag == 'Rollups':
                if 'Layer 2' not in new_tags:
                    new_tags.append('Layer 2')
            elif tag == 'Mining':
                if 'PoW' not in new_tags:
                    new_tags.append('PoW')
            elif tag not in new_tags:
                new_tags.append(tag)
    
    return ','.join(new_tags)

# Apply the adjustments
df['adjusted_tags'] = df['tags'].apply(adjust_tags)

# Function to count tags
def count_tags(tags):
    return len(tags.split(',')) if pd.notna(tags) else 0

# Apply the counting function to the 'adjusted_tags' column
df['tag_count'] = df['adjusted_tags'].apply(count_tags)

# Calculate statistics
total_tags = df['tag_count'].sum()
unique_tags = len(set(','.join(df['adjusted_tags'].dropna()).split(',')))
avg_tags_per_symbol = df['tag_count'].mean()
median_tags_per_symbol = df['tag_count'].median()
max_tags = df['tag_count'].max()

print("Tag Statistics:")
print(f"Total number of tags (including duplicates): {total_tags}")
print(f"Number of unique tags: {unique_tags}")
print(f"Average tags per symbol: {avg_tags_per_symbol:.2f}")
print(f"Median tags per symbol: {median_tags_per_symbol}")
print(f"Maximum tags for a symbol: {max_tags}")

# Plot histogram of tag counts
plt.figure(figsize=(10, 6))
df['tag_count'].hist(bins=range(max_tags + 2))
plt.title('Distribution of Tag Counts per Symbol')
plt.xlabel('Number of Tags')
plt.ylabel('Number of Symbols')
plt.savefig('final_adjusted_tag_count_distribution.png')
plt.close()

# Print symbols with the most tags
print("\nSymbols with the most tags:")
print(df.nlargest(10, 'tag_count')[['Symbol', 'tag_count', 'adjusted_tags']])

# Calculate tag frequencies
all_tags = ','.join(df['adjusted_tags'].dropna()).split(',')
tag_freq = Counter(all_tags)

print("\nAll tags and their frequencies:")
for tag, count in sorted(tag_freq.items(), key=lambda x: x[1], reverse=True):
    print(f"{tag}: {count}")

# Analyze co-occurrence of tags
def get_tag_pairs(tags):
    tags = tags.split(',')
    return [(a, b) for i, a in enumerate(tags) for b in tags[i + 1:]]

all_pairs = [pair for tags in df['adjusted_tags'].dropna() for pair in get_tag_pairs(tags)]
pair_freq = Counter(all_pairs)

print("\nTop 10 most common tag pairs:")
for pair, count in pair_freq.most_common(10):
    print(f"{pair}: {count}")

# Calculate the percentage of symbols with each number of tags
tag_count_percentages = df['tag_count'].value_counts(normalize=True).sort_index() * 100

print("\nPercentage of symbols with each number of tags:")
for count, percentage in tag_count_percentages.items():
    print(f"{count} tag(s): {percentage:.2f}%")

# Analyze relationship between tags and rating_score
print("\nAverage rating score by tag:")
for tag in tag_freq.keys():
    avg_score = df[df['adjusted_tags'].str.contains(tag, na=False)]['rating_score'].mean()
    print(f"{tag}: {avg_score:.2f}")

# Print symbols and their tags for the least frequent tags
print("\nSymbols with the least frequent tags (lowest 20 or all if less than 20):")
rare_tags = sorted(tag_freq.items(), key=lambda x: x[1])[:20]
for tag, count in rare_tags:
    symbols = df[df['adjusted_tags'].str.contains(tag, na=False)]
    print(f"\n{tag} (frequency: {count}):")
    for _, row in symbols.iterrows():
        print(f"  {row['Symbol']}: {row['adjusted_tags']}")

# Save the adjusted DataFrame
output_file_path = '/Users/cyh/Desktop/coin_rating_model/data/coin_rating_final_adjusted_tags.csv'
df[['tid', 'name', 'Symbol', 'rating_level', 'rating_score', 'underlying_technology_security', 
    'token_performance', 'ecosystem_development', 'team_partners_investors', 'token_economics', 
    'roadmap_progress', 'adjusted_tags', 'In_Stats']].to_csv(output_file_path, index=False)
print(f"\nAdjusted data saved to {output_file_path}")