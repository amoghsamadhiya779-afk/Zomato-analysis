import pandas as pd

# Load data
meta = pd.read_csv('C:\Users\Lenovo\Desktop\Projects\Zomato\data\Zomato Restaurant names and Metadata.csv')
reviews = pd.read_csv('C:\Users\Lenovo\Desktop\Projects\Zomato\data\Zomato Restaurant reviews.csv')

# Merge on 'Name' or 'Restaurant' column
# Note: Check column names in your CSV first!
df = pd.merge(reviews, meta, left_on='Restaurant', right_on='Name')

# Save a sample to processed folder
df.to_csv('data/processed/merged_data.csv', index=False)
print("Data merged and saved!")