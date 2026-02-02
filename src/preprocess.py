import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords

# 1. Setup NLP Essentials
print("📥 Downloading NLP components...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('english'))

# 2. Define Paths
RAW_PATH = 'data/raw'
PROCESSED_PATH = 'data/processed'

# Ensure the processed folder exists
os.makedirs(PROCESSED_PATH, exist_ok=True)

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove noise
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])

def run_pipeline():
    try:
        # STEP A: Load Raw Data
        print("📖 Reading raw CSV files...")
        meta = pd.read_csv(f'{RAW_PATH}/Zomato Restaurant names and Metadata.csv')
        reviews = pd.read_csv(f'{RAW_PATH}/Zomato Restaurant reviews.csv')

        # STEP B: Merge Datasets
        print("🔗 Merging Metadata and Reviews...")
        df = pd.merge(reviews, meta, left_on='Restaurant', right_on='Name', how='left')

        # STEP C: Data Cleaning
        print("✨ Cleaning text and handling missing values...")
        # Drop rows where rating or review is missing
        df = df.dropna(subset=['Rating', 'Review'])
        
        # Clean the reviews
        df['cleaned_review'] = df['Review'].apply(clean_text)
        
        # Convert Rating to numeric
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        df = df.dropna(subset=['Rating'])

        # STEP D: Save Final Output
        output_file = f'{PROCESSED_PATH}/final_clean_data.csv'
        df.to_csv(output_file, index=False)
        print(f"🚀 Success! Final cleaned data saved at: {output_file}")

    except FileNotFoundError as e:
        print(f"❌ Error: Could not find the raw files. Check if they are in {RAW_PATH}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_pipeline()