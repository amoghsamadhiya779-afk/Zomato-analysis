import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "final_clean_data.csv"
MODEL_DIR = BASE_DIR / "models"

def train():
    logger.info("🚀 Starting Multi-Model Training...")
    
    # 1. Load Data
    if not DATA_PATH.exists():
        logger.error(f"Data file missing at {DATA_PATH}. Run preprocess.py first.")
        return
    
    df = pd.read_csv(DATA_PATH)
    # Ensure no missing values in critical columns
    df = df.dropna(subset=['cleaned_review', 'Rating'])
    
    # Labels: 1 = Positive (>3.0), 0 = Negative
    y = df['Rating'].apply(lambda x: 1 if x > 3.0 else 0)
    X = df['cleaned_review'].astype(str)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Define Pipelines (The "Brains")
    pipelines = {
        "RandomForest": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        "LogisticRegression": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('classifier', LogisticRegression(random_state=42))
        ])
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {}

    # 3. Train & Save All Models
    for name, pipeline in pipelines.items():
        logger.info(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        metrics[name] = acc
        logger.info(f"✅ {name} Accuracy: {acc:.2%}")
        
        # Save
        joblib.dump(pipeline, MODEL_DIR / f"{name}.pkl")

    # Save Metrics for the UI to display
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info("🎉 All models trained and saved!")

if __name__ == "__main__":
    train()