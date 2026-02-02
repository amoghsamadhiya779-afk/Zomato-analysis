import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Define paths
PROCESSED_DATA_PATH = 'data/processed/final_clean_data.csv'
MODELS_PATH = 'models'

def train_model():
    # 1. Load Data
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"❌ Error: Data file not found at {PROCESSED_DATA_PATH}. Run preprocess.py first.")
        return

    print("📊 Loading data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Handle missing values just in case
    df = df.dropna(subset=['cleaned_review', 'Rating'])

    # 2. Create Labels (Sentiment)
    # Rule: Rating > 3.0 is Positive (1), else Negative (0)
    print("🏷️ Creating sentiment labels...")
    df['label'] = df['Rating'].apply(lambda x: 1 if x > 3.0 else 0)

    # 3. Vectorization (Text -> Numbers)
    print("🔠 Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000) # Keep top 5000 words
    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df['label']

    # 4. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train Model
    print("🤖 Training Random Forest Model (this may take a moment)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. Evaluate
    print("📈 Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save Model & Vectorizer (The "MLOps" part)
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    joblib.dump(model, f'{MODELS_PATH}/sentiment_model.pkl')
    joblib.dump(vectorizer, f'{MODELS_PATH}/tfidf_vectorizer.pkl')
    print(f"💾 Model saved to {MODELS_PATH}/sentiment_model.pkl")

if __name__ == "__main__":
    train_model()