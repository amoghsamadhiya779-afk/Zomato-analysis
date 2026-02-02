import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# Deep Learning Imports - CORRECTED
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "final_clean_data.csv"
MODEL_DIR = BASE_DIR / "models"
VOCAB_SIZE = 10000  # We will look at top 10k words
MAX_LENGTH = 100    # Max words in a review to look at
EMBEDDING_DIM = 16  # Size of the vector for each word

def train_neural_network():
    print("🧠 Starting Deep Learning Training...")

    # 1. Load Data
    if not DATA_PATH.exists():
        print(f"❌ Data not found at {DATA_PATH}!")
        return
    
    df = pd.read_csv(DATA_PATH)
    # Ensure critical columns exist
    if 'cleaned_review' not in df.columns or 'Rating' not in df.columns:
        print("❌ Data missing required columns.")
        return

    df = df.dropna(subset=['cleaned_review', 'Rating'])
    
    # Labels: 1 = Positive, 0 = Negative
    y = df['Rating'].apply(lambda x: 1 if x > 3.0 else 0).values
    X_text = df['cleaned_review'].astype(str).values

    # 2. Tokenization (Text -> Numbers)
    print("🔢 Tokenizing text...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_text)
    
    # Convert text to sequences of numbers
    sequences = tokenizer.texts_to_sequences(X_text)
    
    # Padding (Ensure all inputs are the same length)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)

    # 4. Build the Neural Network Architecture
    print("🏗️ Building Model Architecture...")
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # 5. Train (Backpropagation)
    print("🔥 Training Model (Epochs)...")
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)

    # 6. Save Model & Tokenizer
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save Keras Model
    model.save(MODEL_DIR / "zomato_nn.h5")
    
    # Save Tokenizer
    with open(MODEL_DIR / "tokenizer.pickle", 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"✅ Neural Network saved to {MODEL_DIR}")
    
    # Final Evaluation
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"🏆 Final Test Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    train_neural_network()