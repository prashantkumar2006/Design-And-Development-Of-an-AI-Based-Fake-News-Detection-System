"""
Phase 5: Feature Engineering (TF-IDF)
Reads data/cleaned_dataset.csv (from Phase 3) and converts the clean_text
column into numeric TF-IDF vectors, saving the vectorizer and split data
for Phase 6 (model training).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs("models", exist_ok=True)

# 1. Load cleaned dataset
df = pd.read_csv("data/cleaned_dataset.csv")
df["clean_text"] = df["clean_text"].fillna("")

X = df["clean_text"]
y = df["label"]

# 2. Split into train and test sets BEFORE vectorizing
# Why before? To avoid "data leakage" — the vectorizer should only learn
# vocabulary/statistics from the training set, never peek at test data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# 3. Create the TF-IDF vectorizer
# max_features limits vocabulary size to the top N most informative words
# (keeps things fast and avoids overfitting to rare words)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# 4. Fit on training data ONLY, then transform both train and test
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF matrix shape (train):", X_train_tfidf.shape)
print("TF-IDF matrix shape (test):", X_test_tfidf.shape)

# 5. Save everything needed for Phase 6
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(X_train_tfidf, "models/X_train_tfidf.pkl")
joblib.dump(X_test_tfidf, "models/X_test_tfidf.pkl")
joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(y_test, "models/y_test.pkl")

print("\nSaved vectorizer and train/test splits to models/ folder.")
print("Ready for Phase 6: Model Training.")