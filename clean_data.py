"""
Phase 3: Clean the text data.
Reads data/combined_dataset.csv (created in Phase 2) and produces
data/cleaned_dataset.csv with a new 'clean_text' column ready for modeling.
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords list (only needs to run once, but safe to leave in)
nltk.download("stopwords")

# 1. Load the combined dataset from Phase 2
df = pd.read_csv("data/combined_dataset.csv")

# 2. Combine title + text into one field (title often carries strong signal too)
df["content"] = df["title"] + " " + df["text"] 

# 3. Define the cleaning function
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()                          # lowercase everything
    text = re.sub(r"<.*?>", " ", text)                 # remove HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)         # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)               # remove numbers, punctuation, special chars
    text = re.sub(r"\s+", " ", text).strip()            # collapse extra whitespace
    words = text.split()
    words = [w for w in words if w not in stop_words]  # remove stopwords
    return " ".join(words)

# 4. Apply cleaning to every article (this may take a minute or two on 44,898 rows)
print("Cleaning text... this can take a minute.")
df["clean_text"] = df["content"].apply(clean_text)

# 5. Quick before/after check
print("\nBEFORE:\n", df["content"].iloc[0][:300])
print("\nAFTER:\n", df["clean_text"].iloc[0][:300])

# 6. Drop rows that became empty after cleaning (rare, but possible)
before_count = len(df)
df = df[df["clean_text"].str.strip() != ""]
after_count = len(df)
print(f"\nDropped {before_count - after_count} empty rows after cleaning.")

# 7. Save the cleaned dataset for Phase 4 (EDA) and Phase 5 (feature engineering)
df.to_csv("data/cleaned_dataset.csv", index=False)
print("\nSaved cleaned_dataset.csv to data/ folder.")
print("Final shape:", df.shape) 

