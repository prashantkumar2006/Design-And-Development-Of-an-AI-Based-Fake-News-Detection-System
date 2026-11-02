"""Build word and character TF-IDF features for article and short-title models."""

import re
import joblib
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import StratifiedGroupKFold


def clean(text, stops):
    text = str(text).lower()
    text = re.sub(r"<.*?>|http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(word for word in text.split() if word not in stops)


df = pd.read_csv("data/combined_dataset.csv").fillna("")
stops = set(stopwords.words("english"))
titles = df["title"].map(lambda value: clean(value, stops))
articles = (df["title"] + " " + df["text"]).map(lambda value: clean(value, stops))
labels = df["label"].astype(int)

groups = titles.str.replace(r"\s+", " ", regex=True).str.strip()
train_idx, test_idx = next(StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
).split(articles, labels, groups))


def make_vectorizer(word_features, char_features):
    return FeatureUnion([
        ("word", TfidfVectorizer(max_features=word_features, ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(max_features=char_features, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])


article_vectorizer = make_vectorizer(50000, 50000)
X_train = article_vectorizer.fit_transform(articles.iloc[train_idx])
X_test = article_vectorizer.transform(articles.iloc[test_idx])

short_vectorizer = make_vectorizer(30000, 40000)
X_short_train = short_vectorizer.fit_transform(titles.iloc[train_idx])
X_short_test = short_vectorizer.transform(titles.iloc[test_idx])

joblib.dump(article_vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(short_vectorizer, "models/short_vectorizer.pkl")
joblib.dump(X_train, "models/X_train_tfidf.pkl")
joblib.dump(X_test, "models/X_test_tfidf.pkl")
joblib.dump(X_short_train, "models/X_short_train.pkl")
joblib.dump(X_short_test, "models/X_short_test.pkl")
joblib.dump(labels.iloc[train_idx], "models/y_train.pkl")
joblib.dump(labels.iloc[test_idx], "models/y_test.pkl")
joblib.dump(titles.iloc[train_idx], "models/titles_train.pkl")
joblib.dump(titles.iloc[test_idx], "models/titles_test.pkl")
print(f"Article features: {X_train.shape[1]:,}; short-text features: {X_short_train.shape[1]:,}")
print(f"Training rows: {len(train_idx):,}; test rows: {len(test_idx):,}")
