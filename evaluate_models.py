"""Report held-out metrics for the article and short-text classifiers."""

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_test = joblib.load("models/y_test.pkl")
evaluations = [
    ("Full article", "models/svm.pkl", "models/X_test_tfidf.pkl"),
    ("Short headline", "models/short_text_model.pkl", "models/X_short_test.pkl"),
]
rows = []

for name, model_path, data_path in evaluations:
    model = joblib.load(model_path)
    features = joblib.load(data_path)
    predicted = model.predict(features)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predicted),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted, zero_division=0),
        "F1 Score": f1_score(y_test, predicted, zero_division=0),
    })
    print(f"\n{name} confusion matrix [Fake, Real]:")
    print(confusion_matrix(y_test, predicted))

summary = pd.DataFrame(rows).set_index("Model").round(4)
print("\nGrouped ISOT holdout metrics (not current-news accuracy):")
print(summary)
summary.to_csv("data/model_comparison.csv")
