"""
Phase 9: Detailed Model Evaluation
Computes accuracy, precision, recall, F1, confusion matrices and ROC curves
for all three trained models. Charts are saved in the charts/ folder.
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# 1. Load test data and trained models
X_test = joblib.load("models/X_test_tfidf.pkl")
y_test = joblib.load("models/y_test.pkl")

models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
    "SVM": joblib.load("models/svm.pkl"),
}

rows = []
roc_data = {}

for name, model in models.items():
    preds = model.predict(X_test)

    # 2. Core metrics
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1 Score": f1_score(y_test, preds),
    })

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print(classification_report(y_test, preds, target_names=["Fake", "Real"]))

    # 3. Confusion matrix (saved as an image)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"charts/confusion_matrix_{name.replace(' ', '_').lower()}.png")
    plt.close()

    # 4. Scores for the ROC curve
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_data[name] = (fpr, tpr, auc(fpr, tpr))

# 5. ROC curves for all models on one chart
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("charts/roc_curves.png")
plt.close()

# 6. Summary table
summary = pd.DataFrame(rows).set_index("Model").round(4)
print("\n" + "=" * 50)
print("SUMMARY TABLE")
print("=" * 50)
print(summary)
summary.to_csv("data/model_comparison.csv")

print("\nSaved confusion matrices and roc_curves.png to charts/ folder.")
print("Saved model_comparison.csv to data/ folder.")