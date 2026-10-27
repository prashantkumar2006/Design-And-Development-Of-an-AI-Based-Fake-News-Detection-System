"""
Phase 6: Train Machine Learning Models
Loads the TF-IDF features from Phase 5, trains three models
(Logistic Regression, Naive Bayes, SVM), and compares their accuracy.
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 1. Load the saved TF-IDF data from Phase 5
X_train = joblib.load("models/X_train_tfidf.pkl")
X_test = joblib.load("models/X_test_tfidf.pkl")
y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")

results = {}

# 2. Logistic Regression
print("Training Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
results["Logistic Regression"] = accuracy_score(y_test, log_reg.predict(X_test))
joblib.dump(log_reg, "models/logistic_regression.pkl")

# 3. Naive Bayes
print("Training Naive Bayes...")
nb = MultinomialNB()
nb.fit(X_train, y_train)
results["Naive Bayes"] = accuracy_score(y_test, nb.predict(X_test))
joblib.dump(nb, "models/naive_bayes.pkl")

# 4. SVM (linear kernel, well suited to high-dimensional text data)
print("Training SVM...")
svm = LinearSVC()
svm.fit(X_train, y_train)
results["SVM"] = accuracy_score(y_test, svm.predict(X_test))
joblib.dump(svm, "models/svm.pkl")

# 5. Compare results
print("\n" + "=" * 40)
print("MODEL COMPARISON")
print("=" * 40)
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {acc*100:.2f}%")

best = max(results, key=results.get)
print(f"\nBest model: {best} ({results[best]*100:.2f}% accuracy)")
print("\nAll models saved to models/ folder.")