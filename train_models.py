"""Train article and short-title classifiers from Phase 5 TF-IDF features."""

import joblib
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")
X_train = joblib.load("models/X_train_tfidf.pkl")
X_test = joblib.load("models/X_test_tfidf.pkl")
X_short_train = joblib.load("models/X_short_train.pkl")
X_short_test = joblib.load("models/X_short_test.pkl")


def fit_report(name, train_x, test_x, output_path):
    model = LinearSVC(C=0.5, class_weight="balanced", max_iter=5000)
    model.fit(train_x, y_train)
    predicted = model.predict(test_x)
    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(y_test, predicted):.4f}")
    print(classification_report(y_test, predicted, target_names=["Fake", "Real"]))
    print("Confusion matrix [Fake, Real]:\n", confusion_matrix(y_test, predicted))
    joblib.dump(model, output_path)


fit_report("Full article model", X_train, X_test, "models/svm.pkl")
fit_report("Short headline model", X_short_train, X_short_test, "models/short_text_model.pkl")
print("\nSaved full-article and short-text models.")
