import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from nlp_pipeline import build_nlp_features


def execute_model_suite():

    X_train, X_test, y_train, y_test, vectorizer = build_nlp_features()

    models_pool = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),

        "Linear SVM (Best NLP Model)": LinearSVC(),

        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0)
    }

    performance_records = {}

    chart_dir = os.path.join("outputs", "charts")
    os.makedirs(chart_dir, exist_ok=True)

    for name, model in models_pool.items():

        print(f"\n[*] Training: {name}")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average="macro")
        rec = recall_score(y_test, predictions, average="macro")
        f1 = f1_score(y_test, predictions, average="macro")

        performance_records[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "Model": model
        }

        print("\n", classification_report(
            y_test,
            predictions,
            target_names=["Deceptive", "Genuine"]
        ))

        cm = confusion_matrix(y_test, predictions)

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Deceptive", "Genuine"],
            yticklabels=["Deceptive", "Genuine"]
        )
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        save_path = os.path.join(
            chart_dir,
            f"cm_{name.lower().replace(' ', '_')}.png"
        )

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    # Summary table
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 60)

    summary = []

    for name, m in performance_records.items():
        summary.append({
            "Model": name,
            "Accuracy": round(m["Accuracy"], 4),
            "Precision": round(m["Precision"], 4),
            "Recall": round(m["Recall"], 4),
            "F1": round(m["F1_Score"], 4),
        })

    print(pd.DataFrame(summary).to_string(index=False))

    # Best model selection
    best_model_name = max(
        performance_records,
        key=lambda x: performance_records[x]["F1_Score"]
    )

    best_model = performance_records[best_model_name]["Model"]

    model_path = os.path.join("models", "best_model.pkl")
    joblib.dump(best_model, model_path)

    print("\n" + "#" * 60)
    print(f"BEST MODEL: {best_model_name}")
    print(f"SAVED TO: {model_path}")
    print("#" * 60)


if __name__ == "__main__":
    execute_model_suite()
