
import json

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data_utils import BASE_DIR


def main():
    results = BASE_DIR / "results"
    models_dir = BASE_DIR / "models"
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "best_model.joblib"
    x_path = results / "X_test.csv"
    y_path = results / "y_test.csv"

    for path in [model_path, x_path, y_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run python train.py first."
            )

    artifact = joblib.load(model_path)
    model = artifact["pipeline"]

    X_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path).iloc[:, 0].astype(str)

    expected_columns = artifact["feature_columns"]

    if list(X_test.columns) != expected_columns:
        raise ValueError("Test features do not match the saved model.")

    predictions = model.predict(X_test)
    classes = list(artifact["classes"])

    metrics = {
        "selected_model": artifact["model_name"],
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "macro_precision": float(
            precision_score(
                y_test, predictions, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_test, predictions, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_test, predictions, average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_test, predictions, average="weighted", zero_division=0
            )
        ),
        "note": "Educational research evaluation, not clinical validation.",
    }

    (results / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    report = classification_report(
        y_test,
        predictions,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        results / "classification_report.csv"
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=classes)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(f"Confusion Matrix - {artifact['model_name']}")
    plt.xlabel("Predicted cancer type")
    plt.ylabel("Actual cancer type")
    plt.tight_layout()
    plt.savefig(figures / "confusion_matrix.png", dpi=180)
    plt.close()

    # Save test predictions for error analysis.
    pd.DataFrame({
        "actual_class": y_test.to_numpy(),
        "predicted_class": predictions,
        "correct": y_test.to_numpy() == predictions,
    }).to_csv(results / "test_predictions.csv", index=False)

    # Comparison plot uses cross-validation scores, not test scores.
    comparison_path = results / "model_comparison.csv"

    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path).sort_values(
            "cv_macro_f1_mean", ascending=False
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=comparison,
            x="cv_macro_f1_mean",
            y="model",
        )
        plt.xlabel("Cross-validation macro F1-score")
        plt.ylabel("Model")
        plt.title("ML Model Comparison")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(figures / "model_comparison.png", dpi=180)
        plt.close()

    print("\n===== FINAL TEST RESULTS =====")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=classes, zero_division=0
    ))
    print("\nSaved results to:", results)


if __name__ == "__main__":
    main()
