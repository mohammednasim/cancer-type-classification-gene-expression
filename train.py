
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning

from data_utils import load_dataset, BASE_DIR


RANDOM_STATE = 42
TEST_SIZE = 0.20
PRIMARY_METRIC = "f1_macro"


def create_pipeline(model, scale=False, select_features=True):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
    ]

    if select_features:
        steps.append(("select", SelectKBest(score_func=f_classif, k=1000)))

    if scale:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model))
    return Pipeline(steps)


def main():
    X, y, info = load_dataset()

    if X.isna().all(axis=0).any():
        print("Warning: all-missing features will be removed by preprocessing.")

    min_count = int(y.value_counts().min())
    if min_count < 2:
        raise ValueError(
            "The smallest class has fewer than two samples. "
            "A stratified train/test split is not reliable."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    folds = min(5, int(y_train.value_counts().min()))
    if folds < 2:
        raise ValueError("Insufficient samples per class for cross-validation.")

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    n_features = X_train.shape[1]
    k = min(1000, n_features)

    models = {
        "Dummy Baseline": create_pipeline(
            DummyClassifier(strategy="most_frequent"),
            select_features=False,
        ),
        "Logistic Regression": create_pipeline(
            LogisticRegression(
                C=1.0,
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            scale=True,
        ),
        "Linear SVM": create_pipeline(
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=10000,
                random_state=RANDOM_STATE,
            ),
            scale=True,
        ),
        "Random Forest": create_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
        "Extra Trees": create_pipeline(
            ExtraTreesClassifier(
                n_estimators=200,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
    }

    # Set k explicitly in each feature-selection pipeline.
    for pipeline in models.values():
        if "select" in pipeline.named_steps:
            pipeline.set_params(select__k=k)

    scoring = {
        "macro_f1": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
    }

    rows = []

    for name, pipeline in models.items():
        print(f"\nTraining and cross-validating: {name}")
        start = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)

            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                error_score="raise",
                return_train_score=False,
            )

        elapsed = time.time() - start

        rows.append({
            "model": name,
            "cv_macro_f1_mean": float(np.mean(scores["test_macro_f1"])),
            "cv_macro_f1_std": float(np.std(scores["test_macro_f1"], ddof=1)),
            "cv_balanced_accuracy_mean": float(
                np.mean(scores["test_balanced_accuracy"])
            ),
            "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "elapsed_seconds": round(elapsed, 2),
        })

        print(
            f"Macro F1: {rows[-1]['cv_macro_f1_mean']:.4f} "
            f"+/- {rows[-1]['cv_macro_f1_std']:.4f}"
        )

    comparison = pd.DataFrame(rows).sort_values(
        ["cv_macro_f1_mean", "cv_balanced_accuracy_mean"],
        ascending=False,
    )

    results_dir = BASE_DIR / "results"
    models_dir = BASE_DIR / "models"
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    comparison.to_csv(results_dir / "model_comparison.csv", index=False)

    # Select the winner using training cross-validation only.
    best_name = comparison.iloc[0]["model"]
    best_pipeline = models[best_name]

    # Fit the chosen pipeline on the full training partition.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        best_pipeline.fit(X_train, y_train)

    artifact = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "feature_columns": list(X.columns),
        "classes": list(best_pipeline.classes_),
        "random_state": RANDOM_STATE,
        "dataset_metadata": info,
    }

    joblib.dump(artifact, models_dir / "best_model.joblib")

    # Preserve the held-out test set for the evaluation step.
    X_test.to_csv(results_dir / "X_test.csv", index=False)
    y_test.to_csv(results_dir / "y_test.csv", index=False, header=True)

    summary = {
        "selected_model": best_name,
        "selection_metric": PRIMARY_METRIC,
        "cv_folds": folds,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": X.shape[1],
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "note": "Selected using training CV; test data was not used for selection.",
    }

    (results_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n===== MODEL COMPARISON =====")
    print(comparison.to_string(index=False))
    print("\nSelected model:", best_name)
    print("Saved model:", models_dir / "best_model.joblib")


if __name__ == "__main__":
    main()

