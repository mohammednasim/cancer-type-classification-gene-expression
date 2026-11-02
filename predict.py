
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_utils import BASE_DIR


def load_new_samples(file_path, expected_features):
    X = pd.read_csv(file_path, low_memory=False)

    # Remove a clearly exported sequential Pandas index.
    for col in list(X.columns):
        if str(col).strip().lower().startswith("unnamed"):
            values = pd.to_numeric(X[col], errors="coerce")

            if values.notna().all() and (
                np.array_equal(values.to_numpy(), np.arange(len(X)))
                or np.array_equal(values.to_numpy(), np.arange(1, len(X) + 1))
            ):
                X = X.drop(columns=col)

    missing = [c for c in expected_features if c not in X.columns]
    extra = [c for c in X.columns if c not in expected_features]

    if missing or extra:
        raise ValueError(
            "Input columns do not match the training data.\n"
            f"Missing columns: {missing[:10]}\n"
            f"Unexpected columns: {extra[:10]}\n"
            "Use a CSV with the same gene feature names as the training data."
        )

    # Preserve the exact training feature order.
    X = X.loc[:, expected_features]

    numeric = X.apply(pd.to_numeric, errors="coerce")
    invalid = X.notna() & numeric.isna()

    if invalid.any().any():
        raise ValueError("Non-numeric gene-expression values were found.")

    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Input contains infinite values.")

    return numeric


def main():
    parser = argparse.ArgumentParser(
        description="Predict cancer types from gene expression."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to new gene-expression CSV.",
    )
    parser.add_argument(
        "--output",
        default="results/predictions.csv",
        help="Path to save predictions.",
    )

    args = parser.parse_args()

    artifact = joblib.load(
        BASE_DIR / "models" / "best_model.joblib"
    )

    X_new = load_new_samples(
        args.input,
        artifact["feature_columns"],
    )

    model = artifact["pipeline"]
    predictions = model.predict(X_new)

    output = pd.DataFrame({
        "predicted_cancer_type": predictions,
    })

    # Decision scores or probabilities are model outputs, not necessarily
    # calibrated probabilities of a patient having a particular cancer.
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_new)

        for i, class_name in enumerate(model.classes_):
            output[f"score_{class_name}"] = scores[:, i]

    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X_new))

        if scores.ndim == 2:
            for i, class_name in enumerate(model.classes_):
                output[f"decision_score_{class_name}"] = scores[:, i]
        else:
            output["decision_score"] = scores

    output_path = Path(args.output)

    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(output.head())
    print("\nPredictions saved to:", output_path)


if __name__ == "__main__":
    main()
