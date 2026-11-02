
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Dataset"

DATA_FILE = DATA_DIR / "data.csv"
LABEL_FILE = DATA_DIR / "labels.csv"


def read_csv_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    return pd.read_csv(path, low_memory=False)


def is_sequential_index(series):
    """Identify a typical exported Pandas row index."""
    values = pd.to_numeric(series, errors="coerce")

    if values.isna().any():
        return False

    arr = values.to_numpy()

    return (
        np.array_equal(arr, np.arange(len(arr)))
        or np.array_equal(arr, np.arange(1, len(arr) + 1))
    )


def remove_exported_index(df):
    """Remove only clearly recognizable exported row-index columns."""
    df = df.copy()

    for col in list(df.columns):
        name = str(col).strip().lower()

        if name.startswith("unnamed") and is_sequential_index(df[col]):
            df = df.drop(columns=col)

    return df


def load_dataset():
    """
    Load gene expression features and cancer labels.

    Returns
    -------
    X : pandas DataFrame
        Numeric gene-expression features.
    y : pandas Series
        Cancer-type labels.
    metadata : dict
        Basic information about the dataset.
    """
    data = read_csv_file(DATA_FILE)
    labels = read_csv_file(LABEL_FILE)

    if data.empty or labels.empty:
        raise ValueError("The dataset or label file is empty.")

    # Find the target column using common label names.
    target_candidates = [
        col for col in labels.columns
        if str(col).strip().lower()
        in {"class", "label", "target", "cancer_type", "cancer_type", "diagnosis"}
    ]

    if target_candidates:
        target_col = target_candidates[0]
    else:
        # Standard UCI labels.csv normally has the class in its last column.
        target_col = labels.columns[-1]

    y = labels[target_col].astype("string").str.strip()

    if y.isna().any() or y.eq("").any():
        raise ValueError("Missing or blank cancer labels were found.")

    # Attempt to align samples using shared, unique columns.
    # This is especially useful if both CSVs contain sample IDs.
    common_columns = [
        col for col in data.columns
        if col in labels.columns and col != target_col
    ]

    matched_key = None

    for col in common_columns:
        data_values = data[col].astype(str).str.strip()
        label_values = labels[col].astype(str).str.strip()

        if (
            data_values.is_unique
            and label_values.is_unique
            and set(data_values) == set(label_values)
        ):
            matched_key = col
            break

    if matched_key is not None:
        if data[matched_key].astype(str).duplicated().any():
            raise ValueError("Duplicate sample identifiers found.")

        feature_data = data.copy()
        label_map = pd.DataFrame({
            matched_key: labels[matched_key].astype(str).str.strip(),
            "_target": y.astype(str),
        })

        merged = feature_data.merge(
            label_map,
            on=matched_key,
            how="left",
            validate="one_to_one",
            sort=False,
        )

        if merged["_target"].isna().any():
            raise ValueError("Some samples have no matching cancer label.")

        y = merged["_target"].astype(str).reset_index(drop=True)
        X = merged.drop(columns=[matched_key, "_target"])
        alignment = f"Matched using shared identifier: {matched_key}"

    else:
        # No shared sample ID found. Row order cannot be proven from shape alone.
        if len(data) != len(labels):
            raise ValueError(
                f"Sample count mismatch: data.csv has {len(data)} rows "
                f"and labels.csv has {len(labels)} rows. "
                "Inspect the CSV files and identify the sample IDs."
            )

        data = remove_exported_index(data)
        labels_clean = remove_exported_index(labels)

        if target_col not in labels_clean.columns:
            target_col = labels_clean.columns[-1]

        y = labels_clean[target_col].astype("string").str.strip()

        if y.isna().any() or y.eq("").any():
            raise ValueError("Missing labels were found.")

        X = data
        alignment = (
            "Row order assumed. Verify from the original dataset documentation "
            "that data.csv and labels.csv have the same sample order."
        )

        print("WARNING:", alignment)

    # All model features must be numeric.
    original_columns = list(X.columns)
    X = X.apply(pd.to_numeric, errors="coerce")

    for col in original_columns:
        invalid = pd.to_numeric(
            data[col] if col in data.columns else X[col],
            errors="coerce"
        ).isna()

        # Missing numeric values can be imputed later; strings cannot be silently
        # treated as missing values.
        source = data[col] if col in data.columns else X[col]
        bad = source.notna() & pd.to_numeric(source, errors="coerce").isna()

        if bad.any():
            raise ValueError(
                f"Non-numeric values detected in feature column '{col}'. "
                "Inspect the CSV headers and remove only confirmed identifiers."
            )

    X.columns = X.columns.astype(str)
    y = pd.Series(y.astype(str).to_numpy(), name="cancer_type")

    if len(X) != len(y):
        raise ValueError("Feature and label counts do not match.")

    if X.shape[1] == 0:
        raise ValueError("No gene-expression features were found.")

    if y.nunique() < 2:
        raise ValueError("At least two cancer classes are required.")

    if not np.isfinite(
        X.to_numpy(dtype=float, na_value=np.nan)
    )[~np.isnan(X.to_numpy(dtype=float, na_value=np.nan))].all():
        raise ValueError("Infinite values were found in gene-expression data.")

    metadata = {
        "samples": int(len(X)),
        "features": int(X.shape[1]),
        "classes": int(y.nunique()),
        "class_counts": y.value_counts().to_dict(),
        "alignment": alignment,
        "feature_names": list(X.columns),
    }

    return X.reset_index(drop=True), y.reset_index(drop=True), metadata


if __name__ == "__main__":
    X, y, info = load_dataset()

    print("\nDataset loaded successfully")
    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)
    print("Cancer classes:", sorted(y.unique()))
    print("Class distribution:\n", y.value_counts())
    print("Alignment:", info["alignment"])
