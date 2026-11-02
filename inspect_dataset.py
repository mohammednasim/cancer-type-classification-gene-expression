"""
Dataset inspection only — no modeling.
Reads data.csv and labels.csv, reports structure, alignment, and data quality.
Does NOT modify the original files.
"""
import os
import sys
import pandas as pd
import numpy as np

DATA_DIR = os.path.join("Dataset")
DATA_PATH = os.path.join(DATA_DIR, "data.csv")
LABELS_PATH = os.path.join(DATA_DIR, "labels.csv")

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def main():
    for p in (DATA_PATH, LABELS_PATH):
        if not os.path.isfile(p):
            print(f"ERROR: missing file {p}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # 1. RAW PEEK — determine structure before full load
    # ------------------------------------------------------------------
    section("1. RAW FILE PEEK (first 2 lines, truncated)")

    for path, name in ((LABELS_PATH, "labels.csv"), (DATA_PATH, "data.csv")):
        print(f"\n--- {name} ---")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 2:
                    break
                # show up to 300 chars
                preview = line.rstrip("\n")
                if len(preview) > 300:
                    preview = preview[:300] + f" ... [truncated, total {len(line)} chars]"
                print(f"line {i}: {preview}")

    # ------------------------------------------------------------------
    # 2. LOAD LABELS
    # ------------------------------------------------------------------
    section("2. LABELS.CSV INSPECTION")
    labels = pd.read_csv(LABELS_PATH)
    print(f"shape: {labels.shape}")
    print(f"columns: {list(labels.columns)}")
    print(f"dtypes:\n{labels.dtypes}")
    print(f"missing values:\n{labels.isna().sum()}")
    print(f"duplicate rows: {labels.duplicated().sum()}")
    print(f"first 5 rows:\n{labels.head()}")
    print(f"last 5 rows:\n{labels.tail()}")

    # Identify target column: non-ID, non-numeric column with few unique values
    target_col = None
    id_col = None
    for c in labels.columns:
        if labels[c].dtype == object:
            nunique = labels[c].nunique()
            if nunique <= 50 and nunique > 1:
                target_col = c
        else:
            # potential index column
            if labels[c].is_unique and labels[c].dtype in (np.int64, np.int32):
                pass

    if target_col is None:
        print("\nWARNING: could not auto-detect target column.")
        target_col = labels.columns[-1]
        print(f"Falling back to last column: {target_col}")

    print(f"\nDetected target column: '{target_col}'")
    print(f"unique classes ({labels[target_col].nunique()}):\n{labels[target_col].value_counts()}")
    print(f"samples per class:\n{labels[target_col].value_counts()}")

    # Check for an unnamed index column (pandas writes ',Class' style header)
    unnamed_cols = [c for c in labels.columns if str(c).startswith("Unnamed")]
    print(f"\nUnnamed (index-like) columns: {unnamed_cols}")

    # If first column looks like sample IDs
    first_col = labels.columns[0]
    print(f"first column: '{first_col}' — sample values: {labels[first_col].head(3).tolist()}")

    # ------------------------------------------------------------------
    # 3. LOAD DATA.CSV — full load (206 MB is manageable)
    # ------------------------------------------------------------------
    section("3. DATA.CSV INSPECTION (full load)")
    print("Loading data.csv ... this may take a moment.")
    data = pd.read_csv(DATA_PATH)
    print(f"shape: {data.shape}")
    print(f"number of columns: {data.shape[1]}")
    print(f"first 5 column names: {list(data.columns[:5])}")
    print(f"last 5 column names: {list(data.columns[-5:])}")
    print(f"first 5 rows (first 5 columns only):\n{data.iloc[:5, :5]}")
    print(f"dtypes summary:\n{data.dtypes.value_counts()}")
    print(f"total missing values: {data.isna().sum().sum()}")
    print(f"total duplicate rows: {data.duplicated().sum()}")

    # Non-numeric columns in data (excluding possible ID column)
    non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"\nnon-numeric columns in data.csv: {len(non_numeric)}")
    if non_numeric:
        print(f"  examples: {non_numeric[:10]}")

    # Check if first column is an identifier (non-numeric or all unique strings)
    first_data_col = data.columns[0]
    print(f"\nfirst column of data.csv: '{first_data_col}'")
    print(f"  dtype: {data[first_data_col].dtype}")
    print(f"  sample values: {data[first_data_col].head(3).tolist()}")

    # ------------------------------------------------------------------
    # 4. ORIENTATION & ALIGNMENT CHECK
    # ------------------------------------------------------------------
    section("4. ORIENTATION & SAMPLE ALIGNMENT CHECK")

    n_data_rows = data.shape[0]
    n_label_rows = labels.shape[0]
    print(f"data.csv rows: {n_data_rows}")
    print(f"labels.csv rows: {n_label_rows}")
    print(f"row counts match: {n_data_rows == n_label_rows}")
# Attempt sample-ID based alignment
    # A string-like column: dtype object OR pandas string dtype (str)
    def is_stringlike(series):
        return series.dtype == object or str(series.dtype) in ("str", "string")

data_id_col = None
    label_id_col = None

# Candidate: first column of data is string-like
    if is_stringlike(data[first_data_col]):
        data_id_col = first_data_col
    # Candidate: labels has an ID column separate from target
    for c in labels.columns:
        if c != target_col and is_stringlike(labels[c]):
            label_id_col = c
            break

print(f"\ndata ID column candidate: {data_id_col}")
    print(f"label ID column candidate: {label_id_col}")


    print(f"\ndata ID column candidate: {data_id_col}")
    print(f"label ID column candidate: {label_id_col}")

    alignment_verified = False
    if data_id_col is not None and label_id_col is not None:
        data_ids = data[data_id_col].astype(str).tolist()
        label_ids = labels[label_id_col].astype(str).tolist()
        if len(data_ids) == len(label_ids):
            same_order = data_ids == label_ids
            same_set = set(data_ids) == set(label_ids)
            print(f"ID lists identical in order: {same_order}")
            print(f"ID sets identical (regardless of order): {same_set}")if same_order:
                alignment_verified = True
                print("[OK] Alignment VERIFIED by matching sample IDs in the same order.")
            elif same_set:
                print("[WARN] Same sample IDs but DIFFERENT order - reindexing required.")
                print("  Cannot assume row i of data corresponds to row i of labels.")
            else:
                print("[FAIL] Sample ID sets differ - alignment CANNOT be verified.")
        else:
            print("[FAIL] ID lists have different lengths - alignment CANNOT be verified.")
    else:
        print("[WARN] No sample-ID columns found in one or both files.")
        print("  Alignment cannot be verified by ID matching; row order is assumed only.")
        print("  This is a LIMITATION - do not proceed to modeling without confirmation.")


    # ------------------------------------------------------------------
    # 5. LEAKAGE / STRUCTURE SANITY
    # ------------------------------------------------------------------
    section("5. LEAKAGE & STRUCTURE SANITY CHECKS")

    print(f"data.csv numeric-only (excluding ID col): "
          f"{data.drop(columns=[data_id_col]).select_dtypes(include=[np.number]).shape[1] if data_id_col else data.select_dtypes(include=[np.number]).shape[1]} numeric feature columns")

    # Check constant columns (zero variance)
    num_data = data.select_dtypes(include=[np.number])
    constant_cols = [c for c in num_data.columns if num_data[c].nunique(dropna=False) <= 1]
    print(f"constant / zero-variance columns: {len(constant_cols)}")
    if constant_cols:
        print(f"  examples: {constant_cols[:10]}")

    # Check for obvious target leakage: column names matching class names
    classes = labels[target_col].dropna().unique().tolist()
    suspicious = [c for c in data.columns if any(str(cl) in str(c) for cl in classes)]
    print(f"columns whose name contains a class label (potential leakage): {suspicious[:10]}")

    # Overall numeric stats on a small slice (full describe is huge)
    print(f"\nSummary stats (first 5 numeric columns):\n{num_data.iloc[:, :5].describe()}")

    # ------------------------------------------------------------------
    # 6. FINAL SUMMARY
    # ------------------------------------------------------------------
    section("6. FINAL SUMMARY")
    n_samples = n_data_rows
    n_genes = data.shape[1] - (1 if data_id_col else 0)
    n_classes = labels[target_col].nunique()

    print(f"Samples (rows in data.csv): {n_samples}")
    print(f"Genes / features (columns in data.csv, excluding ID): {n_genes}")
    print(f"Cancer classes: {n_classes}")
    print(f"Classes: {sorted(classes)}")
    print(f"Samples per class:\n{labels[target_col].value_counts()}")
    print(f"\nOrientation: samples as ROWS, genes as COLUMNS" if n_samples < n_genes else "\nOrientation: genes as ROWS, samples as COLUMNS")
    print(f"Alignment verified: {alignment_verified}")
    print(f"\nOriginal files untouched: {os.path.getsize(DATA_PATH)} bytes (data.csv), "
          f"{os.path.getsize(LABELS_PATH)} bytes (labels.csv)")

if __name__ == "__main__":
    main()
