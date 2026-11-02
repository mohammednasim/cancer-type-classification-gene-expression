
import json

from data_utils import load_dataset, BASE_DIR


def main():
    X, y, info = load_dataset()

    report = {
        **info,
        "missing_feature_values": int(X.isna().sum().sum()),
        "duplicate_samples": int(X.duplicated().sum()),
        "constant_features": int((X.nunique(dropna=False) <= 1).sum()),
        "non_numeric_features": [],
    }

    report_dir = BASE_DIR / "results"
    report_dir.mkdir(parents=True, exist_ok=True)

    output = report_dir / "dataset_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n===== DATASET AUDIT =====")
    print("Samples:", info["samples"])
    print("Gene features:", info["features"])
    print("Cancer classes:", info["classes"])
    print("\nClass distribution:")
    print(y.value_counts())
    print("\nMissing values:", report["missing_feature_values"])
    print("Duplicate feature rows:", report["duplicate_samples"])
    print("Constant features:", report["constant_features"])
    print("\nAlignment:", info["alignment"])
    print("\nSaved audit:", output)


if __name__ == "__main__":
    main()

