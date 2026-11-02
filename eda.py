
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data_utils import load_dataset, BASE_DIR


def main():
    X, y, info = load_dataset()

    results = BASE_DIR / "results"
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # 1. Cancer class distribution
    plt.figure(figsize=(9, 5))
    y.value_counts().plot(kind="bar")
    plt.title("Cancer Type Distribution")
    plt.xlabel("Cancer type")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "class_distribution.png", dpi=180)
    plt.close()

    # 2. Distribution of gene-expression variances
    variances = X.var(axis=0).replace([np.inf, -np.inf], np.nan).dropna()
    positive = variances[variances > 0]

    plt.figure(figsize=(9, 5))
    if len(positive):
        sns.histplot(np.log10(positive), bins=40)
    plt.title("Distribution of Gene-Expression Variance")
    plt.xlabel("Log10 variance")
    plt.ylabel("Number of genes")
    plt.tight_layout()
    plt.savefig(figures / "gene_variance.png", dpi=180)
    plt.close()

    # 3. PCA visualization
    # Exploratory only: this plot uses the available dataset and is not an
    # estimate of predictive performance.
    values = X.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    values = SimpleImputer(strategy="median").fit_transform(values)
    values = StandardScaler().fit_transform(values)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(values)

    pca_df = pd.DataFrame({
        "PC1": components[:, 0],
        "PC2": components[:, 1],
        "Cancer type": y.to_numpy()
    })

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Cancer type",
        s=55
    )
    plt.title("PCA Visualization of Gene-Expression Samples")
    plt.tight_layout()
    plt.savefig(figures / "pca_visualization.png", dpi=180)
    plt.close()

    report = f"""# Exploratory Data Analysis

- Samples: {info['samples']}
- Gene features: {info['features']}
- Cancer classes: {info['classes']}
- Missing values: {int(X.isna().sum().sum())}
- Duplicate rows: {int(X.duplicated().sum())}

## Class counts

{y.value_counts().to_string()}

## Generated figures

- `results/figures/class_distribution.png`
- `results/figures/gene_variance.png`
- `results/figures/pca_visualization.png`

PCA is an exploratory visualization, not a measure of model accuracy.
"""

    (results / "eda_report.md").write_text(report, encoding="utf-8")
    print(report)
    print("EDA figures saved to:", figures)


if __name__ == "__main__":
    main()

