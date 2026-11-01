"""Builds notebooks/02_preprocessing.ipynb. Run: uv run python scripts/build_02.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbtools import save, step, text

cells: list = []


def maybe(name: str) -> None:
    """Insert a hand-written interpretation file (written AFTER reading the executed output), if it exists."""
    path = Path(__file__).with_name(f"{name}.py")
    if path.exists():
        exec(path.read_text(), globals())


text(cells, """
# 02 Preprocessing and dimensionality reduction

In notebook 01 we found that the money columns are heavily right-skewed, that some features repeat each other, and that a few values are missing. Here we fix those problems so KMeans can work well.

**Plan**
1. Keep the 11 features chosen in notebook 01 and compare several ways of transforming and scaling them (Step 5).
2. Build the chosen option as one scikit-learn Pipeline, so the exact same steps apply to any new customer.
3. Use PCA to look at the structure of the data (Step 6).

Every code cell has a short note above it with three lines. Imports sit inside the cells that first use them.

## Step 5: Preprocessing
""")

step(cells,
     "Loads the data, drops the ID and the six redundant features from notebook 01, and fills the few blanks with the median (inside a Pipeline step, so it is reusable).",
     "The shape (8,950 rows by 11 columns) and zero missing values after imputation.",
     "These 11 columns are the starting point for every option we compare. Filling blanks first means the transformers below never see a missing value.",
     '''
# Imported here so this step only loads the libraries when it runs
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def find_project_root(start: Path = Path.cwd()) -> Path:
    """Walk up from the current folder until we see pyproject.toml (nbconvert runs inside notebooks/)."""
    return next(folder for folder in [start, *start.parents] if (folder / "pyproject.toml").exists())


ROOT = find_project_root()
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Decided in notebook 01 (Step 3): each of these repeats information that another kept feature already holds
FEATURES_TO_DROP = [
    "PURCHASES", "PURCHASES_TRX", "CASH_ADVANCE_TRX",
    "CASH_ADVANCE_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
]

raw = pd.read_csv(ROOT / "data" / "raw" / "credit_card.csv").set_index("CUST_ID")  # keep the ID as the row label
all_features = raw.columns.tolist()  # the 17 original features
kept_features = [name for name in all_features if name not in FEATURES_TO_DROP]  # the 11 we cluster on
print(f"{raw.shape=}, {len(kept_features)=}")

# Median imputation (decision from notebook 01). We fit on all 17 columns so Step 8 can still describe dropped columns.
imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
imputed_all = imputer.fit_transform(raw)           # learn medians and fill blanks
imputed = imputed_all[kept_features]               # the 11 clustering features, still in original units
print(f"{imputed.isna().sum().sum()=}")
''')

step(cells,
     "Decides which features get a log transform and checks that log1p is safe for them.",
     "The two lists, and that the minimum of the log group is not negative.",
     "`log1p(x)` means log(1 + x). It pulls in long right tails, and unlike a plain log it works when x is 0 (many customers have 0). It breaks for negative numbers, so we confirm there are none. The four bounded columns (shares from 0 to 1, and TENURE) stay as they are, as decided in notebook 01.",
     '''
# Money and count columns: right-skewed, so log1p helps
log_features = [
    "BALANCE", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS",
]
# Bounded or near-constant columns: a log would be the wrong tool, we only scale them
plain_features = [name for name in kept_features if name not in log_features]
print(f"{log_features=}")
print(f"{plain_features=}")
print(f"{imputed[log_features].min().min()=}")  # must be >= 0 for log1p
assert imputed[log_features].min().min() >= 0
''')

text(cells, """
### Comparing four options

| Option | What it does |
|---|---|
| **A. StandardScaler alone** | Subtracts the mean and divides by the standard deviation, so each feature has mean 0 and spread 1 (a *z-score*). It does not change the shape, so skew stays. |
| **B. log1p, then StandardScaler** | Log-transforms the skewed columns first, then z-scores everything. |
| **C. QuantileTransformer** | Replaces each value with its rank, mapped onto a bell curve. Forces a normal shape no matter what. |
| **D. PowerTransformer (Yeo-Johnson)** | Finds the best power for each column to make it more symmetric. Works with zeros. |

We measure two things: **skewness** after the transform (closer to 0 is better) and **the largest absolute z-score** (how far the most extreme customer is from the average, in standard deviations). A KMeans distance is dominated by features with huge z-scores.
""")

step(cells,
     "Applies the four options to the 11 features and builds one table of skewness per feature, plus two summary rows.",
     "Column A should keep the original skew. Look at which option brings the most skewed features close to 0, and how large the biggest z-score stays.",
     "Skewness near 0 means a symmetric shape. A max |z| of 20 means one customer is 20 standard deviations from average, which would drag a KMeans centre. Smaller is calmer.",
     '''
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, QuantileTransformer, StandardScaler

# Option B by hand here, to compare; the real pipeline is built further down
logged = imputed.copy()
logged[log_features] = np.log1p(logged[log_features])  # log(1 + x) on the skewed columns only

options = {
    "A: Standard only": StandardScaler().fit_transform(imputed),
    "B: log1p + Standard": StandardScaler().fit_transform(logged),
    "C: Quantile (normal)": QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0).fit_transform(imputed),
    "D: Yeo-Johnson": PowerTransformer(method="yeo-johnson").fit_transform(imputed),
}
option_frames = {name: pd.DataFrame(values, columns=kept_features, index=imputed.index) for name, values in options.items()}

# One-liner: for each option, skew() gives per-feature skewness; concat puts them side by side as columns
skew_table = pd.concat({name: frame.skew() for name, frame in option_frames.items()}, axis=1).round(2)
summary = pd.DataFrame({
    name: {
        "mean |skew|": frame.skew().abs().mean(),   # average lopsidedness across features
        "max |z-score|": frame.abs().max().max(),   # the single most extreme value anywhere
    }
    for name, frame in option_frames.items()
}).round(2)
display(skew_table)
display(summary)
''')

step(cells,
     "Runs a quick KMeans (k=4) on each option as a sanity check and reports the silhouette score.",
     "Whether any option is clearly better or worse, keeping in mind these scores come from different spaces.",
     "Silhouette (from -1 to 1, higher means tighter, better-separated clusters) is not fully comparable across different transforms, so we use it only as a sanity check, not as the decision rule. The decision rests on skewness, extremes and interpretability.",
     '''
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sanity = {}
for name, frame in option_frames.items():
    labels = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(frame)
    # sample_size keeps this fast; random_state makes the sample repeatable
    sanity[name] = silhouette_score(frame, labels, sample_size=4000, random_state=0)
pd.Series(sanity, name="silhouette (k=4)").round(3).to_frame()
''')

step(cells,
     "Draws before-and-after histograms for three of the most skewed features: raw, Option A, and Option B.",
     "The raw and A histograms (same shape, one huge spike near 0), versus B (a spread-out shape).",
     "This is the visual proof of what the skewness table says: scaling alone does not change the shape, a log transform does.",
     '''
import matplotlib.pyplot as plt  # imported here because plotting starts in this cell
import seaborn as sns

examples = ["MINIMUM_PAYMENTS", "ONEOFF_PURCHASES", "CASH_ADVANCE"]
fig, axes = plt.subplots(len(examples), 3, figsize=(13, 8))
for row, feature in enumerate(examples):
    sns.histplot(imputed[feature], bins=50, ax=axes[row, 0], color="gray")
    axes[row, 0].set_title(f"{feature}: raw")
    sns.histplot(option_frames["A: Standard only"][feature], bins=50, ax=axes[row, 1], color="steelblue")
    axes[row, 1].set_title(f"{feature}: A, Standard only")
    sns.histplot(option_frames["B: log1p + Standard"][feature], bins=50, ax=axes[row, 2], color="seagreen")
    axes[row, 2].set_title(f"{feature}: B, log1p + Standard")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "transform_comparison.png", dpi=110)
plt.show()
''')

maybe("findings_02_step5")

step(cells,
     "Builds the final preprocessing as one scikit-learn Pipeline: impute, then (log1p + scale) for skewed columns and (scale) for bounded columns.",
     "The output shape (8,950 by 11), column means near 0 and standard deviations near 1.",
     "A Pipeline bundles every step. Fitted once, it can be saved and used on any new customer in exactly the same way: it reuses the training medians, the training scales, and applies the same log. ColumnTransformer sends different columns through different steps.",
     '''
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Branch 1: log1p then z-score, for the skewed money columns
log_branch = Pipeline([
    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scale", StandardScaler()),
])
# Branch 2: only z-score, for the bounded columns
plain_branch = Pipeline([("scale", StandardScaler())])

# Send each group of columns through its own branch; remainder="drop" throws away the 6 redundant columns
column_steps = ColumnTransformer(
    transformers=[("log", log_branch, log_features), ("plain", plain_branch, plain_features)],
    remainder="drop",
    verbose_feature_names_out=False,  # keep the plain column names
)

full_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),  # fills blanks using the training medians
    ("transform", column_steps),                   # log1p + scale, or scale only, depending on the column
]).set_output(transform="pandas")  # keep column names between steps
''')

step(cells,
     "Fits the pipeline on the raw data and checks the result.",
     "Shape (8,950, 11), no missing values, mean near 0 and standard deviation near 1 for every column.",
     "This is the table KMeans will use. Each column is now on the same scale, so no single feature can dominate the distance.",
     '''
scaled = full_pipeline.fit_transform(raw)  # learn medians, scales; return the scaled table
print(f"{scaled.shape=}, {scaled.isna().sum().sum()=}")
scaled.describe().loc[["mean", "std", "min", "max"]].round(2)
''')

step(cells,
     "Tests that the pipeline treats a brand new customer the same way, including a blank value.",
     "The transformed row should have finite numbers and the same column order as the training table.",
     "This proves reuse: in real life a new customer arrives with raw numbers and maybe a blank, and the saved pipeline can score them.",
     '''
new_customer = raw.iloc[[0]].copy()
new_customer["MINIMUM_PAYMENTS"] = np.nan  # pretend this value is missing
print(full_pipeline.transform(new_customer).round(2).T.rename(columns={new_customer.index[0]: "scaled value"}))
''')

step(cells,
     "Saves the fitted pipeline, the scaled data and the imputed original-unit data to data/processed/.",
     "That three files exist.",
     "Notebook 03 loads the scaled table to cluster, and notebook 04 loads the original-unit table so explanations are in real dollars. Saving them means each notebook can run on its own.",
     '''
import joblib  # imported here because this is the only place we save a model object

joblib.dump(full_pipeline, PROCESSED_DIR / "preprocessing_pipeline.joblib")   # the fitted recipe
scaled.to_csv(PROCESSED_DIR / "credit_card_scaled.csv")                       # for clustering
imputed_all.to_csv(PROCESSED_DIR / "credit_card_imputed_original_units.csv")  # all 17 columns, for profiling and SHAP
print(sorted(path.name for path in PROCESSED_DIR.glob("*") if path.name != ".gitkeep"))
''')

text(cells, """
---
## Step 6: Dimensionality reduction (for visualization and for checking structure)

*Dimensionality reduction* squeezes many columns into a few while keeping as much information as possible. **PCA** (principal component analysis) finds new axes, called components, that are combinations of the original columns. Component 1 captures the most variation, component 2 the next most, and so on, and they are uncorrelated with each other.

**Important:** we use PCA to *look* at the data and to check how much structure there is. We do **not** cluster on the PCA output. KMeans runs on all 11 scaled features.
""")

step(cells,
     "Runs PCA on the scaled data and plots the explained variance per component and the cumulative total.",
     "How quickly the bars shrink, and how many components reach 80% and 90% of the variance.",
     "*Explained variance* is the share of the total spread in the data that one component captures. If a few components capture most of it, the data has simple structure. If it needs many, the information is spread over many features.",
     '''
from sklearn.decomposition import PCA

pca = PCA(random_state=0).fit(scaled)  # keep all 11 components for now
explained = pca.explained_variance_ratio_
cumulative = explained.cumsum()
components_for_80 = int(np.searchsorted(cumulative, 0.80) + 1)  # first position where the running total passes 80%
components_for_90 = int(np.searchsorted(cumulative, 0.90) + 1)
print(f"{components_for_80=}, {components_for_90=}")

fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5))
left.bar(range(1, len(explained) + 1), explained * 100, color="steelblue")
left.set(xlabel="Component", ylabel="Variance explained (%)", title="Explained variance per component")
right.plot(range(1, len(cumulative) + 1), cumulative * 100, marker="o", color="seagreen")
right.axhline(80, color="gray", linestyle="--", label="80%")
right.axhline(90, color="gray", linestyle=":", label="90%")
right.set(xlabel="Number of components", ylabel="Cumulative variance (%)", title="Cumulative explained variance")
right.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "pca_variance.png", dpi=110)
plt.show()
pd.DataFrame({"explained_%": (explained * 100).round(1), "cumulative_%": (cumulative * 100).round(1)}, index=[f"PC{i}" for i in range(1, 12)])
''')

step(cells,
     "Shows which original features drive the first three components (their loadings).",
     "The largest positive and negative numbers in each column.",
     "A loading says how much a feature contributes to a component. If PC1 is led by purchase features, then moving along PC1 means 'buys more or less'. This helps us read the 2D plot.",
     '''
loadings = pd.DataFrame(pca.components_[:3].T, index=kept_features, columns=["PC1", "PC2", "PC3"]).round(2)
loadings.sort_values("PC1", key=abs, ascending=False)
''')

step(cells,
     "Projects every customer onto the first two components and plots them, once as points and once as a density map.",
     "Is there one blob, several blobs, or a smooth cloud? Are there dense spots and thin edges?",
     "This plot is for seeing structure, not for final clustering. Two components keep only part of the information, so groups that overlap here may still be separate in the full 11 dimensions, and groups that look separate here might not be.",
     '''
coordinates = pca.transform(scaled)[:, :2]  # first two components for each customer

fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
left.scatter(coordinates[:, 0], coordinates[:, 1], s=4, alpha=0.25, color="steelblue")
left.set(xlabel=f"PC1 ({explained[0]:.0%})", ylabel=f"PC2 ({explained[1]:.0%})", title="Customers in 2D (points)")
density = right.hexbin(coordinates[:, 0], coordinates[:, 1], gridsize=45, cmap="viridis", mincnt=1, bins="log")
right.set(xlabel=f"PC1 ({explained[0]:.0%})", ylabel=f"PC2 ({explained[1]:.0%})", title="Customers in 2D (density, log colour scale)")
fig.colorbar(density, ax=right, label="customers (log scale)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "pca_2d.png", dpi=110)
plt.show()
''')

maybe("findings_02_step6")

save(cells, Path(__file__).parent.parent / "notebooks" / "02_preprocessing.ipynb")
print("built")
