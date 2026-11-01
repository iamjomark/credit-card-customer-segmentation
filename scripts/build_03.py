"""Builds notebooks/03_clustering.ipynb. Run: uv run python scripts/build_03.py
Pass 1 builds the metric cells. After reading them, scripts/chosen_k.txt is created and part 2 is added."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbtools import save, step, text

cells: list = []
HERE = Path(__file__).parent


def maybe(name: str) -> None:
    """Insert a hand-written interpretation file (written AFTER reading the executed output), if it exists."""
    path = HERE / f"{name}.py"
    if path.exists():
        exec(path.read_text(), globals())


text(cells, """
# 03 Clustering and cluster profiling

**Clustering** means grouping customers so that people in the same group are alike and people in different groups are not. There are no right answers to check against (no labels), so we judge the groups with *internal* measures that only use the data itself, plus a stability test.

We use **KMeans**: pick k centre points, assign each customer to the nearest centre, move each centre to the average of its customers, and repeat until nothing changes. We must choose k ourselves, which is what Step 7 is about.

## Step 7: Clustering and metrics
""")

step(cells,
     "Loads the scaled table made in notebook 02.",
     "8,950 rows and 11 columns, no missing values.",
     "This is the data every clustering method below uses. Because it was saved to disk, this notebook runs on its own.",
     '''
# Imported here so this step only loads the libraries when it runs
from pathlib import Path
import numpy as np
import pandas as pd


def find_project_root(start: Path = Path.cwd()) -> Path:
    """Walk up from the current folder until we see pyproject.toml (nbconvert runs inside notebooks/)."""
    return next(folder for folder in [start, *start.parents] if (folder / "pyproject.toml").exists())


ROOT = find_project_root()
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

scaled = pd.read_csv(PROCESSED_DIR / "credit_card_scaled.csv", index_col=0)  # CUST_ID is the row label
print(f"{scaled.shape=}, {scaled.isna().sum().sum()=}")
RANDOM_STATE = 42  # fixed so the results can be repeated
K_VALUES = range(2, 11)
''')

text(cells, """
### The four measures

| Measure | What it asks | Better is |
|---|---|---|
| **Inertia** | How far, in total, are customers from their own cluster centre? (Always falls as k grows, so we look for the *elbow* where the gains slow down.) | lower, but only until the elbow |
| **Silhouette** (-1 to 1) | For each customer: am I closer to my own cluster than to the next nearest one? | higher |
| **Davies-Bouldin** | How much do clusters overlap relative to how spread out they are? | lower |
| **Calinski-Harabasz** | Ratio of the spread *between* clusters to the spread *inside* them. | higher |
""")

step(cells,
     "Fits KMeans for every k from 2 to 10 and computes all four measures for each.",
     "How each column changes with k. Note where silhouette peaks, where Davies-Bouldin is lowest and where Calinski-Harabasz peaks.",
     "The measures often disagree. When they do, we weigh them together with how useful the segments are for the business, rather than trusting one number.",
     '''
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

kmeans_rows = []
kmeans_labels = {}  # keep the labels for later use
for k in K_VALUES:
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(scaled)  # n_init=10: try 10 starts, keep the best
    kmeans_labels[k] = model.labels_
    kmeans_rows.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette": silhouette_score(scaled, model.labels_),
        "davies_bouldin": davies_bouldin_score(scaled, model.labels_),
        "calinski_harabasz": calinski_harabasz_score(scaled, model.labels_),
        "smallest_cluster_%": np.bincount(model.labels_).min() / len(scaled) * 100,  # guards against tiny clusters
    })
kmeans_table = pd.DataFrame(kmeans_rows).set_index("k")
kmeans_table.round(3)
''')

step(cells,
     "Finds the elbow automatically with the kneed library and plots all four measures against k.",
     "A bend in the inertia curve (the dashed line marks kneed's pick) and the peaks or dips in the other three panels.",
     "kneed looks for the point where the curve bends most, meaning extra clusters stop paying off (the 'knee'). It is a helpful hint, not a verdict, because real elbows are often gentle.",
     '''
import matplotlib.pyplot as plt  # imported here because plotting starts in this cell
from kneed import KneeLocator   # automatic elbow finder

knee = KneeLocator(list(K_VALUES), kmeans_table["inertia"].tolist(), curve="convex", direction="decreasing")
print(f"{knee.knee=}")

fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
plots = [("inertia", "Inertia (lower, find the elbow)"), ("silhouette", "Silhouette (higher is better)"),
         ("davies_bouldin", "Davies-Bouldin (lower is better)"), ("calinski_harabasz", "Calinski-Harabasz (higher is better)")]
for axis, (column, title) in zip(axes, plots):
    axis.plot(kmeans_table.index, kmeans_table[column], marker="o", color="steelblue")
    axis.set(title=title, xlabel="k")
if knee.knee:
    axes[0].axvline(knee.knee, color="red", linestyle="--")  # kneed's elbow pick
fig.tight_layout()
fig.savefig(FIGURES_DIR / "kmeans_metrics.png", dpi=110)
plt.show()
''')

text(cells, """
### Stability: would we get the same groups on different customers?

Clusters that appear only for one particular sample are a fluke. To test this, we repeat the following 30 times for each k:

1. Draw a *bootstrap sample* (a random sample of 8,950 customers drawn **with replacement**, so some customers appear twice and others not at all).
2. Fit KMeans on that sample only.
3. Use that model to assign **all** 8,950 customers to a cluster.
4. Compare those labels with our reference labels (from the fit on all data) using the **Adjusted Rand Index (ARI)**.

ARI measures how much two ways of grouping the same customers agree. 1 means identical groupings, 0 means no better than random. Cluster *numbers* do not matter (cluster 2 in one run can be cluster 5 in another), only who is grouped with whom. Each bootstrap fit gets the same 10 random starts as the reference fit, so the comparison is fair (with fewer starts a fit can land on a worse solution and look unstable when it is not). We predict on the full data so both sets of labels cover the same customers. (Comparing the labels of two different bootstrap samples directly is a common mistake, because they contain different people.)
""")

step(cells,
     "Runs the bootstrap stability test for every k and summarises the ARI scores.",
     "A high mean ARI (above about 0.8 is solid, below 0.6 is shaky) and a small spread.",
     "High stability means the same groups keep appearing when the customers change slightly, so they are not a fluke. A sharp drop at some k suggests the data does not really support that many groups.",
     '''
from sklearn.metrics import adjusted_rand_score

N_BOOTSTRAPS = 30
random_generator = np.random.default_rng(RANDOM_STATE)  # the modern way to get repeatable randomness
stability_rows = []
for k in K_VALUES:
    reference_labels = kmeans_labels[k]  # labels from the fit on ALL customers
    scores = []
    for _ in range(N_BOOTSTRAPS):
        picked = random_generator.integers(0, len(scaled), size=len(scaled))   # row numbers drawn with replacement
        bootstrap_model = KMeans(n_clusters=k, n_init=10, random_state=int(random_generator.integers(1_000_000))).fit(scaled.iloc[picked])
        predicted_on_all = bootstrap_model.predict(scaled)                     # label every customer with this model
        scores.append(adjusted_rand_score(reference_labels, predicted_on_all)) # agreement with the reference
    stability_rows.append({"k": k, "mean_ARI": np.mean(scores), "std_ARI": np.std(scores), "min_ARI": np.min(scores)})
stability_table = pd.DataFrame(stability_rows).set_index("k")
stability_table.round(3)
''')

text(cells, """
### A second method for comparison

- **Gaussian Mixture Model (GMM):** assumes the data is a mix of k bell-shaped clouds and gives each customer a probability for each cloud. It allows clusters of different sizes and shapes, while KMeans prefers round clusters of similar size.
- **Agglomerative clustering (Ward):** starts with every customer alone and repeatedly merges the two closest groups. Needs no random start.

We compare them with the same internal measures only. We have no true labels, so we do not use accuracy. GMM also reports **BIC** (Bayesian information criterion), a score that rewards a good fit but penalises extra parameters (lower is better).
""")

step(cells,
     "Fits GMM and Agglomerative clustering for k from 2 to 10 and builds one comparison table with the same measures.",
     "Whether either method beats KMeans at the same k on silhouette, Davies-Bouldin and Calinski-Harabasz.",
     "If a different method gives a similar score, the structure is a property of the data, not of KMeans. If it is clearly better, we should think about switching.",
     '''
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

comparison_rows = []
gmm_bic = {}
gmm_labels = {}
agglomerative_labels = {}
for k in K_VALUES:
    gmm = GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=RANDOM_STATE).fit(scaled)
    gmm_labels[k] = gmm.predict(scaled)       # hard assignment: most likely cloud for each customer
    gmm_bic[k] = gmm.bic(scaled)
    agglomerative_labels[k] = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(scaled)
    for method, label_set in [("KMeans", kmeans_labels), ("GMM", gmm_labels), ("Agglomerative", agglomerative_labels)]:
        labels = label_set[k]
        comparison_rows.append({
            "method": method, "k": k,
            "silhouette": silhouette_score(scaled, labels),
            "davies_bouldin": davies_bouldin_score(scaled, labels),
            "calinski_harabasz": calinski_harabasz_score(scaled, labels),
            "smallest_cluster_%": np.bincount(labels).min() / len(labels) * 100,
        })
comparison_table = pd.DataFrame(comparison_rows)
# One-liner: pivot makes one row per k and one column per method for a chosen measure
display(comparison_table.pivot(index="k", columns="method", values="silhouette").round(3).rename_axis(columns="silhouette"))
display(comparison_table.pivot(index="k", columns="method", values="davies_bouldin").round(3).rename_axis(columns="davies_bouldin"))
display(comparison_table.pivot(index="k", columns="method", values="calinski_harabasz").round(0).rename_axis(columns="calinski_harabasz"))
print("GMM BIC by k (lower is better):")
print(pd.Series(gmm_bic).round(0).to_string())
''')

maybe("findings_03_metrics")

if (HERE / "chosen_k.txt").exists():
    exec((HERE / "part2_03_chosen.py").read_text())
    if (HERE / "part3_03_profiling.py").exists():
        exec((HERE / "part3_03_profiling.py").read_text())

save(cells, HERE.parent / "notebooks" / "03_clustering.ipynb")
print("built")
