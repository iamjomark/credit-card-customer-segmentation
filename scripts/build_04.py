"""Builds notebooks/04_explainability.ipynb. Run: uv run python scripts/build_04.py"""
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
# 04 Explainability for unsupervised learning

## Step 9: Why did customers end up in these clusters?

**XAI** (explainable AI) means tools that show *why* a model gave its answer. KMeans has no built-in "feature importance", because it never predicted anything from labels. So we use four indirect methods:

1. **Surrogate model.** We train a supervised model (a Random Forest) to *predict the cluster label* from the customer's features. If it predicts well, the clusters can be described by simple rules on the features.
2. **SHAP** (SHapley Additive exPlanations). For each customer and each feature, SHAP gives a number that says how much that feature pushed the surrogate towards a cluster. It is based on fair-share ideas from game theory.
3. **Distance contributions.** KMeans assigns a customer to the nearest cluster centre. We split that distance feature by feature to see which features pull a customer towards their own centre. This one explains KMeans directly.
4. **Permutation importance.** Shuffle one feature at a time and see how much the surrogate's accuracy drops. A big drop means the feature matters. It is a cross-check on SHAP.

**Limitation to keep in mind:** these tools explain the clusters *the model found* and the *surrogate's behaviour*. They do not reveal a hidden truth about customers. We report how faithful the surrogate is (its *fidelity*) so you know how far to trust them.
""")

step(cells,
     "Loads the original-unit data (11 clustering features), the cluster labels, the cluster names, the scaled data and the saved KMeans model.",
     "11 feature columns, 8,950 rows, and six cluster names.",
     "We use the features in their original dollar units for the surrogate, so every explanation later reads in real money. Tree models are not affected by log or scaling (they only compare 'bigger than', not distances), so this does not change which splits they find.",
     '''
# Imported here so this step only loads the libraries when it runs
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def find_project_root(start: Path = Path.cwd()) -> Path:
    """Walk up from the current folder until we see pyproject.toml (nbconvert runs inside notebooks/)."""
    return next(folder for folder in [start, *start.parents] if (folder / "pyproject.toml").exists())


ROOT = find_project_root()
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"
RANDOM_STATE = 42

scaled = pd.read_csv(PROCESSED_DIR / "credit_card_scaled.csv", index_col=0)
imputed_original = pd.read_csv(PROCESSED_DIR / "credit_card_imputed_original_units.csv", index_col=0)
labels = pd.read_csv(PROCESSED_DIR / "cluster_labels.csv", index_col=0)["cluster"]
cluster_names = {int(key): value for key, value in json.loads((PROCESSED_DIR / "cluster_names.json").read_text()).items()}
kmeans_model = joblib.load(PROCESSED_DIR / "kmeans_k6.joblib")

features = scaled.columns.tolist()          # same 11 columns, same order the model was trained on
original_units = imputed_original[features]  # the same 11 features, in dollars and counts
print(f"{original_units.shape=}, {labels.shape=}")
print(cluster_names)
''')

text(cells, """
### 1. Surrogate model

We split customers into a **training set** (75%) to build the surrogate and a **test set** (25%) it has never seen. Accuracy on the test set is the honest measure of how well the clusters can be recreated from the features. `stratify` keeps the same cluster proportions in both parts.

We also run **cross-validation**: the data is split into 5 parts, and the model is trained on 4 and tested on the 5th, five times in turn. If the five scores are similar, the result does not depend on one lucky split.
""")

step(cells,
     "Splits the customers, trains a Random Forest to predict the cluster, and reports test accuracy, 5-fold cross-validated accuracy and a per-cluster report.",
     "Accuracy far above the 'always guess the biggest cluster' baseline (26.9%), and which clusters have lower recall (the share of that cluster's customers the model found).",
     "High accuracy means the clusters are easy to describe with the features, so explanations based on the surrogate are trustworthy. Low accuracy would be a warning: the explanations would be describing something the surrogate itself cannot reproduce.",
     '''
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    original_units, labels, test_size=0.25, stratify=labels, random_state=RANDOM_STATE)

surrogate = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE)
surrogate.fit(X_train, y_train)

test_accuracy = surrogate.score(X_test, y_test)  # accuracy on customers the model has never seen
baseline = labels.value_counts(normalize=True).max()  # what 'always guess the biggest cluster' would score
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE),
    original_units, labels, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="accuracy")
print(f"{test_accuracy=:.3f}  {baseline=:.3f}")
print(f"{cv_scores.round(3)=}  mean={cv_scores.mean():.3f}  std={cv_scores.std():.3f}")
print(classification_report(y_test, surrogate.predict(X_test), target_names=[cluster_names[c] for c in sorted(cluster_names)], digits=3))
''')

step(cells,
     "Shows the confusion matrix of the surrogate on the test customers.",
     "Strong values on the diagonal, and which pairs of clusters get mixed up (off-diagonal cells).",
     "Rows are the true cluster, columns what the surrogate predicted. Confusion between two clusters means they sit next to each other in feature space, which agrees with the touching clouds we saw in PCA.",
     '''
import matplotlib.pyplot as plt  # imported here because plotting starts in this cell
import seaborn as sns
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, surrogate.predict(X_test), normalize="true")  # each row sums to 1
short_names = [f"{c}: {cluster_names[c][:22]}" for c in sorted(cluster_names)]
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=short_names, yticklabels=short_names)
plt.xlabel("Predicted by surrogate"); plt.ylabel("Actual cluster")
plt.title("Surrogate confusion matrix (share of each actual cluster)")
plt.xticks(rotation=45, ha="right"); plt.tight_layout()
plt.savefig(FIGURES_DIR / "surrogate_confusion.png", dpi=110)
plt.show()
''')

maybe("findings_04_surrogate")

text(cells, """
### 2. SHAP

For each customer, SHAP splits the surrogate's prediction into one contribution per feature. For cluster *c*, a **positive** SHAP value means "this feature value pushed the model towards cluster *c*" and a **negative** one means it pushed away. Contributions are in the model's probability units. We explain the **test customers** (never used to train the surrogate).

`shap.TreeExplainer` is the fast, exact method for tree models such as a Random Forest.
""")

step(cells,
     "Computes SHAP values for the test customers and prints the shape of the result.",
     "The shape should be (customers, features, clusters) = (about 2,238, 11, 6).",
     "Each customer has one SHAP value per feature per cluster. The third axis is the cluster, which we need for the plots below.",
     '''
import shap  # imported here because SHAP is slow to load and only needed from this step on

explainer = shap.TreeExplainer(surrogate)                    # exact explanations for tree ensembles
shap_values = explainer(X_test, check_additivity=False)      # an Explanation object; skip the slow additivity check
print(f"{shap_values.values.shape=}  (customers, features, clusters)")
''')

step(cells,
     "Draws the global SHAP summary: the average size of each feature's contribution, split by cluster.",
     "Which features have the longest bars, and which colours (clusters) dominate each bar.",
     "A long bar means the feature moves predictions a lot, on average. The colours show for which clusters it matters, so a feature may be vital for one cluster and irrelevant for the others.",
     '''
plt.figure()
shap.summary_plot(
    shap_values.values, X_test, plot_type="bar", show=False,
    class_names=[cluster_names[c][:30] for c in sorted(cluster_names)])  # one coloured segment per cluster
plt.title("Global SHAP importance (mean absolute value), split by cluster")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_global_bar.png", dpi=110, bbox_inches="tight")
plt.show()

# The same numbers as a table: mean |SHAP| per feature (rows) and cluster (columns)
mean_abs_shap = pd.DataFrame(
    np.abs(shap_values.values).mean(axis=0), index=features, columns=[cluster_names[c] for c in sorted(cluster_names)])
mean_abs_shap["ALL CLUSTERS"] = mean_abs_shap.sum(axis=1)
mean_abs_shap.sort_values("ALL CLUSTERS", ascending=False).round(3)
''')

step(cells,
     "Draws a beeswarm SHAP plot for the cluster with the most distinctive profile (the 'big spenders'), so we can see direction as well as size.",
     "Each dot is a customer. Red means a high feature value, blue a low one. Dots to the right pushed towards this cluster.",
     "This shows direction: for example, red dots on the right for BALANCE means a high balance pushes a customer into this cluster.",
     '''
BIG_SPENDERS = 3  # cluster number from notebook 03 (named 'Big spenders, high limit, carry a balance')
plt.figure()
shap.plots.beeswarm(shap_values[:, :, BIG_SPENDERS], show=False, max_display=11)
plt.title(f"SHAP for cluster {BIG_SPENDERS}: {cluster_names[BIG_SPENDERS]}")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_beeswarm_cluster3.png", dpi=110, bbox_inches="tight")
plt.show()
''')

step(cells,
     "Picks one typical customer from each cluster (the test customer closest to that cluster's centre) and draws a SHAP waterfall plot for each.",
     "The bars show how each feature pushed the probability of the customer's own cluster up (red) or down (blue), starting from the average and ending at the model's final value f(x).",
     "A waterfall explains one customer. Reading it top to bottom tells the story: 'this customer is in this cluster mainly because of these two or three features'. Feature values are shown in real units, as dollars or counts.",
     '''
# Distance of each test customer to their own cluster centre, in the scaled space KMeans used
test_scaled = scaled.loc[X_test.index]
own_centres = kmeans_model.cluster_centers_[y_test.values]
distance_to_own_centre = pd.Series(np.linalg.norm(test_scaled.values - own_centres, axis=1), index=X_test.index)

for cluster in sorted(cluster_names):
    # the most typical member: the customer closest to the centre of their own cluster
    typical_customer = distance_to_own_centre[y_test == cluster].idxmin()
    row_position = X_test.index.get_loc(typical_customer)
    plt.figure()
    shap.plots.waterfall(shap_values[row_position, :, cluster], max_display=8, show=False)
    plt.title(f"Customer {typical_customer}: why cluster {cluster} ({cluster_names[cluster]})", fontsize=10)
    plt.savefig(FIGURES_DIR / f"shap_waterfall_cluster{cluster}.png", dpi=110, bbox_inches="tight")
    plt.show()
''')

maybe("findings_04_shap")

text(cells, """
### 3. Distance contributions (direct explanation of KMeans)

KMeans puts each customer in the cluster whose centre is closest. The squared distance to a centre is the sum, over all 11 features, of (customer value minus centre value) squared. So we can split that distance **per feature**.

For each customer we compute, for every feature:

> **advantage = (squared gap to the nearest *other* centre) minus (squared gap to their *own* centre)**

A **positive** advantage means this feature makes the customer look more like their own cluster than like the runner-up. A **negative** one means this feature actually pulls them towards the other cluster. Adding up all 11 advantages gives the total reason the customer was assigned where they were. We average these by cluster.
""")

step(cells,
     "Computes per-feature distance contributions for every customer and averages the advantage by cluster.",
     "The sign check at the top (the smallest total advantage should be 0 or above) and the biggest positive values in each row of the table.",
     "Large positive numbers show the features that really hold each cluster together. A feature that is near 0 does not help separate that cluster from its neighbour.",
     '''
centres = kmeans_model.cluster_centers_                         # shape (6 clusters, 11 features)
customer_values = scaled.values                                 # shape (8950, 11)
own_cluster = labels.loc[scaled.index].values

# Squared gap to every centre, feature by feature: shape (customers, clusters, features)
squared_gap = (customer_values[:, None, :] - centres[None, :, :]) ** 2
total_distance = squared_gap.sum(axis=2)                        # squared distance to each centre

# Find each customer's nearest OTHER centre by hiding their own with infinity
distance_to_others = total_distance.copy()
distance_to_others[np.arange(len(scaled)), own_cluster] = np.inf
nearest_other = distance_to_others.argmin(axis=1)

rows = np.arange(len(scaled))
advantage = squared_gap[rows, nearest_other, :] - squared_gap[rows, own_cluster, :]  # (customers, features)
print(f"{advantage.sum(axis=1).min()=:.4f}  (should be 0 or above: every customer is nearest to their own centre)")

advantage_by_cluster = pd.DataFrame(advantage, columns=features, index=scaled.index).groupby(own_cluster).mean()
advantage_by_cluster.index = [f"{c}: {cluster_names[c]}" for c in advantage_by_cluster.index]
advantage_by_cluster.round(2)
''')

step(cells,
     "Draws the distance-contribution table as a heatmap.",
     "Dark red cells: the feature strongly pulls members towards their own cluster. Blue cells: it pulls them towards the neighbouring cluster.",
     "Compare with the SHAP importances. They answer slightly different questions (SHAP: what predicts the cluster; distance: what makes members close to their own centre), so agreement between them makes the story stronger.",
     '''
plt.figure(figsize=(11, 4.5))
sns.heatmap(advantage_by_cluster, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Average distance advantage per feature (positive = pulls members to their own cluster)")
plt.xticks(rotation=40, ha="right"); plt.tight_layout()
plt.savefig(FIGURES_DIR / "distance_contributions.png", dpi=110, bbox_inches="tight")
plt.show()
''')

maybe("findings_04_distance")

text(cells, """
### 4. Permutation importance (cross-check on SHAP)

Permutation importance shuffles one feature's values among the **test** customers, which breaks its link to the cluster, and measures how much accuracy falls. A big drop means the surrogate relied on that feature. A weakness: if two features carry the same information, shuffling one can look harmless because the other still covers for it, so correlated features can be understated.
""")

step(cells,
     "Computes permutation importance on the test set (10 shuffles per feature) and puts it beside the SHAP ranking.",
     "Whether the order of features is broadly the same in both columns, and the rank correlation at the bottom.",
     "If two quite different methods give a similar ranking, we can trust the ranking more. Differences are worth a closer look, often because of correlated features.",
     '''
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

permutation = permutation_importance(surrogate, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="accuracy", n_jobs=-1)
importance_table = pd.DataFrame({
    "permutation_drop_in_accuracy": permutation.importances_mean,   # average accuracy lost when this feature is shuffled
    "permutation_std": permutation.importances_std,
    "mean_abs_SHAP": mean_abs_shap["ALL CLUSTERS"],
}, index=features).sort_values("permutation_drop_in_accuracy", ascending=False)
importance_table["rank_permutation"] = importance_table["permutation_drop_in_accuracy"].rank(ascending=False).astype(int)
importance_table["rank_SHAP"] = importance_table["mean_abs_SHAP"].rank(ascending=False).astype(int)
display(importance_table.round(3))
rho = spearmanr(importance_table["permutation_drop_in_accuracy"], importance_table["mean_abs_SHAP"]).statistic
print(f"Rank agreement between permutation importance and SHAP (Spearman): {rho:.2f}")
''')

maybe("findings_04_permutation")

save(cells, HERE.parent / "notebooks" / "04_explainability.ipynb")
print("built")
