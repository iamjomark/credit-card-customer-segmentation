# Part 2 of notebook 03 (sensitivity checks, save labels and model). Exec'd by build_03.py.
CHOSEN_K = int((HERE / "chosen_k.txt").read_text())
RUNNER_UP_K = 4

step(cells,
     "Records the k we chose after weighing the metrics, the stability test and the segment descriptions (reasoning in the notes below), plus the runner-up.",
     "The two numbers.",
     "Everything after this cell uses these two values. Keeping them in one place makes the choice easy to change and to audit.",
     f"CHOSEN_K = {CHOSEN_K}  # our choice (see the reasoning below)\nRUNNER_UP_K = {RUNNER_UP_K}  # the simpler alternative we check against")

text(cells, """
### Robustness checks: do our earlier choices change the answer?

In notebooks 01 and 02 we made three choices that we said we would test. For each one we rebuild the data, refit KMeans with the same settings, and measure the **Adjusted Rand Index** against our main grouping. Here ARI is an *agreement between two groupings of the same customers*, not an accuracy (we have no true labels). 1 means the same groups, and values near 0 mean the groups are unrelated.

1. **Drop `MINIMUM_PAYMENTS`**, which is correlated 0.90 with `BALANCE`.
2. **Fill missing `MINIMUM_PAYMENTS` with 0** for customers who made no payments (instead of the median).
3. **Yeo-Johnson** (the runner-up transform) instead of log1p.
""")

step(cells,
     "Rebuilds the data three alternative ways, refits KMeans for the chosen k and the runner-up k, and compares each result with the main grouping.",
     "ARI values near 1 (the answer barely changes) versus values below about 0.7, and the '% moved' columns, which say directly what share of customers end up in a different group.",
     "If the groups survive these changes, they are a property of the customers and not of one preprocessing decision. If a choice changes them a lot, we must say so in the limitations.",
     '''
import joblib  # imported here to load the saved preprocessing pipeline from notebook 02
from scipy.optimize import linear_sum_assignment  # imported here to match old and new cluster numbers fairly
from sklearn.base import clone  # imported here to refit the same recipe on changed data
from sklearn.metrics import adjusted_rand_score  # imported here to measure agreement between two groupings
from sklearn.preprocessing import PowerTransformer, StandardScaler  # imported here for the Yeo-Johnson variants

raw = pd.read_csv(ROOT / "data" / "raw" / "credit_card.csv").set_index("CUST_ID")
saved_pipeline = joblib.load(PROCESSED_DIR / "preprocessing_pipeline.joblib")
imputed_original = pd.read_csv(PROCESSED_DIR / "credit_card_imputed_original_units.csv", index_col=0)

# Variant 2: customers with no payments and a blank minimum payment get 0 instead of the median
raw_zero_fill = raw.copy()
no_payment_blank = raw_zero_fill["MINIMUM_PAYMENTS"].isna() & (raw_zero_fill["PAYMENTS"] == 0)
raw_zero_fill.loc[no_payment_blank, "MINIMUM_PAYMENTS"] = 0
print(f"{no_payment_blank.sum()=} customers filled with 0")

# The seven skewed money columns that Option B logs (list from notebook 02); the other four are bounded
log_features = ["BALANCE", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"]
plain_features = [name for name in scaled.columns if name not in log_features]

# Like-for-like Yeo-Johnson: replaces log1p on the SAME seven columns, and leaves the four bounded columns with plain scaling
yeo_johnson_money_only = pd.concat([
    pd.DataFrame(PowerTransformer(method="yeo-johnson").fit_transform(imputed_original[log_features]), columns=log_features, index=scaled.index),
    pd.DataFrame(StandardScaler().fit_transform(imputed_original[plain_features]), columns=plain_features, index=scaled.index),
], axis=1)[scaled.columns]

variants = {
    "Drop MINIMUM_PAYMENTS": scaled.drop(columns="MINIMUM_PAYMENTS"),
    "Zero-fill MINIMUM_PAYMENTS": clone(saved_pipeline).fit_transform(raw_zero_fill),  # same recipe, refit on changed data
    "Yeo-Johnson instead of log1p (same 7 columns)": yeo_johnson_money_only,
    "Yeo-Johnson on all 11 columns": pd.DataFrame(
        PowerTransformer(method="yeo-johnson").fit_transform(imputed_original[scaled.columns]),
        columns=scaled.columns, index=scaled.index),
}


def share_reassigned(main_labels, new_labels) -> float:
    """Share of customers whose group changes after matching the new cluster numbers to the old ones as well as possible."""
    overlap = pd.crosstab(main_labels, new_labels).values            # how many customers each old/new pair share
    old_index, new_index = linear_sum_assignment(-overlap)           # best one-to-one matching (minus sign: maximise overlap)
    return 1 - overlap[old_index, new_index].sum() / overlap.sum()   # everyone not in a matched pair has moved


sensitivity_rows = []
for variant_name, variant_data in variants.items():
    row = {"variant": variant_name}
    for k in (CHOSEN_K, RUNNER_UP_K):
        variant_labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(variant_data)
        row[f"ARI at k={k}"] = adjusted_rand_score(kmeans_labels[k], variant_labels)  # vs the main grouping at the same k
        row[f"% moved at k={k}"] = share_reassigned(kmeans_labels[k], variant_labels) * 100  # share of customers who change group
    sensitivity_rows.append(row)
pd.DataFrame(sensitivity_rows).set_index("variant").round(3)
''')

maybe("findings_03_decision")

step(cells,
     "Fits the final KMeans model with the chosen k and saves the cluster labels and the model to data/processed/.",
     "The size of each cluster, and that the two saved files exist.",
     "Notebook 04 loads these files to explain the clusters. The model's centroids (cluster centres) follow the column order of the scaled table, which puts the log-transformed columns first.",
     '''
final_model = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=RANDOM_STATE).fit(scaled)
cluster_labels = pd.Series(final_model.labels_, index=scaled.index, name="cluster")
assert (cluster_labels.values == kmeans_labels[CHOSEN_K]).all()  # same settings as the table above, so identical labels

cluster_labels.to_csv(PROCESSED_DIR / "cluster_labels.csv")                 # one row per customer
joblib.dump(final_model, PROCESSED_DIR / f"kmeans_k{CHOSEN_K}.joblib")      # the fitted model (centroids inside)
print(sorted(path.name for path in PROCESSED_DIR.glob("*") if path.name != ".gitkeep"))
cluster_labels.value_counts().sort_index().to_frame("customers").assign(percent=lambda t: (t["customers"] / len(scaled) * 100).round(1))
''')
