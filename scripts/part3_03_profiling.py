# Step 8 of notebook 03: cluster profiling. Exec'd by build_03.py.
text(cells, """
---
## Step 8: Cluster profiling

Now we describe who is in each cluster, in plain money terms. We use the **original dollar values** (after filling blanks) and all 17 original features, including the six we dropped for clustering, because they help explain each group even though they were not used to build it.
""")

step(cells,
     "Loads the original-unit data and the cluster labels, and joins them into one table.",
     "17 feature columns plus a `cluster` column, and 8,950 rows.",
     "Each customer now has their real values and their cluster, which is what we need to describe the groups.",
     '''
imputed_original = pd.read_csv(PROCESSED_DIR / "credit_card_imputed_original_units.csv", index_col=0)  # all 17, blanks filled
labels = pd.read_csv(PROCESSED_DIR / "cluster_labels.csv", index_col=0)["cluster"]
profile_data = imputed_original.join(labels)  # align on CUST_ID
print(f"{profile_data.shape=}, {profile_data['cluster'].nunique()=}")
''')

step(cells,
     "Computes the size of each cluster as a percentage of customers, and the mean and median of every feature per cluster.",
     "Compare the mean with the median within a cluster: a mean far above the median means a few very large customers inside that group.",
     "Medians describe the typical member. Means include the big customers, so they show where the money is. We use both on purpose.",
     '''
cluster_sizes = profile_data["cluster"].value_counts().sort_index()
size_table = pd.DataFrame({"customers": cluster_sizes, "percent": (cluster_sizes / len(profile_data) * 100).round(1)})
display(size_table)

cluster_means = profile_data.groupby("cluster").mean().round(2)      # average per feature, per cluster
cluster_medians = profile_data.groupby("cluster").median().round(2)  # typical customer per feature, per cluster
print("MEDIANS (the typical customer in each cluster)")
display(cluster_medians.T)   # .T puts features on the rows so the table fits on screen
print("MEANS (pulled up by the biggest customers)")
display(cluster_means.T)
''')

step(cells,
     "Draws a heatmap of cluster means. Each feature (row) is rescaled from 0 to 1 across the six clusters.",
     "Dark cells mean a cluster is the highest on that feature, pale cells the lowest. Read across a row to see which clusters stand out.",
     "Rescaling per feature lets us compare features that have very different units (dollars versus shares). It shows only the *ranking between clusters*, not how large the gap is, so read it together with the tables above.",
     '''
import matplotlib.pyplot as plt  # imported here because plotting starts in this cell
import seaborn as sns

feature_means = profile_data.groupby("cluster").mean()
# One-liner: (value - column min) / (column max - column min) rescales each feature to 0..1 across clusters
means_scaled = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min())

plt.figure(figsize=(9, 8))
sns.heatmap(means_scaled.T, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "0 = lowest cluster, 1 = highest cluster"})
plt.title("Cluster means, scaled per feature (0 to 1)")
plt.xlabel("Cluster")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "cluster_heatmap.png", dpi=110)
plt.show()
''')

maybe("findings_03_profile")
