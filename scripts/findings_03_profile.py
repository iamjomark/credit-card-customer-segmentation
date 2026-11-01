text(cells, """
### Reading the profile

How each group differs, from the medians (typical member) and means (where the money is):

| Cluster | Share | What stands out | Name |
|---|---|---|---|
| 3 | 18.3% | Highest balance (median 2,716), highest purchases (median 1,705, mean 2,663), highest credit limit (7,000) and highest payments, **and** also cash advances (mean 1,561). Almost never pays in full (mean 3%). | **Big spenders, high limit, carry a balance** |
| 2 | 13.9% | Buys in almost every period (purchase frequency 1.0) with real volume (median 1,043), yet has the lowest balance of the active groups (median 94) because it scores high on the full-payment measure `PRC_FULL_PAYMENT` (median 0.80). Limit 4,500. | **Frequent buyers, pay in full** |
| 1 | 22.9% | Regular but moderate purchases (median 497, frequency 0.67), no cash advances, a small limit (2,000), and a balance that is carried (never pays in full). | **Steady everyday spenders** |
| 4 | 26.9% | Almost no purchases (mean frequency 0.06) but the biggest cash advances (median 1,275, mean 2,107), a balance of 1,765 that is carried, and limit 3,000. | **Cash advance borrowers** |
| 0 | 6.6% | Short tenure (median **7 months**, versus 12 for everyone else), the smallest limit (1,500), and a mix of small purchases and cash advances. | **Newer customers, small limit, mixed use** |
| 5 | 11.3% | Balance of about 14 and a balance that is rarely updated (balance frequency 0.36), small purchases (frequency 0.17), no cash advances. | **Dormant, barely used card** |

**Why these names, and what to keep in mind.** A name should tell a manager what the group does with the card, in a few words. They come from three things: how much and how often the group buys, whether it borrows (cash advance or carried balance), and whether it pays in full. Cluster 3 mixes two roles (heavy spending and heavy borrowing), which is why its name carries both. Cluster 0 is separated mainly by its short tenure, so we should not read too much into how it spends, since a newer customer has had less time to build up a record. The cluster numbers (0 to 5) are arbitrary labels from KMeans and could change if the model is refit with another seed. The names are what stay meaningful.

*Measured from the data:* only cluster 2 scores high on the full-payment measure `PRC_FULL_PAYMENT` (mean 0.75), while every other group averages between 0.03 and 0.15. So paying in full is a defining trait of one group only. (There is no data dictionary, so we read this column as the share of the bill, or of billing periods, paid in full.)
""")

step(cells,
     "Saves the cluster names and the profile tables, so the next notebook and the README can use them.",
     "The printed dictionary of names.",
     "Names are saved next to the data so notebook 04 uses exactly the same words. The profile tables go to reports/ for the write-up.",
     '''
import json  # json saves and reads the cluster names as a small text file

cluster_names = {
    0: "Newer customers, small limit, mixed use",
    1: "Steady everyday spenders",
    2: "Frequent buyers, pay in full",
    3: "Big spenders, high limit, carry a balance",
    4: "Cash advance borrowers",
    5: "Dormant, barely used card",
}
(PROCESSED_DIR / "cluster_names.json").write_text(json.dumps(cluster_names, indent=2))  # names for notebook 04

named_profile = cluster_medians.T.copy()
named_profile.columns = [f"{cluster}: {cluster_names[cluster]}" for cluster in named_profile.columns]
named_profile.to_csv(FIGURES_DIR.parent / "cluster_profile_medians.csv")  # reports/cluster_profile_medians.csv
cluster_means.T.to_csv(FIGURES_DIR.parent / "cluster_profile_means.csv")
size_table.assign(name=size_table.index.map(cluster_names))
''')
