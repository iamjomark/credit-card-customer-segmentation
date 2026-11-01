text(cells, """
### What the surrogate tells us (and its fidelity)

- **Fidelity is high: 95.0% on customers it never saw** (cross-validated: 95.1%, folds between 94.5% and 95.8%), against a 26.9% baseline for always guessing the biggest cluster. In other words, the surrogate reproduces KMeans's cluster for 19 customers out of 20. Recall (how many of a cluster's members it found) ranges from 92.1% (dormant) to 97.0% (cash advance borrowers).
- **Where it goes wrong:** the top confusion is 17 big spenders (of 409) predicted as steady everyday spenders, followed by smaller mixes of the same neighbours. Mistakes happen between clusters that sit next to each other, which fits the touching clouds we saw in PCA.
- **How to read the 95%.** KMeans is itself a fixed rule on these same features (nearest centre), so a flexible model learning that rule is expected to do well. High accuracy tells us the clusters are *describable* with the features. It does **not** prove the clusters are meaningful or that customers truly fall into six types. A low score would have been the warning sign, and we do not have one.
""")
