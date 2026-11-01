text(cells, """
### What permutation importance tells us

- **The two methods agree on the ranking** (Spearman 0.85, where 1 is identical order). Both put `PURCHASES_FREQUENCY` first (shuffling it costs 17.2 points of accuracy) and `ONEOFF_PURCHASES` last (1.8 points). `PRC_FULL_PAYMENT` and `CASH_ADVANCE` take places 2 and 3 in both.
- **Where they differ:** `TENURE` is 4th for permutation (10.1 points) but 7th for SHAP. Shuffling tenure creates customers with the wrong tenure, and the surrogate sends many of them to the newer-customer cluster, so the accuracy drop is large even though only one cluster depends on it. `INSTALLMENTS_PURCHASES` goes the other way (8th versus 5th), most likely because `PURCHASES_FREQUENCY` is correlated with it and can cover for it when it is shuffled. This is the weakness of permutation importance with correlated features, noted above.

### Limitations of these explanations

- They describe **the clusters KMeans produced** and **what the surrogate learned about them**. They are not a ground truth about why customers behave as they do, and they do not show cause and effect.
- **Fidelity:** the surrogate matches KMeans on 95.0% of unseen customers. For the other 5%, the explanations may describe a different cluster than the one the customer is actually in.
- **Correlated features** (for example purchase frequency and installment purchases) can share credit with each other in SHAP and hide each other in permutation importance, so single-feature rankings should not be over-read.
- The explanations rely on the feature set, the transform, and the choice of k=6 made earlier. If any of them changed, the explanations would change too (the Yeo-Johnson check in notebook 03 showed the boundaries are not perfectly fixed).
""")
