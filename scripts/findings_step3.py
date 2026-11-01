text(cells, """
### Decision: which features to keep

Nine pairs are above 0.8. They fall into clear families. Our rule: when two features are the *same quantity counted differently* (an amount versus a count or a frequency), keep one and drop the others, so one behaviour does not get counted two or three times in the distance.

| Pair (Spearman) | Decision | Reason |
|---|---|---|
| `CASH_ADVANCE_FREQUENCY` and `CASH_ADVANCE_TRX` (0.98), `CASH_ADVANCE` with each (0.94, 0.95) | **Keep `CASH_ADVANCE`, drop the other two** | Three versions of "how much cash does this customer borrow". The amount is the one with direct business meaning (money borrowed at a high rate). |
| `PURCHASES` and `PURCHASES_TRX` (0.89), `PURCHASES_FREQUENCY` and `PURCHASES_TRX` (0.92) | **Drop `PURCHASES` and `PURCHASES_TRX`, keep `PURCHASES_FREQUENCY`** | `PURCHASES` is (almost exactly) one-off plus installments, which we keep separately, so it adds no new information. The transaction count says what the frequency already says. |
| `ONEOFF_PURCHASES` and `ONEOFF_PURCHASES_FREQUENCY` (0.95) | **Keep the amount, drop the frequency** | Same behaviour, and overall purchase frequency is kept through `PURCHASES_FREQUENCY`. |
| `INSTALLMENTS_PURCHASES` and `PURCHASES_INSTALLMENTS_FREQUENCY` (0.92), `PURCHASES_FREQUENCY` and `PURCHASES_INSTALLMENTS_FREQUENCY` (0.85) | **Keep the amount, drop the frequency** | Same reasoning, and this also resolves the 0.85 pair. |
| `BALANCE` and `MINIMUM_PAYMENTS` (0.90) | **Keep both** | This is the one exception. They are different things (what is owed versus the smallest payment due), 0.90 is close to our line, and the minimum payment helps show who under-pays. We will check in notebook 03 that dropping it does not change the clusters. |

**Result:** we drop 6 features (`PURCHASES`, `PURCHASES_TRX`, `CASH_ADVANCE_TRX`, `CASH_ADVANCE_FREQUENCY`, `ONEOFF_PURCHASES_FREQUENCY`, `PURCHASES_INSTALLMENTS_FREQUENCY`) and keep 11 for clustering. In Step 8 we still show the dropped columns when we describe each cluster, because they are useful for explaining it in business terms.
""")

step(cells,
     "Records the six dropped features in one place and checks which strong pairs remain among the 11 kept features.",
     "Only the BALANCE and MINIMUM_PAYMENTS pair should remain above 0.8.",
     "This proves the drop list does what we claimed, instead of trusting the table above by eye.",
     '''
FEATURES_TO_DROP = [
    "PURCHASES", "PURCHASES_TRX", "CASH_ADVANCE_TRX",
    "CASH_ADVANCE_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
]
kept_features = [name for name in features if name not in FEATURES_TO_DROP]
print(f"{len(kept_features)=}")

kept_pairs = high_pairs[high_pairs["feature_a"].isin(kept_features) & high_pairs["feature_b"].isin(kept_features)]
print("Strong pairs left among kept features:")
print(kept_pairs.to_string(index=False))
''')
