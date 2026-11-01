text(cells, """
### What the distance contributions tell us

The biggest positive numbers in each row are the features that hold the cluster together. They match SHAP almost one for one:

- **Newer customers:** `TENURE` at **12.2**, by far the largest number in the table. On nearly every other feature these customers are closer to their neighbour, the cash advance cluster (negative values for spending, cash advance and payments). In practice this cluster is a **tenure group first** and a spending group second.
- **Frequent buyers, pay in full:** `PRC_FULL_PAYMENT` at **5.6**.
- **Dormant:** `BALANCE_FREQUENCY` at **6.1** and `BALANCE` at **2.4**.
- **Cash advance borrowers:** `PURCHASES_FREQUENCY` (2.5), `INSTALLMENTS_PURCHASES` (2.2) and `CASH_ADVANCE` (1.7).
- **Big spenders:** the pull is spread over many features at once (`CREDIT_LIMIT` 1.4, `ONEOFF_PURCHASES` 0.8, `PAYMENTS` 0.7, `PURCHASES_FREQUENCY` 0.7, `INSTALLMENTS_PURCHASES` 0.7). They are defined by a combination, not by one feature.
- **Steady everyday spenders:** no feature is above 1.0 (`CASH_ADVANCE` 1.0, `CREDIT_LIMIT` 0.9, `PRC_FULL_PAYMENT` 0.9). This is the most "ordinary" group. It is also the nearest neighbour for four of the other five clusters, so it sits in the middle of the map, which fits the 17 big spenders that the surrogate mistook for it.
""")
