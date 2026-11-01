text(cells, """
### What SHAP tells us

- **Global:** the features that move predictions most are `PURCHASES_FREQUENCY` (mean absolute SHAP 0.35), `CASH_ADVANCE` (0.30), `PRC_FULL_PAYMENT` (0.29) and `BALANCE` (0.26). `ONEOFF_PURCHASES` matters least (0.08): once we know *how often* someone buys, the size of their one-off purchases adds little.
- **Each cluster has its own signature feature:**
  - *Newer customers:* `TENURE` (0.09, versus about 0.03 or less for every other cluster).
  - *Frequent buyers, pay in full:* `PRC_FULL_PAYMENT` (0.14, the largest single value in their column).
  - *Dormant:* `BALANCE_FREQUENCY` and `BALANCE`.
  - *Cash advance borrowers:* `PURCHASES_FREQUENCY` (0.15), `CASH_ADVANCE` (0.11) and `INSTALLMENTS_PURCHASES` (0.09): no buying, but cash advances.
  - *Steady everyday spenders:* `CASH_ADVANCE` (the absence of it) and `PRC_FULL_PAYMENT`.
- **Waterfall plots, one typical customer per cluster** (the customer closest to the cluster centre). In every case the surrogate is 97% to 100% sure of the cluster, and the top drivers agree with the profile from Step 8:

| Cluster | Customer | Main drivers (feature value, push towards the cluster) |
|---|---|---|
| 0 Newer | C10816 | `TENURE` of 8 months (+0.63) |
| 1 Steady | C18632 | no `CASH_ADVANCE` (+0.16), `PRC_FULL_PAYMENT` of 0 (+0.11), balance 1,158 (+0.09) |
| 2 Pay in full | C14620 | `PRC_FULL_PAYMENT` of 0.75 (+0.47) |
| 3 Big spenders | C14402 | purchase frequency 0.92 (+0.19), balance 4,709 (+0.16), installments 421 (+0.09) |
| 4 Cash advance | C10124 | purchase frequency 0 (+0.25), cash advance 509 (+0.13), no installments (+0.12) |
| 5 Dormant | C16039 | `BALANCE_FREQUENCY` of 0.36 (+0.35), balance of 13 (+0.25) |

SHAP values here are in probability units, so "+0.47" means that feature alone added 47 percentage points to the probability of that cluster.
""")
