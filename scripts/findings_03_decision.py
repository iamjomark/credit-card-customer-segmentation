text(cells, """
### What the robustness checks tell us

| Change | ARI at k=6 | % of customers who change group | ARI at k=4 | % moved at k=4 |
|---|---|---|---|---|
| Drop `MINIMUM_PAYMENTS` | 0.85 | 6.5% | 0.88 | 5.2% |
| Fill blank minimum payments with 0 (for customers with no payments) | 0.90 | 4.4% | 0.90 | 4.3% |
| Yeo-Johnson instead of log1p, on the **same 7 money columns** | **0.92** | **3.5%** | 0.80 | 8.7% |
| Yeo-Johnson on **all 11 columns** (also reshapes the 3 bounded ones) | 0.67 | 17.4% | 0.75 | 11.4% |

*"% moved" is the share of customers placed in a different group, after matching the new cluster numbers to the old ones as well as possible. ARI is a score of agreement; "% moved" is the easier number to read.*

- **Dropping the feature or filling zeros changes little:** 4% to 7% of customers move. The median-imputation decision from notebook 01 and the 0.90 correlation pair are not driving the result.
- **Swapping log1p for Yeo-Johnson on the same seven columns barely matters** at k=6 (3.5% move), and k=6 is clearly sturdier than k=4 on this test (0.92 against 0.80).
- **What matters is the treatment of the three bounded columns.** When Yeo-Johnson also reshapes `TENURE`, `BALANCE_FREQUENCY` and `PRC_FULL_PAYMENT`, 17.4% of customers move at k=6. Those three columns are the signature features of three of our clusters (newer, dormant, pay in full), so reshaping them changes who sits close to whom. Our first version of this check mixed the two effects together, which made the transform look more important than it is. We chose to leave the bounded columns as they are (they are shares or a near-constant, so a skew correction is the wrong tool), but this is a modelling choice, and a different one gives noticeably different boundaries. We report it as a limitation.

### Decision: k = 6 (runner-up: k = 4)

**The trade-off in plain words.** Fewer groups are easier to explain and to act on. More groups catch finer differences, but each extra group is a worse bet: smaller, less stable and harder to tell apart. We look for the point just before the extra groups stop being reliable.

**Why 6:**
1. **Most stable:** ARI 0.976 with a worst case of 0.961. These six groups come back almost unchanged when the customers are resampled.
2. **Davies-Bouldin is as good as at k=5 or 7** (about 1.50) and better than k=4 (1.60). Silhouette (0.217) is essentially equal to every other k.
3. **Every group is large enough to act on** (the smallest is 6.6% of customers, 595 people).
4. **Business value:** moving from 4 to 6 separates a group of large-limit, high-balance, high-spending customers (about 18% of the base) from the other active customers. That is exactly the kind of group a credit limit review would target, and at k=4 it is blended in.

**Why not k=4 (runner-up):** it is simpler, and it holds up a little better only if all 11 columns are reshaped (0.75 against 0.67). On the like-for-like transform check k=6 is the sturdier one. k=4 is a perfectly reasonable choice for a short executive summary. It just blends together groups whose needs differ.

**An honest caveat:** silhouette near 0.2 means these are *practical divisions of a continuous range of customers*, not natural islands. The segments are useful, and stable under resampling, but a customer near a boundary could easily have been placed in the next group.
""")
