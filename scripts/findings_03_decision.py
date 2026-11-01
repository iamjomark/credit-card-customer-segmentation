text(cells, """
### What the robustness checks tell us

| Change | Agreement (ARI) at k=6 | at k=4 |
|---|---|---|
| Drop `MINIMUM_PAYMENTS` | 0.85 | 0.88 |
| Fill blank minimum payments with 0 (for customers with no payments) | 0.90 | 0.90 |
| Yeo-Johnson instead of log1p | 0.67 | 0.75 |

- Dropping the feature or filling zeros changes the grouping only modestly. The median-imputation decision from notebook 01 and the 0.90 correlation pair are not driving the result.
- **The transformation matters more.** With Yeo-Johnson, agreement falls to 0.67 at k=6 (0.75 at k=4), so a noticeable share of customers land in different groups. Most of the structure survives, but the exact boundaries depend on how the skew is handled. We report this as a limitation. k=4 is a bit more robust here, but not by enough to change our choice.

### Decision: k = 6 (runner-up: k = 4)

**The trade-off in plain words.** Fewer groups are easier to explain and to act on. More groups catch finer differences, but each extra group is a worse bet: smaller, less stable and harder to tell apart. We look for the point just before the extra groups stop being reliable.

**Why 6:**
1. **Most stable:** ARI 0.976 with a worst case of 0.961. These six groups come back almost unchanged when the customers are resampled.
2. **Davies-Bouldin is as good as at k=5 or 7** (about 1.50) and better than k=4 (1.60). Silhouette (0.217) is essentially equal to every other k.
3. **Every group is large enough to act on** (the smallest is 6.6% of customers, 595 people).
4. **Business value:** moving from 4 to 6 separates a group of large-limit, high-balance, high-spending customers (about 18% of the base) from the other active customers. That is exactly the kind of group a credit limit review would target, and at k=4 it is blended in.

**Why not k=4 (runner-up):** it is simpler and slightly more robust to a change of transform, and it is a perfectly reasonable choice for a short executive summary. It just blends together groups whose needs differ.

**An honest caveat:** silhouette near 0.2 means these are *practical divisions of a continuous range of customers*, not natural islands. The segments are useful, and stable under resampling, but a customer near a boundary could easily have been placed in the next group.
""")
