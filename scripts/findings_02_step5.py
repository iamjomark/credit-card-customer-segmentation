text(cells, """
### What the comparison shows

| Option | Mean abs skew | Largest absolute z-score | Comment |
|---|---|---|---|
| A. Standard only | 4.83 | 32.4 | Shape unchanged. One customer sits 32 standard deviations out, so a handful of giants would steer KMeans. |
| **B. log1p + Standard** | **0.95** | **5.1** | Calmest extremes of all four. The seven log-transformed columns average about 0.5 skew. |
| C. Quantile (normal) | 0.61 | 5.2 | Looks good, but it ranks customers and *discards the size of the gaps*. It also smears the big pile of zeros into an artificial slope. |
| D. Yeo-Johnson | 0.43 | 5.9 | Lowest skew and a close second. Each column gets its own hidden power, which is harder to explain and to repeat by hand. |

**Reading the B column honestly.** The 0.95 average is held up by four columns we deliberately did not log: `TENURE` (-2.9), `BALANCE_FREQUENCY` (-2.0) and `PRC_FULL_PAYMENT` (1.9), which are bounded and have no better fix, and `PURCHASES_FREQUENCY`, which is already fine. `PAYMENTS` is slightly over-corrected (-1.8) and `BALANCE` a little (-0.9).

**The histograms add a warning that the skew number hides.** After `log1p`, `ONEOFF_PURCHASES` and `CASH_ADVANCE` look like a tall spike at the far left plus a bell curve. The spike is customers with exactly zero: 4,302 (48%) for one-off purchases, 4,628 (52%) for cash advance and 3,916 (44%) for installments. No transform removes this, and it should not be removed: "never uses this feature" is real behaviour and probably the main thing that separates the segments. It just means the skew number looks better than the shape really is.

(The sanity silhouette for C is the highest at 0.253. We do not treat that as a reason to choose it: the scores come from different spaces, and C's gain partly comes from squeezing the data into a tidier shape.)

### Decision: Option B (log1p, then StandardScaler)

- It has the **lowest extreme values** (largest |z| of 5.1), which is the property that protects KMeans from a few giants.
- It is **easy to explain and to repeat**: "we took the log of the money columns, then put everything on the same scale". A business reader can follow that, and a new customer can be treated with the same formula.
- It **keeps the zeros as a clear group** instead of smearing them.
- **Runner-up:** Yeo-Johnson (D). In notebook 03 we check that using D instead gives similar clusters.

Below we build B as a proper Pipeline.
""")
