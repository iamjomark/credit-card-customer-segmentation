text(cells, """
### What PCA tells us

- **The data is not simple.** The first two components hold only 51.4% of the variation. It takes 6 components to pass 80% (85.2%) and 7 to pass 90%. This is why we must not cluster on the 2D picture: it throws away about half of the information. KMeans uses all 11 scaled features.
- **What the first two axes mean** (from the loadings): PC1 (28%) rises with `BALANCE` (0.54), `CREDIT_LIMIT` (0.46), `INSTALLMENTS_PURCHASES` and `PAYMENTS`, so it reads as **account size and activity**. PC2 (23%) is led by `MINIMUM_PAYMENTS` (0.54) and `ONEOFF_PURCHASES` (0.49), so it mostly separates **big one-off buyers from those who do not buy**.
- **There are no clean islands.** The cloud is one continuous mass, so clusters will touch each other. Expect modest scores from the metrics in Step 7 (roughly 0.2 to 0.3 for silhouette, not 0.7) and treat the segments as regions of a continuum.
- **One structure does stand out.** The dense streak at the lower right is 1,297 customers (14%). All of them use cash advances, 94% make no one-off purchases, and the median customer in it has zero purchases of any kind. That is a **cash-advance-only** group, and it is the first segment we can already name.
""")
