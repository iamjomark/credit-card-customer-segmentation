text(cells, """
### What the missing values tell us

- 313 customers lack `MINIMUM_PAYMENTS`, and **77% of them (240) made exactly zero payments**. Among customers who do have a minimum payment, none have zero payments. So the blank is not random: it most likely means "no payment was due or made". (We did not look for a pattern in the remaining 73.)
- `CREDIT_LIMIT` is missing for a single customer, with tiny balance and a tenure of 6 months. Nothing special.

### Decision: median imputation, inside a Pipeline

- **Why not drop rows?** We would lose 313 customers who are probably a real group (inactive on payments), and the clusters would not describe them.
- **Why not model-based?** For 3.5% of one column, KNN or iterative imputation adds complexity and run time with little gain, and is harder to explain to a business reader.
- **Why median?** Both columns are heavily skewed, so the mean (pulled up by giants) would be a poor guess. The median (about 312 for `MINIMUM_PAYMENTS`, 3,000 for `CREDIT_LIMIT`) is a typical value.
- **Honest limitation:** the median is not perfect for the 240 zero-payment customers, for whom "0" may be more truthful. We will run a quick check in notebook 03 (fill those with 0 instead) and report whether the clusters change.
""")
