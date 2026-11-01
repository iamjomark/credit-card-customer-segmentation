text(cells, """
### Five findings from the profile

- **The table is clean in structure.** There are 8,950 rows, no duplicate rows, and every `CUST_ID` is unique. *Why it matters:* one row is one customer, so no de-duplication is needed and cluster sizes can be read directly as shares of customers.
- **Missing values are few and limited to two columns.** `MINIMUM_PAYMENTS` is missing for 313 customers (3.50%) and `CREDIT_LIMIT` for just 1. *Why it matters:* KMeans cannot accept blanks, but this is small enough to fill in with a simple rule inside a pipeline (Step 4) instead of throwing customers away.
- **Money columns are extremely right-skewed.** Skewness (how lopsided a distribution is; 0 is symmetric) is 13.6 for `MINIMUM_PAYMENTS`, 10.0 for `ONEOFF_PURCHASES` and 8.1 for `PURCHASES`. The top `PURCHASES` value is about 49,000 while 99% of customers are below about 9,000. *Why it matters:* KMeans measures straight-line distance, so a few huge values would drag the centres and hide the smaller groups. We will transform the skewed columns (Step 5).
- **Many columns are mostly zeros.** 2,044 customers (about 23%) have made no purchases at all, and the cash advance columns are about 52% zeros. *Why it matters:* these zeros are real behaviour, not errors. They hint at natural groups (for example customers who only take cash advances), and they mean we should use `log1p` (log of 1 + x), which is safe at zero, rather than a plain log.
- **Several features say nearly the same thing.** `PURCHASES` equals `ONEOFF_PURCHASES` plus `INSTALLMENTS_PURCHASES` in 99.8% of rows, and pairs such as `CASH_ADVANCE_FREQUENCY` and `CASH_ADVANCE_TRX` move together almost perfectly (Spearman correlation 0.98). *Why it matters:* duplicated information counts twice in a distance calculation, so we will decide which features to drop in Step 3.

**Other things noticed:** 2,365 customers have a minimum payment larger than their payments (we will look at this in Step 4), and 7,584 of 8,950 customers (85%) have `TENURE` of 12 months, so tenure carries little information to separate groups.
""")
