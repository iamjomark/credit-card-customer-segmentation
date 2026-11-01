text(cells, """
### What the outliers tell us

- **The IQR rule flags a lot, and that is itself a finding.** Between 3% and 12% of customers are flagged on each money column, because these columns are heavy-tailed. 643 customers (7.2%) are flagged on five or more features at once. Being extreme on many features together points to genuinely different customers, not typos.
- **The bounded columns are flagged for the wrong reason.** `BALANCE_FREQUENCY` (16.7%), `PRC_FULL_PAYMENT` (16.5%) and `TENURE` (15.3%) show many "outliers" only because most customers sit at one value (1.0, 0, 12), which makes the IQR near zero. The IQR rule is not meaningful for these three, so we ignore their flags.
- **The top customers look real.**
  - *Top spenders:* purchases of 35,000 to 49,000 over 100 to 358 transactions (typical purchase about 100 to 490), credit limits of 12,000 to 22,500, and payments close to the purchase total. These are heavy users who pay what they spend.
  - *Top cash advance users:* up to 47,137 in cash over 123 transactions, with payments of 39,000. Large but internally consistent.
  - *Top balances:* about 16,000 to 19,000 on credit limits of 18,000 to 22,000. Plausible.
- **Real data errors exist but are tiny.** Out of 8,950 rows: 8 rows have `CASH_ADVANCE_FREQUENCY` above 1 (a share cannot exceed 100%), 3 have purchases but zero purchase transactions, and 3 have one-off purchases above total purchases. That is about 14 rows (0.2%), too few to matter. 227 customers (2.5%) have a balance above their credit limit and 9 spent more than 5 times their limit over 6 months while repaying a lot; both are plausible when someone repays and spends again.
- **A caution about `MINIMUM_PAYMENTS` above `PAYMENTS`.** This happens for 2,365 customers (26%). It could mean customers who pay less than the minimum due, or it could be a difference in how the two columns are measured (we have no data dictionary). We treat it as behaviour, not error, but we will not build a business claim on it.

### Decision

**Keep every row.** We do not remove outliers: the high-value customers are the segment a bank most wants to find. Instead, the log transform in Step 5 will tame their pull on the distances, and the 14 inconsistent rows are left as they are.
""")
