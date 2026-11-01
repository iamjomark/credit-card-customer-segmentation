# Credit card customer segments: business summary

## Summary

We grouped 8,950 credit card holders into six segments, using how they spend, pay and borrow over six months. The groups are stable: they come back almost unchanged when we resample customers. A separate model can also recover each customer's segment from their numbers 95% of the time. The boundaries between neighbouring segments are soft, because customers form a continuous range and not six islands.

| Segment | Share | What they do | What they likely need |
|---|---|---|---|
| Cash advance borrowers | 27% | Almost no purchases, a carried balance and the largest cash advances | A risk and exposure review. Cash advances usually cost more, so test an offer that moves this borrowing to a cheaper product. |
| Steady everyday spenders | 23% | Regular, moderate purchases on a small limit, a carried balance, no cash advances | Usage nudges, payment reminders, and a modest limit increase for those who pay on time. |
| Big spenders, high limit, carry a balance | 18% | Highest spending, balance and limit (about 7,000), and they borrow too | A credit limit review, retention offers and close monitoring. They carry the largest exposure. |
| Frequent buyers, pay in full | 14% | Buy almost every month and score high on the full-payment measure (median 0.80, versus 0.00 for most other segments) | Rewards and retention. Skip collections-style reminders. Consider higher limits. |
| Dormant, barely used card | 11% | Near-zero balance and rarely any activity | A reactivation campaign, or a decision on whether to keep the account open. |
| Newer customers, small limit, mixed use | 7% | About 7 months as customers, the smallest limits, a mix of small purchases and cash advances | Onboarding support. Re-check them after a year, because their behaviour is still forming. |

**What this analysis cannot tell us.** The data has no revenue, profit, late payment or default information, so we cannot say which segment earns the most or is the riskiest. Where the table says "exposure" or "likely need", that is a hypothesis from behaviour, not a measured result. It is a six-month snapshot, so it shows what customers did, not why, and not how they will change. The segments are associations, not causes. Any action should first be tested on a small group with a control group.

## Limitations

- **Sample size and source.** 8,950 customers from one unnamed card portfolio over six months. We do not know the bank, country, year or how customers were chosen, so the results may not carry over to another portfolio.
- **Missing data.** `MINIMUM_PAYMENTS` was missing for 313 customers (3.5%) and `CREDIT_LIMIT` for one. For 240 of the 313, payments were exactly zero, so the blank probably means "nothing due". We filled with the median. Filling those with 0 instead changed the groups only a little (agreement 0.90).
- **No ground truth.** There are no true customer types to compare with, so we can only use internal measures. Silhouette is about 0.22, which is low: the groups touch.
- **How stable the clusters are.** Resampling customers gave an agreement score of 0.976 for six segments (1 is identical). Our own choices also matter a little: dropping `MINIMUM_PAYMENTS` moves 6.5% of customers to a different segment, and swapping the log transform for another standard one (Yeo-Johnson) moves 3.5%. The choice that matters most is how the three bounded columns (tenure, balance frequency, share paid in full) are treated: reshaping them too moves 17.4% of customers.
- **A small number of odd rows.** About 14 rows (0.2%) break simple rules (for example purchases but zero purchase transactions). We kept them.
- **No data dictionary.** In 26% of customers the minimum payment is above the payments. We could not tell whether this is under-payment or a difference in how the columns are measured.
- **One segment is mostly about tenure.** The newer customers are separated mainly by having a short history, not only by how they use the card.

## Next steps: what data would make the model stronger

1. **Outcomes:** late payments, defaults and write-offs, to check which segments are actually risky.
2. **Revenue:** interest, fees and interchange income per customer, so segments can be ranked by profit.
3. **Time:** monthly figures instead of six-month totals, so we can see trends and segment migration (who moves from steady to dormant).
4. **Customer details:** age, income, region and product type, to describe segments in terms the business already uses.
5. **Campaign history:** which offers each customer received and how they responded, to test the recommendations above.
6. **A data dictionary** for the columns, especially the payment fields.
