# Steps 2 to 4 of notebook 01. Exec'd by build_01.py (shares `cells`, `step`, `text`, `maybe`).

text(cells, """
---
## Step 2: Univariate EDA and normality testing

*Univariate* means looking at one column at a time. For each numeric feature we draw three plots:

- **Histogram with KDE:** bars show how many customers fall in each value range. The KDE (kernel density estimate) is a smooth line drawn over the bars.
- **Boxplot:** the box covers the middle 50% of customers, the line inside is the median, and dots beyond the whiskers are unusually large or small values.
- **Q-Q plot** (quantile-quantile): compares our data with a perfect bell curve. If the dots sit on the red line, the data is roughly normal. Curving away means skew or heavy tails.
""")

step(cells,
     "Computes skewness and kurtosis for every numeric feature and sorts them from most skewed to least.",
     "Skewness far from 0 (above +1 is strongly right-skewed) and kurtosis far from 0.",
     "Skewness measures lopsidedness: positive means a long tail to the right. Kurtosis here is *excess* kurtosis (0 for a normal curve): large positive values mean heavy tails, so more extreme values than a bell curve would give.",
     '''
from scipy import stats  # imported here because this is the first cell that needs scipy

features = df.drop(columns="CUST_ID").columns.tolist()  # the 17 numeric features

shape_table = pd.DataFrame({
    "skewness": [stats.skew(df[col].dropna()) for col in features],      # lopsidedness of each column
    "excess_kurtosis": [stats.kurtosis(df[col].dropna()) for col in features],  # tail heaviness (0 = normal)
}, index=features).round(2).sort_values("skewness", ascending=False)
shape_table
''')

step(cells,
     "Draws the histogram+KDE, boxplot and Q-Q plot for every feature, in three batches, and saves the figures in reports/figures/.",
     "Histograms piled up at the left with a long tail, boxplots with many dots on the right, and Q-Q dots that bend away from the red line.",
     "Together they show the same story as the skewness table, but visually. Bar piles against zero show the many-zeros problem from Step 1.",
     '''
import matplotlib.pyplot as plt  # imported here because plotting starts in this cell
import seaborn as sns            # nicer statistical plots on top of matplotlib

FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_univariate(feature_names: list[str], file_name: str) -> None:
    """One row per feature: histogram+KDE, boxplot, Q-Q plot."""
    fig, axes = plt.subplots(len(feature_names), 3, figsize=(14, 2.6 * len(feature_names)))
    for row, name in enumerate(feature_names):
        values = df[name].dropna()
        sns.histplot(values, kde=True, ax=axes[row, 0], color="steelblue")  # bars + smooth KDE line
        axes[row, 0].set_title(f"{name}: histogram + KDE", fontsize=9)
        sns.boxplot(x=values, ax=axes[row, 1], color="lightgray")           # box = middle 50%, dots = outliers
        axes[row, 1].set_title(f"{name}: boxplot", fontsize=9)
        stats.probplot(values, dist="norm", plot=axes[row, 2])              # Q-Q plot against a normal curve
        axes[row, 2].set_title(f"{name}: Q-Q plot", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / file_name, dpi=110)
    plt.show()


# Three batches of features (6, 6, 5) so each figure stays readable
for batch_number, start in enumerate(range(0, len(features), 6), start=1):
    plot_univariate(features[start:start + 6], f"univariate_{batch_number}.png")
''')

text(cells, """
### Normality tests

A *normality test* asks: "could this column plausibly have come from a normal (bell curve) distribution?" Each test gives a **p-value**. If p is below 0.05 (our alpha, the cut-off for "surprising"), we reject normality.

- **Shapiro-Wilk** is a strong test, but it becomes overly strict on large samples: with thousands of rows it flags even tiny, harmless differences from a bell curve. That is why we run it on a random sample of at most 5,000 rows.
- **D'Agostino K-squared** (`scipy.stats.normaltest`) combines skewness and kurtosis into one test. We run it on the full data.
""")

step(cells,
     "Runs both normality tests on every feature and puts the results in one table with a yes/no column.",
     "The `normal_at_0.05` column, and how small the p-values are.",
     "p-values below 0.05 mean the test rejects normality. We call a feature normal only if it passes both tests. A p-value printed as 0.00e+00 just means it is smaller than the computer can store. (SciPy warns that Shapiro p-values are approximate above 5,000 rows, which is one more reason we sample.)",
     '''
rows = []
for name in features:
    values = df[name].dropna()
    sample = values.sample(n=min(5000, len(values)), random_state=42)  # fixed seed so results repeat
    shapiro_p = stats.shapiro(sample).pvalue          # Shapiro-Wilk on the sample
    dagostino_p = stats.normaltest(values).pvalue     # D'Agostino K-squared on all rows
    rows.append({
        "feature": name,
        "shapiro_p (sample)": shapiro_p,
        "dagostino_p (full)": dagostino_p,
        # one-liner: "yes" only if BOTH p-values are above alpha (0.05), otherwise "no"
        "normal_at_0.05": "yes" if min(shapiro_p, dagostino_p) > 0.05 else "no",
    })

normality_table = pd.DataFrame(rows).set_index("feature")
normality_table.style.format({"shapiro_p (sample)": "{:.2e}", "dagostino_p (full)": "{:.2e}"})  # sci notation for tiny p
''')

step(cells,
     "Counts how many features are strongly right-skewed and lists the ones that are not.",
     "The number of features with skewness above 1, and which features are the exceptions.",
     "This confirms or corrects our expectation that credit card amounts are right-skewed, and it tells us which columns need a transform in Step 5.",
     '''
strongly_right_skewed = shape_table[shape_table["skewness"] > 1].index.tolist()
others = shape_table[shape_table["skewness"] <= 1]
print(f"{len(strongly_right_skewed)=} of {len(features)}")
print(f"{strongly_right_skewed=}")
print("Not strongly right-skewed:")
print(others.to_string())
print(f"{(normality_table['normal_at_0.05'] == 'yes').sum()=} features pass both normality tests")
''')

maybe("findings_step2")

text(cells, """
---
## Step 3: Correlation and multicollinearity

*Correlation* measures how strongly two columns move together (from -1 to +1). *Multicollinearity* means several features carry nearly the same information.

We use **Spearman** correlation instead of Pearson. Pearson measures straight-line relationships and is thrown off by extreme values; Spearman compares only the *ranks* (who is first, second, third), so skew and outliers hardly affect it.
""")

step(cells,
     "Computes the Spearman correlation between every pair of features and draws an annotated heatmap.",
     "Dark red blocks (strong positive) and dark blue (strong negative). Look for groups of features that form a block.",
     "Each cell shows how closely two features rank customers the same way. Values near +1 mean one feature adds almost no new information beyond the other.",
     '''
spearman = df[features].corr(method="spearman")  # rank-based correlation, robust to skew

plt.figure(figsize=(15, 12))
sns.heatmap(spearman, annot=True, fmt=".2f", annot_kws={"size": 7}, cmap="coolwarm", vmin=-1, vmax=1, center=0)
plt.title("Spearman correlation between features")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "spearman_heatmap.png", dpi=110)
plt.show()
''')

step(cells,
     "Lists every pair of different features with an absolute Spearman correlation above 0.8.",
     "Which pairs appear and how many times each feature shows up.",
     "These are the redundant pairs. A feature that appears in many pairs is a good candidate to drop or to keep as the single representative of its group.",
     '''
import numpy as np  # needed to keep only the upper triangle so each pair is listed once

upper_triangle = spearman.where(np.triu(np.ones(spearman.shape, dtype=bool), k=1))  # k=1 skips the diagonal
high_pairs = (
    upper_triangle.stack()                                  # turn the matrix into (feature_a, feature_b) -> value
    .rename_axis(["feature_a", "feature_b"])               # name the two index levels
    .reset_index(name="spearman")                           # turn them into columns
    .loc[lambda pairs: pairs["spearman"].abs() > 0.8]       # keep strong pairs only
    .sort_values("spearman", key=abs, ascending=False)
    .round(2)
)
display(high_pairs.reset_index(drop=True))
print("How often each feature appears in a high pair:")
print(pd.concat([high_pairs["feature_a"], high_pairs["feature_b"]]).value_counts().to_string())
''')

maybe("findings_step3")

text(cells, """
---
## Step 4: Missing values and outliers

### Missing values: the options

1. **Drop the rows.** Simple, but we lose customers (and the customers who are missing may be a special group, which would bias the result).
2. **Median imputation.** Fill each blank with the column's median (the middle value). Robust to extreme values, which matters because these columns are heavily skewed. Fast and easy to explain.
3. **Model-based imputation** (for example KNN or iterative imputation). Predicts each blank from the other columns. More flexible, but slower, harder to explain, and not worth it when only 3.5% is missing in one column.

First we check *why* the values are missing, because the reason affects the choice.
""")

step(cells,
     "Looks at the customers whose MINIMUM_PAYMENTS or CREDIT_LIMIT is missing, and compares their other columns with everyone else.",
     "Whether the missing rows share a pattern, for example all having PAYMENTS equal to 0.",
     "If blanks follow a pattern, they are not random. That tells us whether a plain median fill is a fair guess.",
     '''
missing_min_payment = df["MINIMUM_PAYMENTS"].isna()
print(f"{missing_min_payment.sum()=}")
print("PAYMENTS for customers missing MINIMUM_PAYMENTS:")
print(df.loc[missing_min_payment, "PAYMENTS"].describe().round(2).to_string())
print(f"{(df.loc[missing_min_payment, 'PAYMENTS'] == 0).mean()=:.3f}")
print(f"{(df.loc[~missing_min_payment, 'PAYMENTS'] == 0).sum()=}  (zero payments among rows that have a minimum payment)")
print("Row missing CREDIT_LIMIT:")
df[df["CREDIT_LIMIT"].isna()].T
''')

maybe("findings_step4_missing")

step(cells,
     "Builds the imputation as a scikit-learn Pipeline and shows that it fills blanks the same way on brand new data.",
     "Zero missing values after transform, and the learned medians. The new row should get the same median as the training data.",
     "A Pipeline remembers what it learned (here, the medians) when fitted. Calling transform later on new customers fills blanks using the training medians, never the new data's own, which avoids data leakage (information from outside the training data sneaking in). Notebook 02 reuses this inside the full preprocessing pipeline.",
     '''
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

imputation_pipeline = Pipeline([("median_imputer", SimpleImputer(strategy="median"))]).set_output(transform="pandas")
imputed = imputation_pipeline.fit_transform(df[features])  # learn medians, then fill blanks
print(f"{imputed.isna().sum().sum()=}")
print("Learned medians for the two columns that had blanks:")
print(imputation_pipeline.named_steps["median_imputer"].statistics_[[features.index("CREDIT_LIMIT"), features.index("MINIMUM_PAYMENTS")]])

# Simulate a new customer with blanks: the pipeline fills them with the TRAINING medians
new_customer = df[features].iloc[[0]].copy()
new_customer[["CREDIT_LIMIT", "MINIMUM_PAYMENTS"]] = np.nan
imputation_pipeline.transform(new_customer)[["CREDIT_LIMIT", "MINIMUM_PAYMENTS"]]
''')

text(cells, """
### Outliers with the IQR rule

The **IQR rule** (interquartile range) flags a value as an outlier if it is more than 1.5 times the IQR below the 25th percentile (Q1) or above the 75th percentile (Q3). The IQR is Q3 minus Q1, the spread of the middle half of customers. We only *flag* outliers. We do not remove them, because a high-value customer is exactly the kind of person a segmentation should find.
""")

step(cells,
     "Flags outliers for every feature with the IQR rule and counts them.",
     "Which features have the largest share of outliers, and how many features each customer is flagged on.",
     "A feature with a large share of flagged rows has a heavy tail, not a few mistakes. A customer flagged on many features is consistently extreme, which looks like a real big spender or big borrower rather than a typo.",
     '''
def iqr_outlier_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """True where a value is outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR], column by column."""
    q1 = frame.quantile(0.25)
    q3 = frame.quantile(0.75)
    iqr = q3 - q1
    return (frame < q1 - 1.5 * iqr) | (frame > q3 + 1.5 * iqr)  # broadcast comparison, one bound per column


outlier_flags = iqr_outlier_flags(df[features])
outlier_summary = pd.DataFrame({
    "outliers": outlier_flags.sum(),
    "percent": (outlier_flags.mean() * 100).round(1),
}).sort_values("outliers", ascending=False)
display(outlier_summary)

flags_per_customer = outlier_flags.sum(axis=1)  # on how many features is each customer flagged?
print(flags_per_customer.value_counts().sort_index().to_string())
print(f"{(flags_per_customer >= 5).sum()=} customers are flagged on 5 or more features")
''')

step(cells,
     "Shows the top 5 customers for purchases, cash advance and balance, together with simple ratios that would reveal data errors.",
     "Do the numbers make sense together? For example, purchases that match the number of transactions, a high credit limit for big spenders, and payments in a similar range.",
     "Data errors tend to look inconsistent (a huge purchase total with 0 transactions, a balance far above the credit limit for no reason). Real high-value customers look internally consistent.",
     '''
inspection_columns = ["BALANCE", "PURCHASES", "PURCHASES_TRX", "CASH_ADVANCE", "CASH_ADVANCE_TRX", "CREDIT_LIMIT", "PAYMENTS", "TENURE"]
for column in ["PURCHASES", "CASH_ADVANCE", "BALANCE"]:
    top = df.nlargest(5, column)[inspection_columns].copy()
    top["avg_purchase_size"] = (top["PURCHASES"] / top["PURCHASES_TRX"]).round(0)   # money per purchase
    top["purchases_vs_limit"] = (top["PURCHASES"] / top["CREDIT_LIMIT"]).round(1)   # spend as a multiple of the limit
    print(f"Top 5 by {column}")
    display(top.round(0))
''')

step(cells,
     "Runs simple consistency checks that would point to data errors.",
     "Counts that are zero or tiny. Large counts mean a rule in the data we do not understand yet.",
     "Each line is a rule that clean data should follow. Few violations means the extreme values are probably real behaviour, not mistakes.",
     '''
frequency_columns = [column for column in features if column.endswith("_FREQUENCY") or column == "PRC_FULL_PAYMENT"]
checks = {
    "negative values anywhere": int((df[features] < 0).sum().sum()),
    "ratio columns outside 0 to 1": int(((df[frequency_columns] < 0) | (df[frequency_columns] > 1)).sum().sum()),
    "purchases > 0 but 0 purchase transactions": int(((df["PURCHASES"] > 0) & (df["PURCHASES_TRX"] == 0)).sum()),
    "cash advance > 0 but 0 cash advance transactions": int(((df["CASH_ADVANCE"] > 0) & (df["CASH_ADVANCE_TRX"] == 0)).sum()),
    "one-off purchases above total purchases": int((df["ONEOFF_PURCHASES"] > df["PURCHASES"] + 0.01).sum()),
    "balance above credit limit": int((df["BALANCE"] > df["CREDIT_LIMIT"]).sum()),
    "purchases above 5x credit limit": int((df["PURCHASES"] > 5 * df["CREDIT_LIMIT"]).sum()),
    "minimum payment above payments": int((df["MINIMUM_PAYMENTS"] > df["PAYMENTS"]).sum()),
}
pd.Series(checks, name="rows").to_frame()
''')

maybe("findings_step4_outliers")
