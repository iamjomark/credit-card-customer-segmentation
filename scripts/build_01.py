"""Builds notebooks/01_eda.ipynb. Run: uv run python scripts/build_01.py [--findings]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbtools import save, step, text

cells: list = []

text(cells, """
# 01 Exploratory data analysis: credit card customer segmentation

**Business question:** Which groups of credit card customers exist based on how they spend, pay, and borrow, and what does each group look like in business terms?

**Data:** the CC GENERAL dataset. It has about 8,950 rows (one per credit card holder) and 18 columns, with behaviour measured over the last 6 months. We found it as `CC_GENERAL.csv` and copied it to `data/raw/credit_card.csv`.

**How to read this notebook:** every code cell has a short note above it with three lines (what it does, what to look for, what the output means). Imports sit inside the cells that first use them, with a comment saying why.

## Step 1: Early EDA with ydata-profiling

*EDA* means exploratory data analysis: looking at the data before modelling so nothing surprises us later.
""")

step(cells,
     "Finds the project folder and loads the CSV into a table (a pandas DataFrame).",
     "That the shape matches what we expect: about 8,950 rows and 18 columns.",
     "`shape` is (rows, columns). If it were far off, we would have loaded the wrong file.",
     '''
# Imported here so this step only loads the library when it runs
from pathlib import Path
import pandas as pd  # pandas holds and summarises the customer table


def find_project_root(start: Path = Path.cwd()) -> Path:
    """Walk up from the current folder until we see pyproject.toml (nbconvert runs inside notebooks/)."""
    return next(folder for folder in [start, *start.parents] if (folder / "pyproject.toml").exists())


ROOT = find_project_root()
DATA_PATH = ROOT / "data" / "raw" / "credit_card.csv"
REPORTS_DIR = ROOT / "reports"

df = pd.read_csv(DATA_PATH)  # read the CSV into a DataFrame
print(f"{df.shape=}")
df.head()  # first 5 rows, to eyeball the columns
''')

step(cells,
     "Lists each column's data type (dtype) and counts missing values and duplicate rows.",
     "Any column that is not numeric (we expect only CUST_ID), the columns with missing values, and the duplicate count.",
     "Numeric dtypes mean we can compute on them directly. Missing values must be handled before clustering because KMeans cannot accept blanks. Duplicated rows would count one customer twice.",
     '''
print(f"{df.dtypes.value_counts().to_dict()=}")  # how many columns of each type
print(f"{df.duplicated().sum()=}")  # fully identical rows
print(f"{df['CUST_ID'].is_unique=}")  # every customer ID should appear exactly once

# One-liner: count blanks per column, keep only columns that have some, show count and percent.
# isna() marks blanks as True, sum() counts them, and the mask [> 0] drops complete columns.
missing = df.isna().sum()
missing_table = pd.DataFrame({"missing": missing, "percent": (missing / len(df) * 100).round(2)})[missing > 0]
missing_table
''')

step(cells,
     "Prints summary statistics (count, mean, spread, min, quartiles, max) for the numeric columns.",
     "Big gaps between the median (50%) and the max, and columns that are mostly zero.",
     "When the max is far above the 75% value, a few customers have extreme values (a long right tail). That matters later because KMeans is pulled by extremes.",
     '''
numeric_columns = df.drop(columns="CUST_ID").columns.tolist()  # every column except the ID
df[numeric_columns].describe().T.round(2)  # .T flips the table so each feature is one row
''')

step(cells,
     "Builds the full ydata-profiling HTML report and saves it in reports/.",
     "Open reports/credit_card_profile.html in a browser. Look at the Alerts tab, the Correlations tab, and each variable's histogram.",
     "The report automatically flags skew, many zeros, high correlation and missing values. It is a fast first look, not a final answer.",
     '''
def build_profile(frame: pd.DataFrame, output_path: Path):
    """Create the profiling report and save it as HTML."""
    # Imported here so the notebook opens fast and this step only loads the library when it runs
    from ydata_profiling import ProfileReport

    report = ProfileReport(frame, title="Credit card customers: profile", minimal=False, progress_bar=False)
    report.to_file(output_path)  # writes the interactive HTML page
    return report


REPORTS_DIR.mkdir(exist_ok=True)
profile = build_profile(df.drop(columns="CUST_ID"), REPORTS_DIR / "credit_card_profile.html")  # ID is not a feature
print(f"{(REPORTS_DIR / 'credit_card_profile.html').exists()=}")
''')

step(cells,
     "Prints the warnings (alerts) that ydata-profiling raised, so we can quote them in our findings.",
     "Alert types such as SKEWED, ZEROS, HIGH_CORRELATION, MISSING.",
     "Each alert is a data-quality hint. We read them here so our written findings come from real output, not guesses.",
     '''
alerts = profile.get_description().alerts  # list of Alert objects
alert_table = pd.DataFrame({"column": [a.column_name for a in alerts], "alert": [a.alert_type.name for a in alerts]})
print(alert_table.groupby("alert")["column"].apply(lambda cols: ", ".join(sorted(cols))).to_string())
''')

step(cells,
     "Checks three structural facts the report may hint at: whether PURCHASES equals one-off plus installment purchases, how many rows have zero purchases, and how many have MINIMUM_PAYMENTS above PAYMENTS.",
     "The share of rows where the purchase identity holds, and the counts of zero-purchase and odd payment rows.",
     "If PURCHASES is just the sum of two other columns, it adds no new information (redundancy). Customers with zero purchases are a natural group (cash-advance or inactive users).",
     '''
# Does PURCHASES equal ONEOFF + INSTALLMENTS? np.isclose allows tiny rounding differences.
import numpy as np  # imported here because this is the first cell that needs numpy

identity_holds = np.isclose(df["PURCHASES"], df["ONEOFF_PURCHASES"] + df["INSTALLMENTS_PURCHASES"], atol=0.01)
print(f"{identity_holds.mean()=:.4f}")
print(f"{(df['PURCHASES'] == 0).sum()=}")
print(f"{(df['MINIMUM_PAYMENTS'] > df['PAYMENTS']).sum()=}")
print(f"{df['TENURE'].value_counts().sort_index().to_dict()=}")
print(f"{df['CREDIT_LIMIT'].isna().sum()=}, {df['MINIMUM_PAYMENTS'].isna().sum()=}")
''')


def maybe(name: str) -> None:
    """Insert a hand-written interpretation file (written AFTER reading the executed output), if it exists."""
    path = Path(__file__).with_name(f"{name}.py")
    if path.exists():
        exec(path.read_text(), globals())


maybe("findings_01")
exec(Path(__file__).with_name("part2_steps_2_to_4.py").read_text())

save(cells, Path(__file__).parent.parent / "notebooks" / "01_eda.ipynb")
print("built")
