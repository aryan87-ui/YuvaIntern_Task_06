import pandas as pd

input_path = "data/Sample - Superstore.csv"
output_path = "outputs/Superstore_cleaned.csv"

df = pd.read_csv(input_path, encoding="latin1")
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

df["Ship Duration"] = (df["Ship Date"] - df["Order Date"]).dt.days
df["Profit Margin"] = df["Profit"].div(df["Sales"].replace(0, pd.NA)) * 100
df["Loss Flag"] = (df["Profit"] < 0).astype(int)
df["Order Year"] = df["Order Date"].dt.year
df["Order Month"] = df["Order Date"].dt.month
df["Order Quarter"] = df["Order Date"].dt.quarter
df["Order Month-Year"] = df["Order Date"].dt.to_period("M").astype(str)
df["Discount Bucket"] = pd.cut(
    df["Discount"], bins=[-0.001, 0, 0.2, 0.4, 1],
    labels=["No Discount", "Low", "Medium", "High"], include_lowest=True
)

for col in ["Sales", "Profit"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df[f"{col} Outlier Flag"] = ((df[col] < lower) | (df[col] > upper)).astype(int)

df.to_csv(output_path, index=False)
print("Saved:", output_path)
print("Final shape:", df.shape)
