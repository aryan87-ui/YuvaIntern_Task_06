import pandas as pd

df = pd.read_csv("data/Sample - Superstore.csv", encoding="latin1")
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary statistics:\n", df.describe(include="all"))
print("\nSales:", df["Sales"].sum())
print("Profit:", df["Profit"].sum())
print("Quantity:", df["Quantity"].sum())
