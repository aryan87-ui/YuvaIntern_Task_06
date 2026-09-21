import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/Sample - Superstore.csv", encoding="latin1")
df["Order Date"] = pd.to_datetime(df["Order Date"])
Path("visualizations").mkdir(exist_ok=True)

monthly = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum()
monthly.index = monthly.index.astype(str)
monthly.plot(figsize=(10,5), title="Monthly Sales Trend")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/monthly_sales_trend.png", dpi=200)
plt.close()

cat = df.groupby("Category")[["Sales","Profit"]].sum()
cat.plot(kind="bar", figsize=(9,5), title="Sales and Profit by Category")
plt.tight_layout()
plt.savefig("visualizations/category_sales_profit.png", dpi=200)
plt.close()

region = df.groupby("Region")[["Sales","Profit"]].sum()
region.plot(kind="bar", figsize=(9,5), title="Sales and Profit by Region")
plt.tight_layout()
plt.savefig("visualizations/region_sales_profit.png", dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.scatter(df["Discount"], df["Profit"], alpha=0.25)
plt.title("Discount vs Profit")
plt.xlabel("Discount")
plt.ylabel("Profit")
plt.tight_layout()
plt.savefig("visualizations/discount_profit.png", dpi=200)
plt.close()
