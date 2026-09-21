import pandas as pd
from scipy import stats

df = pd.read_csv("data/Sample - Superstore.csv", encoding="latin1")

# Pearson correlation
r, p = stats.pearsonr(df["Discount"], df["Profit"])
print("Discount-Profit Pearson r:", r)
print("p-value:", p)

# Welch t-test
discounted = df.loc[df["Discount"] > 0, "Profit"]
non_discounted = df.loc[df["Discount"] == 0, "Profit"]
t, p_t = stats.ttest_ind(discounted, non_discounted, equal_var=False)
print("Welch t-test:", t, p_t)

# Category ANOVA
groups = [g["Profit"].values for _, g in df.groupby("Category")]
f_cat, p_cat = stats.f_oneway(*groups)
print("Category ANOVA:", f_cat, p_cat)

# Region ANOVA
groups_r = [g["Profit"].values for _, g in df.groupby("Region")]
f_reg, p_reg = stats.f_oneway(*groups_r)
print("Region ANOVA:", f_reg, p_reg)
