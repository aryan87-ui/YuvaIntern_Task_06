# YuvaIntern Task 06 – End-to-End Data Analysis Capstone

## Project Overview

This project is the final capstone of the YuvaIntern Data Analytics internship. It integrates the major stages completed throughout the previous tasks into one reproducible end-to-end data analytics workflow using the **Sample Superstore** dataset.

The project covers:

- Data Exploration
- Data Cleaning and Feature Engineering
- Data Visualization
- Statistical Analysis
- Machine Learning
- Integrated Business Insights
- Recommendations and Future Work

## Dataset

**Dataset:** Sample Superstore

Original dataset dimensions:

- Rows: 9,994
- Columns: 21
- Total Sales: $2,297,200.86
- Total Profit: $286,397.02
- Total Quantity: 37,873

The source CSV is stored in:

```text
data/Sample - Superstore.csv
```

The dataset is loaded using `encoding="latin1"` because the source file contains a non-UTF-8 character.

## Project Structure

```text
YuvaIntern_Task_06/
│
├── data/
│   └── Sample - Superstore.csv
│
├── notebooks/
│
├── src/
│   ├── data_exploration.py
│   ├── data_cleaning.py
│   ├── visualization.py
│   ├── statistical_analysis.py
│   └── machine_learning.py
│
├── visualizations/
│   ├── monthly_sales_trend.png
│   ├── category_sales_profit.png
│   ├── region_sales_profit.png
│   ├── discount_profit.png
│   ├── actual_vs_predicted_sales.png
│   └── model_error_comparison.png
│
├── outputs/
│   ├── Superstore_cleaned.csv
│   ├── model_comparison.csv
│   └── predictions.csv
│
├── report/
│   └── Week_6_End_to_End_Data_Analysis_Report.docx
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 1. Data Exploration

The exploration stage examines:

- Dataset shape and structure
- Data types
- Summary statistics
- Missing values
- Duplicate records
- Sales and profit distributions
- Important business variables

The initial dataset contains 9,994 observations and 21 columns.

## 2. Data Cleaning and Feature Engineering

The cleaning workflow includes:

- Date conversion
- Ship duration calculation
- Profit margin calculation
- Loss flag creation
- Year/month/quarter extraction
- Month-Year feature
- Discount bucket creation
- IQR-based Sales and Profit outlier flags

The enriched dataset contains 32 columns.

Generated cleaned dataset:

```text
outputs/Superstore_cleaned.csv
```

## 3. Data Visualization

The project generates visualizations for:

1. Monthly sales trend
2. Sales and profit by category
3. Sales and profit by region
4. Discount versus profit
5. Actual versus predicted sales
6. Model error comparison

All charts are stored in:

```text
visualizations/
```

## 4. Statistical Analysis

The following statistical tests were performed.

### Discount vs Profit

Pearson correlation:

```text
r = -0.2194874564
p = 2.7022944362e-109
```

This indicates a statistically significant negative association between discount and profit.

### Discounted vs Non-Discounted Transactions

Welch t-test:

```text
t = -15.7379929410
p = 4.3569303711e-55
```

The test indicates a statistically significant difference in mean profit between the two groups.

### Category Profit Differences

One-way ANOVA:

```text
F = 54.3110230438
p = 3.4699183462e-24
```

This indicates statistically significant differences in mean profit across categories.

### Region Profit Differences

One-way ANOVA:

```text
F = 2.6224781547
p = 0.0488916002
```

The result is just below the conventional 0.05 threshold and should therefore be interpreted cautiously.

> Statistical significance indicates association or group differences under the tested assumptions; it does not by itself establish causation.

## 5. Machine Learning

The project predicts **Sales** using:

- Ship Mode
- Segment
- Region
- Category
- Sub-Category
- Quantity
- Discount
- Order Year
- Order Month

Categorical variables are imputed and one-hot encoded. Numerical variables use median imputation.

The dataset is divided into:

- 80% training data
- 20% testing data
- `random_state=42`

### Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 226.3618 | 693.3422 | 0.1862 |
| Random Forest | 215.2316 | 686.9167 | 0.2012 |
| Tuned Random Forest | **196.7443** | **667.6206** | **0.2454** |

### Tuned Random Forest Parameters

```text
n_estimators = 200
max_depth = 10
min_samples_split = 5
min_samples_leaf = 2
```

The tuned Random Forest was optimized using 5-fold GridSearchCV across 24 parameter combinations.

The tuned model achieved the lowest MAE and RMSE and the highest R² among the evaluated models in this experiment.

## 6. Key Business Insights

- Discount has a negative statistical association with profit.
- Profit differs significantly across product categories.
- Regional profit differences are close to the conventional statistical significance threshold.
- Sales and profit should be monitored together rather than relying on sales alone.
- The tuned Random Forest provides the strongest predictive performance among the tested models.
- The R² of 0.2454 also shows that substantial sales variation remains unexplained by the selected features.

## 7. Recommendations

- Monitor discounting policies and their profitability impact.
- Evaluate category-level profitability separately from sales volume.
- Investigate regional performance using additional KPIs.
- Use the tuned Random Forest as a baseline predictive model.
- Add richer customer, time-series and product-level features in future versions.
- Validate unusual transactions before removing them as outliers.

## 8. How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/aryan87-ui/YuvaIntern_Task_06.git
cd YuvaIntern_Task_06
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell execution policy blocks activation, you can use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the scripts

```bash
python src/data_exploration.py
python src/data_cleaning.py
python src/visualization.py
python src/statistical_analysis.py
python src/machine_learning.py
```

The machine learning stage performs 5-fold GridSearchCV, so it may take longer than the other scripts.

## 9. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- SciPy
- Scikit-learn
- Jupyter/VS Code
- Git and GitHub

## 10. Report

The complete capstone report is available at:

```text
report/Week_6_End_to_End_Data_Analysis_Report.docx
```

The report contains:

- Executive Summary
- Introduction
- Data Exploration
- Data Cleaning
- Data Visualization
- Statistical Analysis
- Machine Learning
- Integrated Findings
- Recommendations
- Limitations
- Future Work
- Conclusion
- Reproducibility notes

## Author

**Aryan Verma**

B.Tech Computer Science Engineering (CSE)
Data Analytics Intern

This project was completed as part of the **YuvaIntern Data Analytics Internship – Task 06 (End-to-End Data Analysis Capstone)**.

### Connect with Me

* GitHub: `https://github.com/aryan87-ui`

## Conclusion

This project demonstrates an end-to-end business data analytics workflow, from raw data exploration and cleaning to statistical analysis and predictive modeling. It integrates the skills developed throughout the internship and provides a reproducible foundation for further business intelligence and machine learning work.
