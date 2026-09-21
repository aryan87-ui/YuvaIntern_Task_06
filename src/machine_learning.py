import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/Sample - Superstore.csv"
OUTPUT_DIR = Path("outputs")
VIZ_DIR = Path("visualizations")

OUTPUT_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH, encoding="latin1")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    errors="coerce"
)

df["Order_Year"] = df["Order Date"].dt.year
df["Order_Month"] = df["Order Date"].dt.month


# Features used for prediction
features = [
    "Ship Mode",
    "Segment",
    "Region",
    "Category",
    "Sub-Category",
    "Quantity",
    "Discount",
    "Order_Year",
    "Order_Month"
]

target = "Sales"

X = df[features]
y = df[target]


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. PREPROCESSING
# ============================================================

categorical_features = [
    "Ship Mode",
    "Segment",
    "Region",
    "Category",
    "Sub-Category"
]

numeric_features = [
    "Quantity",
    "Discount",
    "Order_Year",
    "Order_Month"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            numeric_features
        ),

        (
            "cat",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),

                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 6. BASE MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
}


results = []
predictions = {}


# ============================================================
# 7. TRAIN BASE MODELS
# ============================================================

print("\n==============================")
print("BASE MODEL TRAINING")
print("==============================")

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)

    predictions[name] = pred

    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    r2 = r2_score(
        y_test,
        pred
    )

    results.append(
        {
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
    )

    print(f"\n{name}")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)


# ============================================================
# 8. TUNED RANDOM FOREST
# ============================================================

print("\n==============================")
print("TUNED RANDOM FOREST")
print("==============================")

rf_pipeline = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# Hyperparameter grid
param_grid = {

    "model__n_estimators":
        [100, 200],

    "model__max_depth":
        [None, 10, 20],

    "model__min_samples_split":
        [2, 5],

    "model__min_samples_leaf":
        [1, 2]
}


# Grid Search
grid_search = GridSearchCV(

    estimator=rf_pipeline,

    param_grid=param_grid,

    cv=5,

    scoring="neg_mean_absolute_error",

    n_jobs=-1,

    verbose=1
)


grid_search.fit(
    X_train,
    y_train
)


# Best model
best_model = grid_search.best_estimator_

best_pred = best_model.predict(
    X_test
)

predictions["Tuned Random Forest"] = best_pred


# Metrics
best_mae = mean_absolute_error(
    y_test,
    best_pred
)

best_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        best_pred
    )
)

best_r2 = r2_score(
    y_test,
    best_pred
)


results.append(
    {
        "Model": "Tuned Random Forest",
        "MAE": best_mae,
        "RMSE": best_rmse,
        "R2": best_r2
    }
)


# ============================================================
# 9. PRINT TUNED MODEL RESULTS
# ============================================================

print("\n--------------------------------")
print("Tuned Random Forest Results")
print("--------------------------------")

print("MAE :", best_mae)
print("RMSE:", best_rmse)
print("R2  :", best_r2)

print("\nBest Parameters:")
print(grid_search.best_params_)


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="RMSE"
)


print("\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================")

print(
    results_df.to_string(
        index=False
    )
)


# Save comparison
results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)


# ============================================================
# 11. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "Actual_Sales":
        y_test.to_numpy(),

    "Linear_Regression":
        predictions["Linear Regression"],

    "Random_Forest":
        predictions["Random Forest"],

    "Tuned_Random_Forest":
        predictions["Tuned Random Forest"]
})


prediction_df.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)


# ============================================================
# 12. ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_test,
    best_pred,
    alpha=0.35
)

plt.xlabel(
    "Actual Sales"
)

plt.ylabel(
    "Predicted Sales"
)

plt.title(
    "Actual vs Predicted Sales - Tuned Random Forest"
)

plt.tight_layout()

plt.savefig(
    VIZ_DIR /
    "actual_vs_predicted_sales.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. MODEL ERROR COMPARISON
# ============================================================

plt.figure(
    figsize=(9, 5)
)

x = np.arange(
    len(results_df)
)

width = 0.35

plt.bar(
    x - width / 2,
    results_df["MAE"],
    width,
    label="MAE"
)

plt.bar(
    x + width / 2,
    results_df["RMSE"],
    width,
    label="RMSE"
)

plt.xticks(
    x,
    results_df["Model"],
    rotation=20
)

plt.ylabel(
    "Error"
)

plt.title(
    "Model Error Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    VIZ_DIR /
    "model_error_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. FINAL OUTPUT
# ============================================================

print("\n==============================")
print("FILES CREATED SUCCESSFULLY")
print("==============================")

print(
    "outputs/model_comparison.csv"
)

print(
    "outputs/predictions.csv"
)

print(
    "visualizations/actual_vs_predicted_sales.png"
)

print(
    "visualizations/model_error_comparison.png"
)

print("\nMachine Learning analysis completed successfully!")