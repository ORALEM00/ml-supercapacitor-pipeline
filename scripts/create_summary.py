import os
import numpy as np
import pandas as pd

from src.leakproof_ml.utils import load_results_from_json
from src.leakproof_ml.plots import plot_metric_scores

input_path = "results"

models_name = ['Ridge', 'Lasso', 'ElasticNet', 'SVR', 
               'RandomForestRegressor', 'XGBRegressor', 
               'CatBoostRegressor', 'LGBMRegressor', 'MLPRegressor']


methods = ["trainTest","trainTest_removed","randomCV","randomCV_removed","groupedCV", 
           "groupedCV_removed"]

metrics = ['R2_score', 'MAE_score', 'RMSE_score']

summary_results = []

# Gather metrics for all models
r2_per_model = {
    'train_R2' : [],
    'train_removed_R2' : [],
    'random_R2' : [],
    'random_removed_R2' : [],
    'grouped_R2' : [],
    'grouped_removed_R2' : []
}

# Dataframe summary
for model in models_name:
    # Results per model
    for method in methods: 
        # Load results per methdology
        results = load_results_from_json(f"raw_results/{model}/tuned/{method}.json")
        # Store metrics in dataframe for summary
        for metric in metrics:
            mean_val = np.mean(results[metric])
            std_val = np.std(results[metric])
            summary_results.append({
                "Model": model,
                "Methodology": method,
                "Metric": metric.replace("_score", ""),
                "Mean": round(mean_val, 2),
                "Std": round(std_val, 2)
            })

# Create dataframe for summary
summary_df = pd.DataFrame(summary_results)

# Transpose the table to have it in a wide format
pivot_tables_mean = {}
pivot_tables_std= {}

for metric in summary_df["Metric"].unique():
    # Mean values
    pivot_mean = summary_df[summary_df["Metric"] == metric].pivot(
        index="Methodology",   # rows = methods
        columns="Model",       # columns = models
        values="Mean"         # values = mean metric score
    ).reset_index()

    # Std values
    pivot_std = summary_df[summary_df["Metric"] == metric].pivot(
        index="Methodology",   # rows = methods
        columns="Model",       # columns = models
        values="Std"         # values = mean metric score
    ).reset_index()

    pivot_tables_mean[metric] = pivot_mean
    pivot_tables_std[metric] = pivot_std
    

metric_mean = pivot_tables_mean["R2"]
metric_std = pivot_tables_std["R2"]

mean_with = metric_mean.loc[metric_mean["Methodology"] == "groupedCV"]
mean_with = mean_with.drop(columns="Methodology").values.flatten()
mean_without = metric_mean.loc[metric_mean["Methodology"] == "groupedCV_removed"]
mean_without = mean_without.drop(columns="Methodology").values.flatten()

std_with = metric_std.loc[metric_std["Methodology"] == "groupedCV"]
std_with = std_with.drop(columns="Methodology").values.flatten()
std_without = metric_std.loc[metric_std["Methodology"] == "groupedCV_removed"]
std_without = std_without.drop(columns="Methodology").values.flatten()


# Create dictionary for ploting purposes
r2_data = {
    'labels': metric_mean.columns.values[1:], 
    'metric': "MAE",
    'with_outliers_means': mean_with,
    'with_outliers_stds': std_with,
    'without_outliers_means': mean_without, 
    'without_outliers_stds': std_without
}


plot_metric_scores(r2_data, 'Comparison of MAE Scores of Grouped CV', 'mae_grouped_cv.png')

summary_path = "raw_results/summary_tuned.csv"

os.makedirs(os.path.dirname(summary_path), exist_ok = True)
summary_df.to_csv(summary_path)

print(f"Summary table saved to {summary_path}")