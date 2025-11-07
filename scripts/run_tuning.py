import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold 

# Models to be compared
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR # Support Vector machine
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor # Random Forest 
from xgboost import XGBRegressor #Extreme Gradient Boosting
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

from src.modeling.tunning import train_test_tunning, nested_cv_tunning
from src.utils.model_utils import drop_outliers, CustomGroupKFold
from src.utils.io_utils import save_results_as_json

"""
Baseline results of the implemented models (without hypertunning of parameters)
"""

# Environment variables
RANDOM_SEED = 42 # For reproducibility
outer_n_splits = 10 # Splits in cross-validation
inner_n_splits = 3 # Splits in inner cross-validation (nested CV)

# Loading database
input_path = "data/processed.csv"
index_cols = "Num_Data" 
df = pd.read_csv(input_path, index_col = index_cols) # Complete dataset
df_removed = drop_outliers(df, target_column = "Specific_Capacitance", group_id_colum = "Electrode_ID") # Datset without outliers


# Model's classes to be implemented (not the model itself)
model_class = [Ridge, Lasso, ElasticNet, SVR, # GaussianProcessRegressor, 
               RandomForestRegressor, XGBRegressor, CatBoostRegressor, LGBMRegressor, 
               MLPRegressor]

# Create CV splitters
outer_random_cv_splitter = KFold(n_splits = outer_n_splits, random_state = RANDOM_SEED, shuffle = True)
inner_random_cv_splitter = KFold(n_splits = inner_n_splits, random_state = RANDOM_SEED, shuffle = True)

outer_grouped_cv_splitter = CustomGroupKFold(n_splits = outer_n_splits, random_state = RANDOM_SEED)
inner_grouped_cv_splitter = CustomGroupKFold(n_splits = inner_n_splits, random_state = RANDOM_SEED)

# Output path
output_path = "results"

# To collect a summary of results
summary_results = []

for model in model_class:
    model_name = model.__name__ # Name of folder to store per model

    # Run tunning of each methodology
    # Simple split
    trainTest = train_test_tunning(df, model, feature_selection = True)
    trainTest_removed = train_test_tunning(df_removed, model, feature_selection = True)
    # Random CV
    randomCV = nested_cv_tunning(df, model, outer_random_cv_splitter, inner_random_cv_splitter, feature_selection = True) # With Outliers
    randomCV_removed = nested_cv_tunning(df_removed, model, outer_random_cv_splitter, inner_random_cv_splitter, feature_selection = True) # With Outliers
    # Grouped CV
    groupedCV = nested_cv_tunning(df, model, outer_grouped_cv_splitter, inner_grouped_cv_splitter, feature_selection = True)
    groupedCV_removed = nested_cv_tunning(df_removed, model, outer_grouped_cv_splitter, inner_grouped_cv_splitter, feature_selection = True)

    # List format to easy store
    methods = [
        ("trainTest", trainTest),
        ("trainTest_removed", trainTest_removed),
        ("randomCV", randomCV),
        ("randomCV_removed", randomCV_removed),
        ("groupedCV", groupedCV),
        ("groupedCV_removed", groupedCV_removed)
    ]

    # Save results in JSON files
    for method_name, method in methods:
        try: 
            save_results_as_json(method, output_path, f'{model_name}', "tuned", method_name)
        except TypeError:
            print(method_name)
            print(method)

        # Collect summary of results 
        for metric in ['R2_score', 'MAE_score', 'RMSE_score']:
            mean_val = np.mean(method[metric])
            std_val = np.std(method[metric])
            summary_results.append({
                "Model": model_name,
                "Methodology": method_name,
                "Metric": metric.replace("_score", ""),
                "Mean": round(mean_val, 2),
                "Std": round(std_val, 2)
            })

# Create dataframe for summary
summary_df = pd.DataFrame(summary_results)

summary_path = "results/summary_tuned.csv"

os.makedirs(os.path.dirname(summary_path), exist_ok = True)
summary_df.to_csv(summary_path)

print(f"Summary table saved to {summary_path}")