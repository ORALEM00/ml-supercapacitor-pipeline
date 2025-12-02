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

# From own script
from scripts.search_space import params_space_search

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

# Set X, y, groups with outliers
X = df.drop(columns = ['Specific_Capacitance', 'Electrode_ID'])
y = df['Specific_Capacitance']
groups = df['Electrode_ID']

# set X, y, groups withoud outliers
X_removed = df_removed.drop(columns = ['Specific_Capacitance', 'Electrode_ID'])
y_removed = df_removed['Specific_Capacitance']
groups_removed = df_removed['Electrode_ID']

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
output_path = "raw_results"

# To collect a summary of results
summary_results = []

for model in model_class:
    # Check if Voting Regressor
    if isinstance(model, list):
        model_name = "VotingRegressor"
    # Else get model name
    else:
        model_name = model.__name__ # Name of folder to store per model

    # Defined paramter search space
    model_search_function = params_space_search(model_name)

    # Run tunning of each methodology
    # Simple split
    trainTest = train_test_tunning(X, y, model, model_search_function, feature_selection = True) # With Outliers
    trainTest_removed = train_test_tunning(X_removed, y_removed, model, model_search_function, feature_selection = True) # Without Outliers
    # Random CV
    randomCV = nested_cv_tunning(X, y, model, outer_random_cv_splitter, inner_random_cv_splitter, 
                                    model_search_function, feature_selection = True) # With Outliers
    randomCV_removed = nested_cv_tunning(X_removed, y_removed, model, outer_random_cv_splitter, inner_random_cv_splitter, 
                                    model_search_function, feature_selection = True) # Without Outliers
    # Grouped CV
    groupedCV = nested_cv_tunning(X, y, model, outer_grouped_cv_splitter, inner_grouped_cv_splitter, 
                                  model_search_function, groups=groups, feature_selection = True) # With Outliers
    groupedCV_removed = nested_cv_tunning(X_removed, y_removed, model, outer_grouped_cv_splitter, inner_grouped_cv_splitter, 
                                          model_search_function, groups=groups_removed ,feature_selection = True) # Without Outliers

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

summary_path = "raw_results/summary_tuned.csv"

os.makedirs(os.path.dirname(summary_path), exist_ok = True)
summary_df.to_csv(summary_path)

print(f"Summary table saved to {summary_path}")

print(randomCV)
print(np.mean(randomCV['R2_score']))
print(np.std(randomCV['R2_score']))

print(summary_df)