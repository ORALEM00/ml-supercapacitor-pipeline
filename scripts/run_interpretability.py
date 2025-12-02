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

from src.modeling.interpretability import train_test_interpretability, cv_interpretability
from src.visualizations.metric_plotting import feature_frequency
from src.utils.model_utils import drop_outliers, CustomGroupKFold
from src.utils.io_utils import save_results_as_json, load_results_from_json

# Environment variables
RANDOM_SEED = 42 # For reproducibility
n_splits = 10 # Splits in cross-validation

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
""" model_class = [LinearRegression, Ridge, Lasso, ElasticNet, SVR, GaussianProcessRegressor, 
               RandomForestRegressor, XGBRegressor, CatBoostRegressor, LGBMRegressor, 
               MLPRegressor]  """
model_class = [XGBRegressor]

# Create CV splitters
random_cv_splitter = KFold(n_splits = n_splits, random_state = RANDOM_SEED, shuffle = True)
grouped_cv_splitter = CustomGroupKFold(n_splits = n_splits, random_state = RANDOM_SEED)

# Output path
output_path = "raw_interpretability_results"

# Obtain the most select features per methodology
# Since they are selected model agnostically, we can use any model for this
features_dict = {
    'trainTest': [],
    'trainTest_removed': [],
    'randomCV': [],
    'randomCV_removed': [],
    'groupedCV': [],
    'groupedCV_removed': []
}

for key in features_dict.keys():
    # Read results
    results = load_results_from_json(f"raw_results/XGBRegressor/tuned/{key}.json")

    # Since it has only a single list of features
    if 'trainTest' in key:
        frequent_features = results['features']

    # Multiple lists of features for nested CV
    else:
        features = results['features']

        frequent_features = feature_frequency(X, y, features, return_stable = True, threshold = 0.5)

    features_dict[key] = frequent_features

# Run interpretability analysis per model
i = 0

for model in model_class:
    # Check if Voting Regressor
    if isinstance(model, list):
        model_name = f"VotingRegressor_{i}"
        i += 1
    # Else get model name
    else:
        model_name = model.__name__ # Name of folder to store per model

    # Run tunning of each methodology
    params_dict = {}

    for key in ['trainTest', 'trainTest_removed', 'randomCV', 'randomCV_removed', 'groupedCV', 'groupedCV_removed']:
        results = load_results_from_json(f"raw_results/{model_name}/tuned/{key}.json")
        params_dict[key] = results['params']

    ### Permutation Importace

    # Train Test
    pi_trainTest = train_test_interpretability(X, y, model, method="permutation", 
                                               features_to_use = features_dict['trainTest'], params = params_dict["trainTest"])
    pi_trainTest_removed = train_test_interpretability(X_removed, y_removed, model, method="permutation", 
                                               features_to_use = features_dict['trainTest_removed'], params = params_dict["trainTest_removed"])
    # Random CV
    pi_randomCV = cv_interpretability(X, y, model, random_cv_splitter, method="permutation", 
                                     features_to_use = features_dict['randomCV'], params = params_dict["randomCV"])
    pi_randomCV_removed = cv_interpretability(X_removed, y_removed, model, random_cv_splitter, method="permutation", 
                                     features_to_use = features_dict['randomCV_removed'], params = params_dict["randomCV_removed"])
    # Grouped CV
    pi_groupedCV = cv_interpretability(X, y, model, grouped_cv_splitter, method="permutation", groups=groups, 
                                       features_to_use = features_dict['groupedCV'], params = params_dict["groupedCV"])
    pi_groupedCV_removed = cv_interpretability(X_removed, y_removed, model, grouped_cv_splitter, method="permutation", groups=groups_removed, 
                                       features_to_use = features_dict['groupedCV_removed'], params = params_dict["groupedCV_removed"])
    
    ### SHAP values

    # Train Test
    shap_trainTest = train_test_interpretability(X, y, model, method="shap", 
                                               features_to_use = features_dict['trainTest'], params = params_dict["trainTest"])
    shap_trainTest_removed = train_test_interpretability(X_removed, y_removed, model, method="shap", 
                                               features_to_use = features_dict['trainTest_removed'], params = params_dict["trainTest_removed"])
    # Random CV
    shap_randomCV = cv_interpretability(X, y, model, random_cv_splitter, method="shap", 
                                     features_to_use = features_dict['randomCV'], params = params_dict["randomCV"])
    shap_randomCV_removed = cv_interpretability(X_removed, y_removed, model, random_cv_splitter, method="shap", 
                                     features_to_use = features_dict['randomCV_removed'], params = params_dict["randomCV_removed"])
    # Grouped CV
    shap_groupedCV = cv_interpretability(X, y, model, grouped_cv_splitter, method="shap", groups=groups, 
                                       features_to_use = features_dict['groupedCV'], params = params_dict["groupedCV"])
    shap_groupedCV_removed = cv_interpretability(X_removed, y_removed, model, grouped_cv_splitter, method="shap", groups=groups_removed, 
                                       features_to_use = features_dict['groupedCV_removed'], params = params_dict["groupedCV_removed"])
    
    # List format to easy store
    pi_interpretability_methods = [
        ("pi_trainTest", pi_trainTest),
        ("pi_trainTest_removed", pi_trainTest_removed),
        ("pi_randomCV", pi_randomCV),
        ("pi_randomCV_removed", pi_randomCV_removed),
        ("pi_groupedCV", pi_groupedCV),
        ("pi_groupedCV_removed", pi_groupedCV_removed),
    ]

    shap_interpretability_methods = [
        ("shap_trainTest", shap_trainTest),
        ("shap_trainTest_removed", shap_trainTest_removed),
        ("shap_randomCV", shap_randomCV),
        ("shap_randomCV_removed", shap_randomCV_removed),
        ("shap_groupedCV", shap_groupedCV),
        ("shap_groupedCV_removed", shap_groupedCV_removed)
    ]
    # Save results in JSON files
    for method_name, method in pi_interpretability_methods:
        save_results_as_json(method, output_path, f'{model_name}', "pi", method_name)

    for method_name, method in shap_interpretability_methods:
        save_results_as_json(method, output_path, f'{model_name}', "shap", method_name)
