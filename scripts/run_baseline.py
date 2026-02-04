import pandas as pd
import numpy as np
import os

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.discriminant_analysis import StandardScaler

from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold 

# Models to be compared
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR # Support Vector machine
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor # Random Forest 
from xgboost import XGBRegressor #Extreme Gradient Boosting
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor

from src.leakproof_ml import train_test_analysis, cv_analysis
from src.leakproof_ml.preprocessing import drop_outliers
from src.leakproof_ml.validation import ShuffledGroupKFold
from src.leakproof_ml.preprocessing import CorrelationSelector

from src.leakproof_ml.utils import save_results_as_json


"""
Baseline results of the implemented models (without hypertunning of parameters)
"""

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
""" model_class = [DummyRegressor, LinearRegression, Ridge, Lasso, ElasticNet, SVR, GaussianProcessRegressor, 
               RandomForestRegressor, XGBRegressor, CatBoostRegressor, LGBMRegressor, 
               MLPRegressor, [XGBRegressor, Ridge, CatBoostRegressor]] """
model_class = [Ridge]

# Create CV splitters
random_cv_splitter = KFold(n_splits = n_splits, random_state = RANDOM_SEED, shuffle = True)
grouped_cv_splitter = ShuffledGroupKFold(n_splits = n_splits, random_state = RANDOM_SEED)

# Output path
# output_path = "raw_results"
output_path = "Tests"

# To collect a summary of results
summary_results = []





def polynomial_custom_factory(model, degree=2, interaction_only=False):

    # 1. First Step: Handle missing values and scale
    # We use make_column_selector to be dynamic
    numeric_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipe, make_column_selector(dtype_include='float64')),
        ],
        remainder='passthrough'
    )


    # 2. Construct the full pipeline
    # We add PolynomialFeatures as a middle step
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=degree, interaction_only=interaction_only)),
        ('feature_selection', CorrelationSelector(threshold=0.68)),
        ('model', model)
    ])

    # Ensure output is pandas DataFrame for compatibility
    # pipe.set_output(transform="pandas")

    return pipe



for model in model_class:
    # Check if Voting Regressor
    if isinstance(model, list):
        model_name = "VotingRegressor"
    # Else get model name
    else:
        model_name = model.__name__ # Name of folder to store per model

    # Run analysis of each methodology
    # Simple split
    trainTest = train_test_analysis(X, y, model, feature_selection = True) # With Outliers
    trainTest_removed = train_test_analysis(X_removed, y_removed, model, feature_selection = True) # Without Outliers
    # Random CV
    randomCV = cv_analysis(X, y, model, random_cv_splitter, feature_selection = True)
    randomCV_removed = cv_analysis(X_removed, y_removed, model, random_cv_splitter, feature_selection = True) # With Outliers
    # Grouped CV
    #groupedCV = cv_analysis(df, model, grouped_cv_splitter, feature_selection = True)
    groupedCV = cv_analysis(X, y, model, grouped_cv_splitter, groups=groups, feature_selection = True)
    groupedCV_removed = cv_analysis(X_removed, y_removed, model, grouped_cv_splitter, groups=groups_removed, feature_selection = True)

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
        save_results_as_json(method, output_path, f'{model_name}', "baseline", method_name)

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

print(summary_df)


""" summary_path = "raw_results/summary_baseline.csv"

os.makedirs(os.path.dirname(summary_path), exist_ok = True)
summary_df.to_csv(summary_path)

print(f"Summary table saved to {summary_path}") """
