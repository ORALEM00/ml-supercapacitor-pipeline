import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold 

from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor #Extreme Gradient Boosting
from catboost import CatBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from src.modeling.model_functions import train_test_analysis
from src.utils.model_utils import drop_outliers, CustomGroupKFold
from src.utils.io_utils import save_results_as_json

# Environment variables
RANDOM_SEED = 42 # For reproducibility
outer_n_splits = 10 # Splits in cross-validation
inner_n_splits = 3 # Splits in inner cross-validation (nested CV)

# Loading database
input_path = "data/processed.csv"
index_cols = "Num_Data" 
df = pd.read_csv(input_path, index_col = index_cols) # Complete dataset
df_removed = drop_outliers(df) # Datset without outliers

model_1 = XGBRegressor(random_state = RANDOM_SEED)
model_2 =  GaussianProcessRegressor()
model_3 = CatBoostRegressor(verbose=False, random_state = RANDOM_SEED)

voter = VotingRegressor([('xgb', model_1), ('gp', model_2), ('cat', model_3)])

# Create CV splitters
outer_random_cv_splitter = KFold(n_splits = outer_n_splits, random_state = RANDOM_SEED, shuffle = True)
inner_random_cv_splitter = KFold(n_splits = inner_n_splits, random_state = RANDOM_SEED, shuffle = True)

outer_grouped_cv_splitter = CustomGroupKFold(n_splits = outer_n_splits, random_state = RANDOM_SEED)
inner_grouped_cv_splitter = CustomGroupKFold(n_splits = inner_n_splits, random_state = RANDOM_SEED)

# Output path
output_path = "results"

# To collect a summary of results
summary_results = []

trainTest = train_test_analysis(df, voter, feature_selection = True) # With Outliers