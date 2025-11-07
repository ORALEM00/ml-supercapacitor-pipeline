import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold 

from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor #Extreme Gradient Boosting
from catboost import CatBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,  WhiteKernel
from sklearn.linear_model import Ridge

from src.modeling.model_functions import train_test_analysis, cv_analysis
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
df_removed = drop_outliers(df, target_column = "Specific_Capacitance", group_id_colum = "Electrode_ID")

# Create CV splitters
random_cv_splitter = KFold(n_splits = 10, random_state = RANDOM_SEED, shuffle = True)
grouped_cv_splitter = CustomGroupKFold(n_splits = 10, random_state = RANDOM_SEED)

# Output path
output_path = "results"

# To collect a summary of results
summary_results = []

model_classess1 = [XGBRegressor, GaussianProcessRegressor, CatBoostRegressor]
model_classess2 = [XGBRegressor, Ridge, CatBoostRegressor]

