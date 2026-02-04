import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold 
from sklearn.linear_model import Ridge

from src.leakproof_ml import cv_analysis
from src.leakproof_ml.preprocessing import drop_outliers
from src.leakproof_ml.validation import ShuffledGroupKFold
from src.leakproof_ml.plots import feature_frequency


"""
Feature stability analysis. 
This section is model agnostic, so any model can be used. For this we implement Ridge. 
The stability is calculated for each protocol implemented and for both datasets (with and without outliers)s
"""

# Environment variables
RANDOM_SEED = 42 # For reproducibility
n_splits = 10 # Splits in cross-validation

# Loading database
input_path = "data/processed.csv"
index_cols = "Num_Data" 

df = pd.read_csv(input_path, index_col = index_cols) # Complete dataset

df.rename(columns={
                          "M_Density": "Material Density", 
                          "E_Ionic_Conductivity": "Ionic Conductivity",
                          "E_Bare_Cation_Radius": "Bare Cation Radius",
                          "E_Bare_Anion_Radius": "Bare Anion Radius",
                          "E_Cation_Radius": "Hydrated Cation Radius",
                          "E_Anion_Radius": "Hydrated Anion Radius",
                          "Is_Binder": "Binder Presence",
                          "Binder_Type": "Binder Type",
                          "Morphology_Encoded": "Morphology",
                          "E_pH" : "pH",
                          "Current_Collector": "Current Collector",
                          "CC_Electrical_Conductivity": "CC Electrical Conductivity",
                          "CC_Thermal_Conductivity": "CC Thermal Conductivity",
                          "CC_Work_Function": "CC Work Function",
                          "Potential_Window": "Potential Window",
                          "Synthesis_Method": "Synthesis Method",
                          "Current_Density": "Current Density",
                           }, inplace = True)


df_removed = drop_outliers(df, target_column = "Specific_Capacitance", group_id_colum = "Electrode_ID") # Datset without outliers

# Set X, y, groups with outliers
X = df.drop(columns = ['Specific_Capacitance', 'Electrode_ID'])
y = df['Specific_Capacitance']
groups = df['Electrode_ID']

# set X, y, groups withoud outliers
X_removed = df_removed.drop(columns = ['Specific_Capacitance', 'Electrode_ID'])
y_removed = df_removed['Specific_Capacitance']
groups_removed = df_removed['Electrode_ID']

model = Ridge

# Create CV splitters
random_cv_splitter = KFold(n_splits = n_splits, random_state = RANDOM_SEED, shuffle = True)
grouped_cv_splitter = ShuffledGroupKFold(n_splits = n_splits, random_state = RANDOM_SEED)

# Output path
output_path = "resulting_plots/features"

# Random CV
randomCV = cv_analysis(X, y, model, random_cv_splitter, feature_selection = True)
randomCV_removed = cv_analysis(X_removed, y_removed, model, random_cv_splitter, feature_selection = True) # With Outliers
# Grouped CV
#groupedCV = cv_analysis(df, model, grouped_cv_splitter, feature_selection = True)
groupedCV = cv_analysis(X, y, model, grouped_cv_splitter, groups=groups, feature_selection = True)
groupedCV_removed = cv_analysis(X_removed, y_removed, model, grouped_cv_splitter, groups=groups_removed, feature_selection = True)


# Create feature frequency plots

# Random CV
feature_frequency(X, y, randomCV['features'], filename = os.path.join(output_path, "randomCV_with_outliers.png"))
feature_frequency(X_removed, y_removed, randomCV_removed['features'], filename = os.path.join(output_path, "randomCV_without_outliers.png"))
# Grouped CV
feature_frequency(X, y, groupedCV['features'], filename = os.path.join(output_path, "groupedCV_with_outliers.png"))  
feature_frequency(X_removed, y_removed, groupedCV_removed['features'], filename = os.path.join(output_path, "groupedCV_without_outliers.png"))