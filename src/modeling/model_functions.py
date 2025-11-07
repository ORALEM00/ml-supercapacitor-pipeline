import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.modeling.pipeline_transform import model_pipeline, feature_pipeline
from src.utils.model_utils import _get_model_instance, _create_cv_results_dict, CustomGroupKFold
from src.utils.model_utils import _get_voting_regressor_instance, _validate_cv_inputs


def train_test_analysis(df, model_class, feature_selection = False, params = None, 
                        random_state = 42, target_column = 'Specific_Capacitance', 
                        group_id_column = 'Electrode_ID'):
    # Shuffle df
    X = shuffle(df, random_state = random_state).reset_index(drop = True)
    y = X.pop(target_column)

    # Drop group_id_column if present
    if group_id_column in X.columns: 
        X = X.drop(group_id_column, axis = 1)
    
    # Train/Test split implementation 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)
    
    # For Voting Regressor case (model class pass in list format)
    # If given list format of model classes a Voting Regressor will be implemented
    if isinstance(model_class, list):
        # params will be a list containing dictionary for each model
        model = _get_voting_regressor_instance(model_class, params_list = params, random_state = random_state)
    # Else model instance
    else:
        model = _get_model_instance(model_class, params = params, random_state = random_state)
        
    # Feature selection per train fold
    if feature_selection: 
        pipeline = feature_pipeline(X_train, model)
    else: 
        pipeline = model_pipeline(X_train, model)
            
    pipeline.fit(X_train, y_train)
    y_predict = pipeline.predict(X_test)

    features = pipeline.named_steps['feature_selection'].to_keep_ if feature_selection else None
    
    # Scores
    R2_score = r2_score(y_test, y_predict)
    MAE_score = mean_absolute_error(y_test, y_predict)
    RMSE_score = np.sqrt(mean_squared_error(y_test, y_predict))

    return _create_cv_results_dict(R2_score, MAE_score, RMSE_score, y_predict, y_test, features)



def cv_analysis(df, model_class, cv_splitter, feature_selection = False, params = None, 
                random_state = 42, target_column = 'Specific_Capacitance', 
                group_id_column = 'Electrode_ID'):
    # Copy df
    X = df.copy()
    y = X.pop(target_column)

    # Validate correct input format
    _validate_cv_inputs(model_class, cv_splitter, params)
    
    # Store scores
    R2_score = []
    MAE_score = []
    RMSE_score = []
    
    # Store predictions
    y_predict = pd.Series(np.zeros(len(y)), index=y.index)
    # Store features
    features_per_fold = [] if feature_selection else None

    # Creates groups if CustomGroupKFold else None
    groups = X.get(group_id_column) if (isinstance(cv_splitter, CustomGroupKFold) and group_id_column) else None
    
    # Then drop group_id_column
    if group_id_column in X.columns: 
        X = X.drop(group_id_column, axis = 1)
    
    # Obtain the folds and implement model
    for i, (train_index, test_index) in enumerate(cv_splitter.split(X, y = None, groups = groups)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
        # Initiate model instance (accept both list of parameters per fold and single parameters)
        if params is not None:
            # More than one model (Voting Regressor)
            if isinstance(model_class, list): 
                if np.ndim(params) == 2:
                    current_params = [p[i] for p in params]
                else:
                    # List of dictionaries for each model
                    current_params = params 
            else:
                if (isinstance(params, list)):
                    current_params = params[i]
                else:
                    current_params = params
        else: 
            current_params = None

        # For voting regressor case 
        if isinstance(model_class, list):
            # params will be a list containing dictionary for each model
            model = _get_voting_regressor_instance(model_class, params_list = current_params, random_state = random_state)
        # Else model instance for other models
        else:
            model = _get_model_instance(model_class, params = current_params, random_state = random_state)
        
        # Feature selection per train fold
        if feature_selection: 
            pipeline = feature_pipeline(X_train, model)
        else: 
            pipeline = model_pipeline(X_train, model)
        
        # Model implementation
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        if feature_selection: 
            # Save selected features
            features = pipeline.named_steps['feature_selection'].to_keep_
            features_per_fold.append(features)
        
        # Scores
        R2_score.append(r2_score(y_test, predictions))
        MAE_score.append(mean_absolute_error(y_test, predictions))
        RMSE_score.append(np.sqrt(mean_squared_error(y_test, predictions)))
        
        # Save predictions 
        y_predict.iloc[test_index] = predictions

    return _create_cv_results_dict(R2_score, MAE_score, RMSE_score, y_predict, y, features_per_fold)