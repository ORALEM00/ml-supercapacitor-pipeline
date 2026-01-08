import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.modeling.pipeline_transform import model_pipeline, feature_pipeline
from src.utils.model_utils import _get_model_instance, _create_cv_results_dict, CustomGroupKFold
from src.utils.model_utils import _get_voting_regressor_instance, _validate_cv_params, _validate_cv_inputs


def train_test_analysis(X, y, model_class, feature_selection = False, params = None, weights = None,
                        random_state = 42):
    
    # Train/Test split implementation with data shuffling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)
    
    # If given list format of model classes a Voting Regressor will be implemented
    if isinstance(model_class, list):
        # params will be a list containing dictionary for each model
        model = _get_voting_regressor_instance(model_class, params_list = params, weights = weights, random_state = random_state)
    # Else single model instance
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



def cv_analysis(X, y, model_class, cv_splitter, groups = None, feature_selection = False, params = None, weights = None,
                random_state = 42):

    # Validate correct input format
    _validate_cv_inputs(model_class, cv_splitter, params)

    # CustomGroupKFold need groups column
    if isinstance(cv_splitter, CustomGroupKFold) and groups is None:
        raise ValueError("Groups column must be provided when using CustomGroupKFold.")
    
    # Store scores
    R2_score = []
    MAE_score = []
    RMSE_score = []
    
    # Store predictions
    y_predict = pd.Series(np.zeros(len(y)), index=y.index)
    # Store features
    features_per_fold = [] if feature_selection else None

    # Obtain the folds and implement model
    for i, (train_index, test_index) in enumerate(cv_splitter.split(X, y = None, groups = groups)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
        # Validate model params and give correct order
        current_params = _validate_cv_params(i,model_class, params)

        # For voting regressor case 
        if isinstance(model_class, list):
            # params will be a list containing dictionary for each model
            model = _get_voting_regressor_instance(model_class, params_list = current_params, 
                                                   weights = weights, random_state = random_state)
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