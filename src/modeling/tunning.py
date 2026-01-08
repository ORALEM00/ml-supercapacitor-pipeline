import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.modeling.pipeline_transform import model_pipeline, feature_pipeline
from src.utils.model_utils import _get_model_instance, _create_cv_results_dict, CustomGroupKFold, _get_voting_regressor_instance
from src.utils.tunning_utils import run_study



def train_test_tunning(X, y, model_class, space_search, feature_selection = False, mu = 0, 
                       random_state = 42, params = None, n_trials = 50, inner_splits = 10):
    # Train/Test split implementation with data shuffling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)

    # Create Inner split
    inner_cv_splitter = KFold(n_splits = inner_splits, shuffle = False)

    # Application of hyperparameter tunning
    study_model = run_study(X_train, y_train, model_class, inner_cv_splitter, space_search, mu = mu, 
                            params=params, random_state=random_state, n_trials=n_trials)
    
    # Obtain best parameters
    best_params = study_model.best_params 

    # For voting regressor case 
    if isinstance(model_class, list):
        # Convert weights to list in correct model order
        best_params = [float(best_params[f'weight_{i+1}']) for i in range(len(model_class))]
        # params will be a list containing dictionary for each model
        model = _get_voting_regressor_instance(model_class, params_list = params, 
                                            weights = best_params, random_state = random_state)
    else:
        # Initiate model instance with new parameters
        model = _get_model_instance(model_class, params = best_params, random_state = random_state)

    # Feature selection per train fold
    if feature_selection: 
        pipeline = feature_pipeline(X_train, model)
    else: 
        pipeline = model_pipeline(X_train, model)
            
    pipeline.fit(X_train, y_train)
    y_predict = pipeline.predict(X_test)

    if feature_selection: 
        # Save selected features
        features = pipeline.named_steps['feature_selection'].to_keep_
    
    # Scores
    R2_score = r2_score(y_test, y_predict)
    MAE_score = mean_absolute_error(y_test, y_predict)
    RMSE_score = np.sqrt(mean_squared_error(y_test, y_predict))

    return _create_cv_results_dict(R2_score, MAE_score, RMSE_score, y_predict, y_test, features, best_params)





def nested_cv_tunning(X, y, model_class, outer_cv_splitter, inner_cv_splitter, space_search, groups = None,
                      feature_selection = False, mu = 0,  random_state = 42, params = None,
                      n_trials = 50):
    # Store scores
    R2_score = []
    MAE_score = []
    RMSE_score = []
    
    # Store predictions
    y_predict = pd.Series(np.zeros(len(y)), index=y.index)
    # Store features
    features_per_fold = [] # if feature_selection else None
    # Store new hyperparameters per fold 
    best_params = []

    # Outer CV folds
    for i, (train_index, test_index) in enumerate(outer_cv_splitter.split(X, y = None, groups = groups)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if groups is not None:
            groups_train, groups_test = groups.iloc[train_index], groups.iloc[test_index]
        else:
            groups_train, groups_test = None, None

        # To keep the params used in each fold
        params_fold = []
        if params is not None and isinstance(params, list):
            for model_param in params:
                params_fold.append(model_param[i])
        else:
            params_fold = params
        
        # Apply study and obtain best parameters
        study_model = run_study(X_train, y_train, model_class, inner_cv_splitter, space_search, groups=groups_train, mu = mu, 
                                params=params_fold, random_state=random_state, n_trials=n_trials)
        print("Fold: ", i)

        # Obtain best parameters
        current_params = study_model.best_params 
        # Gather best parameters
        best_params.append(current_params)

        # For voting regressor case 
        if isinstance(model_class, list):
            # Convert weights to list in correct model order
            current_params = [float(current_params[f'weight_{i+1}']) for i in range(len(model_class))]
            # params will be a list containing dictionary for each model
            model = _get_voting_regressor_instance(model_class, params_list = params_fold, 
                                                weights = current_params, random_state = random_state)
        else:
            # Initiate model instance with new parameters
            model = _get_model_instance(model_class, params = current_params, random_state = random_state)
        
        if feature_selection: 
            pipeline = feature_pipeline(X_train, model)
        else: 
            pipeline = model_pipeline(X_train, model)
            
        pipeline.fit(X_train, y_train)
        # Predictions
        predictions = pipeline.predict(X_test)
        
        # Append features
        if feature_selection: 
            features = pipeline.named_steps['feature_selection'].to_keep_
            features_per_fold.append(features)
        
        # Append Scores
        R2_score.append(r2_score(y_test, predictions))
        MAE_score.append(mean_absolute_error(y_test, predictions))
        RMSE_score.append(np.sqrt(mean_squared_error(y_test, predictions)))
        
        # Save predictions 
        y_predict.iloc[test_index] = predictions
        
    return _create_cv_results_dict(R2_score, MAE_score, RMSE_score, y_predict, y, features_per_fold, best_params)