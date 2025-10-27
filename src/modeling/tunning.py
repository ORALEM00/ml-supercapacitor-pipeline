import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.modeling.pipeline_transform import model_pipeline, feature_pipeline
from src.utils.model_utils import _get_model_instance, _create_cv_results_dict, CustomGroupKFold
from src.utils.tunning_utils import run_study, _reconstruct_gpr_params


def train_test_tunning(df, model_class, feature_selection = False, mu = 0, random_state = 42, 
                       target = 'Specific_Capacitance', group_id_column = 'Electrode_ID'):
    # Shuffle df
    X = shuffle(df, random_state = random_state).reset_index(drop = True)
    y = X.pop(target)

    # Drop group_id_column if present
    if group_id_column in X.columns: 
        X = X.drop(group_id_column, axis = 1)
    
    # Train/Test split implementation 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)

    # Concat X_train and y_train to input to study
    innerTrain = pd.concat([X_train, y_train], axis = 1, join='outer')
    # Create Inner split
    inner_splits = 10
    inner_cv_splitter = KFold(n_splits = inner_splits, shuffle = False)

    # Application of hyperparameter tunning
    study_model = run_study(innerTrain, model_class, inner_cv_splitter, mu)
    
    # Obtain best parameters
    best_params = study_model.best_params 

    # For Gaussian process case (parameters of kernel must be treated differently)
    model_name = model_class.__name__
    if model_name == 'GaussianProcessRegressor':
        best_params = _reconstruct_gpr_params(best_params)

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



def nested_cv_tunning(df, model_class, outer_cv_splitter, inner_cv_splitter, 
              feature_selection = False, mu = 0,  random_state = 42, 
              target = "Specific_Capacitance", group_id_column = "Electrode_ID"):
    # Copy df
    X = df.copy()
    y = X.pop(target)
    
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

    # Creates groups if CustomGroupKFold else None
    groups = X.get(group_id_column) if (isinstance(outer_cv_splitter, CustomGroupKFold) and group_id_column) else None

    # Outer CV folds
    for i, (train_index, test_index) in enumerate(outer_cv_splitter.split(X, y = None, groups = groups)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Join X_train and y_train since the functions need to it together
        innerTrain = pd.concat([X_train, y_train], axis = 1, join='outer')
        
        # Apply study and obtain best parameters
        study = run_study(innerTrain, model_class, inner_cv_splitter, mu)
        print("Fold: ", i)

        # Drop id column here since it is used inside innerTrain for grouped case
        if group_id_column in X_train.columns: 
            X_train = X_train.drop(group_id_column, axis = 1)

        # For Gaussian process case (parameters of kernel must be treated differently)
        model_name = model_class.__name__
        if model_name == 'GaussianProcessRegressor':
            params = _reconstruct_gpr_params(study.best_params)
        
        # Gather best parameters
        best_params.append(params)
        
        # Train the model with the best parameters on innerTrain
        model = _get_model_instance(model_class, params=best_params[i], random_state=random_state)
        
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