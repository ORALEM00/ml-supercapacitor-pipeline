import shap
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.modeling.pipeline_transform import model_pipeline
from src.utils.interpretability_utils import _get_final_feature_names
from src.utils.model_utils import _get_model_instance, _get_voting_regressor_instance
from src.utils.model_utils import CustomGroupKFold, _validate_cv_params


def train_test_interpretability(X, y, model_class, method = "permutation", features_to_use = None, 
                                params = None,  weights = None, random_state = 42):
    if features_to_use:
        # To avoid changing the original
        features = list(features_to_use)
        X = X[features]

    # Final feature names after preprocessing
    # In case of voting regressor
    if isinstance(model_class, list):
        # Any model can be used to get feature order
        feature_names = _get_final_feature_names(X, y, model_class[0], random_state=random_state)
    # For single model
    else:
        feature_names = _get_final_feature_names(X, y, model_class, random_state=random_state)

    # Create dictionary to return results
    results_dict = {'method': method}
        
    # Train/Test split implementation 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)
    
    # Initiate model instance
    if isinstance(model_class, list):
        # params will be a list containing dictionary for each model
        model = _get_voting_regressor_instance(model_class, params_list = params, weights = weights, random_state = random_state)
    # Else model instance
    else:
        model = _get_model_instance(model_class, params = params, random_state = random_state)
    
    # Model pipeline (No Feature Selection)
    pipeline = model_pipeline(X_train, model)

    # Fit pipeline to train 
    pipeline.fit(X_train, y_train)

    if (method == "permutation"):
        perm = permutation_importance(pipeline, X_test, y_test, n_repeats=30, 
                                      random_state=random_state, n_jobs=-1)
        
        importance_df = pd.DataFrame({
            'features': X.columns,
            'importance_mean': perm.importances_mean,
            'importance_std': perm.importances_std
        })

        # Sort and add color
        importance_df = importance_df.sort_values(by='importance_mean', ascending=False)

        results_dict = {'importance_mean': importance_df['importance_mean'].tolist(),
                       'importance_std': importance_df['importance_std'].tolist(),
                       'features': importance_df['features'].tolist()} 
            
    elif (method == "shap"):      
        # Transform the data
        X_t_train = pipeline['preprocessor'].transform(X_train)
        X_t_test = pipeline['preprocessor'].transform(X_test)

        # Apply Shap kernel
        # For all callable models (Ridge, XGB, ...)
        try:
            explainer = shap.Explainer(pipeline.named_steps['model'], X_t_train)
            sv = explainer.shap_values(X_t_test)
        # For non callable models (voting regressor, other meta models...)
        except:
            # Wrapper function
            def model_predict(data):
                return pipeline["model"].predict(data)
            
            explainer = shap.KernelExplainer(model_predict, X_t_train)
            sv = explainer.shap_values(X_t_test)
        
        results_dict = {'shap_values': np.array(sv), 
                        'X_test': np.array(X_t_test), 
                        'features': feature_names}
        
    else:
        raise ValueError("Selected method do not exist, please change to shap or permutation")
    
    # Add score results 
    y_predict = pipeline.predict(X_test)
    # Scores
    R2_score = r2_score(y_test, y_predict)
    MAE_score = mean_absolute_error(y_test, y_predict)
    RMSE_score = np.sqrt(mean_squared_error(y_test, y_predict))
    # Add to dictionary
    results_dict["R2_score"] = R2_score
    results_dict["MAE_score"] = MAE_score
    results_dict["RMSE_score"] = RMSE_score
    
    return results_dict





def cv_interpretability(X, y, model_class,  cv_splitter, method = "permutation", groups = None, features_to_use = None,
                        params = None, weights = None, random_state = 42):
    
    if features_to_use:
        # To avoid changing the original
        features = list(features_to_use)
        X = X[features]

    # Model instance
    if isinstance(model_class, list):
        # Any model can be used to get feature order
        feature_names = _get_final_feature_names(X, y, model_class[0], random_state=random_state)
    # For single model
    else:
        feature_names = _get_final_feature_names(X, y, model_class, random_state=random_state)

    # Store results per fold
    shap_values_list = []
    permutation_list = []
    
    X_test_list = [] # To append all for shap visualization

    # Create dictionary to return results
    results_dict = {'method': method}

    # Custom cross-validation loop
    for i, (train_index, test_index) in enumerate(cv_splitter.split(X, y = None, groups = groups)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Initiate model instance (accept both list of parameters per fold and single parameters)
        current_params = _validate_cv_params(i, model_class, params)

        # For voting regressor case 
        if isinstance(model_class, list):
            # params will be a list containing dictionary for each model
            model = _get_voting_regressor_instance(model_class, params_list = current_params, 
                                                   weights = weights, random_state = random_state)
        # Else model instance for other models
        else:
            model = _get_model_instance(model_class, params = current_params, random_state = random_state)

        # Model pipeline (No Feature Selection)
        pipeline = model_pipeline(X_train, model)

        # Fit pipeline to train 
        pipeline.fit(X_train, y_train)
        
        if (method == "permutation"):
            perm = permutation_importance(pipeline, X_test, y_test, n_repeats=30, random_state=random_state, n_jobs=-1)
            permutation_list.append(perm.importances_mean)
            
        elif (method == "shap"):      
            # Transform the data
            X_t_train = pipeline['preprocessor'].transform(X_train)
            X_t_test = pipeline['preprocessor'].transform(X_test)

            # Apply Shap kernel
            # For all callable models (Ridge, XGB, ...)
            try:
                explainer = shap.Explainer(pipeline.named_steps['model'], X_t_train)
                sv = explainer.shap_values(X_t_test)
            # For non callable models (voting regressor, other meta models...)
            except:
                # Wrapper function
                def model_predict(data):
                    return pipeline["model"].predict(data)
                
                explainer = shap.KernelExplainer(model_predict, X_t_train)
                sv = explainer.shap_values(X_t_test)
            
            #Append to list
            shap_values_list.append(np.array(sv))
            X_test_list.append(np.array(X_t_test))
            
        else:
            raise ValueError("Selected method do not exist, please change to shap or permutation")
            
    #Get permutation DataFrame
    if (method == "permutation"):
        permutation_array = np.array(permutation_list)
        
        importance_df = pd.DataFrame({
            'features': X.columns,
            'importance_mean': permutation_array.mean(axis = 0),
            'importance_std': permutation_array.std(axis = 0)
        }).sort_values(by='importance_mean', ascending=False)

        results_dict = {'importance_mean': importance_df['importance_mean'].tolist(),
                       'importance_std': importance_df['importance_std'].tolist(),
                       'features': importance_df['features'].tolist()} 
    
    # Concatenate all CV results to be plotted
    if (method == "shap"):
        shap_values_full = np.concatenate(shap_values_list, axis=0)
        X_test_full = np.concatenate(X_test_list, axis=0)
        
        results_dict = {'shap_values': np.array(shap_values_full), 
                        'X_test': np.array(X_test_full), 
                        'features': feature_names}
        
    return results_dict