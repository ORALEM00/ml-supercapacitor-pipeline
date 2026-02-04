import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Registry to suppress output across different ML libraries
_VERBOSITY_KWARGS = {
    'XGBRegressor': {"verbosity": 0, "silent": True},
    'LGBMRegressor': {"verbose": -1, "verbosity": -1},
    'CatBoostRegressor': {"verbose": False},
    'MLPRegressor': {"verbose": False},
}



# Default standard regression metrics
_REGRESSION_METRICS = {
    'R2_score': r2_score,
    'MAE_score': mean_absolute_error,
    'RMSE_score': lambda y, p: np.sqrt(mean_squared_error(y, p))
}




def _get_model_instance(model_class, params=None, random_state = 42):
    """
    Instantiates a scikit-learn model, handling models verbosity and
    seed handling for models.

    Parameters
    ----------
    model_class : type
        The scikit-learn-compatible model class to instantiate.
    params : dict, optional
        Hyperparameters to pass to the model constructor.
    random_state : int, default=42
        Seed for reproducibility. Injected only if the model class supports it.

    Returns
    -------
    model : object
        An instantiated scikit-learn estimator.
    """
    model_name = model_class.__name__
    # Verbosity suppression if applicable
    silence_params = _VERBOSITY_KWARGS.get(model_name, {})

    if params is not None and not isinstance(params, dict):
        raise TypeError(f"Params must be a dict, got {type(params).__name__}.")

    # Combine user params with silence params
    all_params = {**(params or {}), **silence_params}

    try:
        model = model_class(random_state=random_state, **all_params)
    except TypeError:
        # Fallback for models that don't accept random_state (e.g., LinearRegression)
        model = model_class(**all_params)
    return model




def _get_voting_regressor_instance(models_class_list, params_list = None, weights = None, random_state = 42):
    """
    Create a VotingRegressor ensemble instance from a list of model classes.

    Parameters
    ----------
    models_class_list : list of type
        List of estimator classes to include in the ensemble.
    params_list : list of dict, optional
        List of hyperparameter dictionaries, one for each model class.
    weights : array-like of shape (n_models,), optional
        Sequence of weights for the voting aggregation.
    random_state : int, default=42
        Seed for the individual estimators.

    Returns
    -------
    model_ensemble : VotingRegressor
        An instantiated scikit-learn VotingRegressor.
    """
    # If params list of models and params must be same length
    if params_list is not None and len(models_class_list) != len(params_list):
        raise ValueError(
            f"Length mismatch: models_class_list has {len(models_class_list)} elements, "
            f"but params_list has {len(params_list)} elements."
        ) 

    estimators = []
    for i in range(len(models_class_list)):
        # Parameters must be in same order as model classes in list
        params = params_list[i] if params_list is not None else None
        model_instance = _get_model_instance(models_class_list[i], params = params, random_state = random_state)
        # Add to list of model tuples
        estimators.append((f"model_{i}", model_instance))

    # Use estimators to instance the voting regressor
    return VotingRegressor(estimators = estimators, weights = weights)




def _calculate_metrics(y_true, y_pred, metrics):
    """
    Calculate multiple scores between true and predicted values.

    Parameters
    ----------
    y_true : array-like
        Original target values.
    y_pred : array-like
        Predicted target values.
    metrics : dict
        Mapping of metric names to callable scoring functions.

    Returns
    -------
    scores : dict
        Mapping of metric names to calculated floating-point scores.
    """
    return {name: func(y_true, y_pred) for name, func in metrics.items()}




def _create_cv_results_dict(metrics, y_predict, y_true, features = None, params = None, **kwargs):
    """
    Consolidate evaluation data into a standardized results dictionary.

    Parameters
    ----------
    metrics : dict
        Calculated scores (from _calculate_metrics).
    y_predict : array-like
        Model predictions.
    y_true : array-like
        Actual target values.
    features : list of str, optional
        Names of the features used by the model.
    params : dict, optional
        Hyperparameters used for the model.
    **kwargs : dict
        Additional data to store (e.g., shap_values, importance_std).

    Returns
    -------
    results : dict
        Standardized dictionary containing all experimental outputs.
    """
    results = {
        'y_predict': y_predict,
        'y_true': y_true,
    }
    results.update(metrics)

    if features is not None:
        results['features'] = features
    if params is not None:
        results['params'] = params
    
    results.update(kwargs)

    return results
    


def _validate_ensemble_model_inputs(model_class, params, n_folds = None):
    """
    Validate parameter structure for ensemble (Voting) models.

    Checks if the dimensions of the parameter list align with both the number 
    of models in the ensemble, and if a 2D list is passes, that it is 
    aligned with the number of models and number of CV folds. 

    Parameters
    ----------
    model_class : list
        List of estimator classes/instances.
    params : list
        1D list (one dict per model) or 2D list (models x folds).
    n_folds : int, optional
        The number of CV folds, required if params is a 2D list.
    """
    if not isinstance(model_class, list):
        return
    
    # Number of models
    n_models = len(model_class) 
    params_dim = np.ndim(params)

    # Fixed parameters (1D list)
    if params_dim == 1:
        if not all(isinstance(p, dict) for p in params):
            raise TypeError(
                "For a VotingRegressor, when passing a single list of parameters, "
                "each element must be a dictionary corresponding to a model."
                "An element passed is not a dictionary."
            )
        # Lenght error is already handled inside the _get_voting_regressor_instance
    
    # Parameters change per fold (2D list)
    elif params_dim == 2:
        if n_folds is None:
            raise ValueError(
                "Function expected either a dict or list of dicts (for ensemble model). " \
                "Instead got a 2D list. "
            ) 
        # Not given enough parameters for all models
        if n_models != len(params):
            raise ValueError(
                f"Params expected {n_models} lists of parameter dictionaries (one per model), "
                f"but got {len(params)} params."
            )
        
        # List of parameter contain less parameters than number of folds
        for fold_idx, fold_params in enumerate(params):
            if len(fold_params) != n_folds:
                raise ValueError(
                    f"Params expected {n_models} lists of parameter dictionaries (one per model), "
                    f"or a 2D lists of parameter dictionaries (one per model),"
                    f"each list containing {n_folds} dictionaries (one per fold), "
                    f"but instead got {len(fold_params)} in fold {fold_idx}."
                )
    
    else:
        if n_folds is None:
            raise ValueError(
                f"Function expected either a dict or list of dicts (for ensemble model). " \
                f"Instead got a {params_dim}D list. "
            ) 
        
        raise ValueError(
            f"Invalid params shape for VotingRegressor: expected 1D or 2D list, "
            f"got {params_dim}D."
        )

def _validate_single_model_inputs(model_class, params, n_folds = None):
    """
    Validate parameter structure for a single estimator.

    Ensures params is either a single dictionary (fixed) or a list of 
    dictionaries matching the number of folds.

    Parameters
    ----------
    model_class : list
        List of estimator classes/instances.
    params : dict or list
        1 dict or 1D list (one dict per fold) 
    n_folds : int, optional
        The number of CV folds, required if params is a 2D list.
    """
    if isinstance(params,list):
        # Lenght mismatch 
        if len(params) != n_folds:
            raise ValueError(
                f"Length mismatch: Params expected a single dictionary or a list of {n_folds} dictionaries"
                f"(one per fold), but instead got {len(params)} dicts."
            ) 
        # All must be dictionaries
        elif not all(isinstance(p, dict) for p in params):
            raise TypeError(
                f"Params expected a single dictionary or a list of {n_folds} dictionaries (one per fold),"
                "an element in the list passed is not a dictionary."
            )
    # Single dictionary case
    elif not isinstance(params, dict):
        raise TypeError(
            "Params expected a single dictionary or a list of dictionaries (one per fold),"
            f"but got instead {type(params).__name__}."
        )




def _validate_inputs(model_class, params, n_folds = None):
    """
    Validate the structural integrity of model parameters in the CV analysis.

    This function performs dimensionality and type checking to ensure that 
    the provided parameters match the number of models and the number of 
    cross-validation folds.

    Parameters
    ----------
    model_class : type or list
        The estimator class or a list of classes for ensemble methods.
    params : dict, list, or list of lists, optional
        The hyperparameter structure to be used in the CV process.
    n_folds : int, optional
        The number of cross-validation splits.

    Raises
    ------
    ValueError
        If the length of parameter lists does not match the number of models 
        or CV folds.
    TypeError
        If the nested elements are not dictionaries.
    """
    # No params given
    if params is None:
        return
    
    # Case for ensemble models
    if isinstance(model_class, list):
        _validate_ensemble_model_inputs(model_class, params, n_folds)
        return
    else:   
        # Case for single model
        _validate_single_model_inputs(model_class, params, n_folds)
        return 
        



def _get_cv_params(i,model_class, params):
    """
    Extract the hyperparameter dictionary for the current fold.

    Parameters
    ----------
    i : int
        The current fold index (0-indexed).
    model_class : type or list
        The model class or list of classes.
    params : dict or list
        The full parameter structure.

    Returns
    -------
    current_params : dict or list of dict
        The parameters to be used for the current fold.
    """
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

    return current_params