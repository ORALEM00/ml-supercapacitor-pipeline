import numpy as np
import optuna 
from optuna.samplers import TPESampler
from functools import partial

from ..modeling.training import cv_analysis
from ..modeling._model_utils import _validate_ensemble_model_inputs


def _validate_tuning_inputs(model_class, params, n_folds = None):
    """
    Specific validation for hyperparameter tuning structures. 
    Params can only be a list of dicts or list of lists of dicts 
    for the diferent models in the ensemble. 

    Parameters
    ----------
    model_class : list
        List of estimator classes/instances.
    params : list
        1D list (one dict per model) or 2D list (models x folds).
    n_folds : int, optional
        The number of CV folds, required if params is a 2D list.
    """
    # No params given
    if params is None:
        return

    # Only ensemble params are permited
    if not isinstance(params, list):
        if n_folds is None:
            raise TypeError(
                "Ensemble params must be a list of dictionaries, one for each model. "
                f"Got {type(params).__name__} instead."
            )
        # Case for nested cv
        else:
            raise TypeError(
                f"Params expected a lists of parameter dictionaries (one per model), "
                f"or a 2D list of parameter dictionaries (oner per model)"
                f"each list containing {n_folds} dictionaries (one per fold), "
                f"but instead got {type(params).__name__}."
            )
    elif not isinstance(model_class, list) and isinstance(params, list):
        raise ValueError(
            f"Ensemble params can only be defined when used for tuning a "
            "ensemble model. Here a single model is passed."
        )
    else: 
        # Validate ensemble model inputs
        _validate_ensemble_model_inputs(model_class, params, n_folds = n_folds)




def objective_cv(trial, X, y, model_class, cv_splitter, groups,  space_search, metrics, pipeline_factory, 
                 feature_selection, mu, direction, random_state, **kwargs):
    """
    Objective function for single model hyperparameter optimization. Runs a cross-validation analysis
    and return the penalized score using the ```cv_analysis``` function.

    Calculates a penalized cross-validation score. The penalty factor (mu) 
    adjusts the mean performance by the standard deviation to favor 
    stable models across folds.
    """
    # Define hyperparameter search space for the current trial
    params = space_search(trial)

    # Execute cross-validation analysis using suggested parameters
    cv_scores = cv_analysis(X, y, model_class, cv_splitter, groups = groups, pipeline_factory=pipeline_factory, 
                            feature_selection = feature_selection, params = params, random_state= random_state, 
                            metrics=metrics, return_features=False, **kwargs)
    
    # Extract primary metric for optimization (first key in metrics dictionary)
    primary_key = next(iter(metrics.keys()))
    mean_score = np.mean(cv_scores[primary_key])
    std_score = np.std(cv_scores[primary_key])

    # Calculate penalized score: subtract std if maximizing, add if minimizing
    if direction == 'maximize':
        return mean_score - mu * std_score
    else:
        return mean_score + mu * std_score




def objective_vr_cv(trial, X, y, model_class_list, cv_splitter, groups, space_search, ensemble_params, 
                    metrics, pipeline_factory, feature_selection, mu, direction, random_state, **kwargs):
    """
    Objective function for the voting regressor ensemble hyperparameter optimization, it requires a list
    of model classes. Runs a cross-validation analysis and return the penalized score using the ```cv_analysis``` 
    function.

    Calculates a penalized cross-validation score. The penalty factor (mu) 
    adjusts the mean performance by the standard deviation to favor 
    stable models across folds.
    """
    # Define search space for ensemble weights (sampled from trial)
    weights = space_search(trial)['weights']

    # Execute cross-validation analysis for the ensemble with fixed base-model params
    cv_scores = cv_analysis(X, y, model_class_list, cv_splitter, groups = groups, pipeline_factory=pipeline_factory, 
                            feature_selection = feature_selection, params = ensemble_params, metrics=metrics, weights=weights, 
                            return_features=False, random_state= random_state, **kwargs)
    
    # Select primary metric and calculate stability-adjusted score
    primary_key = next(iter(metrics.keys()))
    mean_score = np.mean(cv_scores[primary_key])
    std_score = np.std(cv_scores[primary_key])

    # Calculate penalized score: subtract std if maximizing, add if minimizing
    if direction == 'maximize':
        return mean_score - mu * std_score
    else:
        return mean_score + mu * std_score




# Optuna study function to obtain the hyperparameters
def run_study(X, y, model_class, cv_splitter,  space_search, metrics, pipeline_factory = None, feature_selection = False, 
              mu = 0, groups = None, ensemble_params = None,  direction = 'maximize', random_state = 42, n_trials = 50, **kwargs): 
    """
    Initialize and execute an Optuna study to find optimal hyperparameters or weights.

    Parameters
    ----------
    X, y : pandas.DataFrame, pandas.Series
        Training design matrix and target vector.
    model_class : type or list of types
        Model class to tune, or list of classes for ensemble weight tuning.
    cv_splitter : scikit-learn splitter
        The cross-validation strategy for evaluating trials.
    space_search : callable
        Function defining the Optuna search space.
    metrics : dict
        Metrics to calculate; the first key is used as the optimization objective.
    pipeline_factory : callable, optional
        A user-defined function that takes ``(model, **kwargs)`` and returns 
        a ``sklearn.pipeline.Pipeline``. If provided, this overrides 
        ``feature_selection``.
    feature_selection : bool, default=False
        Whether to apply automated feature selection (CorrelationSelector) 
        before fitting the model. Only used if ``pipeline_factory`` is None.
    mu : float, default=0
        Penalty coefficient for the standard deviation (stability) of the score.
    groups : array-like, optional
        Group labels for group-based cross-validation.
    ensemble_params : list of dict, optional
        Fixed hyperparameters for base models (used only for VotingRegressor).
    direction : {'maximize', 'minimize'}, default='maximize'
        Optimization direction for the objective value.
    random_state : int, default=42
        Seed for the TPE Sampler and splitting reproducibility.
    n_trials : int, default=50
        Number of optimization trials to perform.

    Returns
    -------
    study : optuna.study.Study
        The completed Optuna study object containing trial history and best parameters.
    """
    # Consolidate arguments shared by both objective function types
    common_args = {
        'X': X, 'y': y, 'cv_splitter': cv_splitter, 'groups': groups, 'pipeline_factory': pipeline_factory,
        'feature_selection': feature_selection,'space_search': space_search, 
        'metrics': metrics, 'mu': mu, 'direction': direction, 
        'random_state': random_state
    }
    # Inject additional keyword arguments for pipelines or selectors
    common_args.update(kwargs)

    # Configure objective function: wrap with partial to bind static arguments
    if isinstance(model_class, list):
        # Handle Voting Regressor case: tuning ensemble weights
        objective = partial(objective_vr_cv, model_class_list = model_class, 
                            ensemble_params = ensemble_params, **common_args)
    else:
        # Handle single model case: tuning estimator hyperparameters
        objective = partial(objective_cv, model_class = model_class, **common_args)
        
    # Initialize the Optuna study:  TPE Sampler for efficient search
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction=direction, sampler=sampler)
    # Execute the optimization loop
    study.optimize(objective, n_trials=n_trials)

    # Output optimization summary to the console
    print("Best Score:", study.best_value)
    print("Best Params:", study.best_params)
    
    return study