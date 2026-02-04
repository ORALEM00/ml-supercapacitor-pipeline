import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from ._tuning_utils import _validate_tuning_inputs, run_study

from ..modeling._model_utils import _REGRESSION_METRICS
from ..modeling._model_utils import _get_model_instance, _get_voting_regressor_instance
from ..modeling._model_utils import _calculate_metrics, _create_cv_results_dict

from ..preprocessing.pipeline import model_pipeline, feature_pipeline
from ..preprocessing._pipeline_utils import  _validate_pipeline, _get_pre_model_pipe


def train_test_tunning(
        X, y, model_class, 
        space_search, 
        inner_splits = 10,
        outer_splitter = None, 
        inner_splitter = None, 
        groups = None,
        ensemble_params = None,
        metrics = _REGRESSION_METRICS, 
        return_features = True,
        pipeline_factory = None, 
        feature_selection = True, 
        mu = 0, direction = 'maximize', n_trials = 50,
        random_state = 42,    
        **kwargs
        ):
    """
    Perform hyperparameter optimization on a train/test split for a given model or ensemble

    This function splits the data into training and testing sets, optimizes hyperparameters
    using cross-validation on the training set, retrains the model with the best parameters,
    and evaluates it on the test set.

    Uses Optuna for hyperparameter optimization via the provided ``run_study`` function in 
    the provided ``search_space``.

    Parameters
    ----------
    X : pandas.DataFrame of shape (n_samples, n_features)
        The design matrix containing the predictor variables.
    y : pandas.Series or array-like of shape (n_samples,)
        The target vector containing the dependent variable.
    model_class : type or list of types
        The estimator class to instantiate (e.g., RandomForestRegressor). 
        If a list of classes is provided, the function instantiates a 
        ``VotingRegressor`` using these models as ensemble.
    space_search : callable
        A function that defines the hyperparameter search space. It must 
        accept an ``optuna.trial.Trial`` object and return a dictionary 
        of sampled parameters.
    inner_splits : int, default=10
        The number of folds for the inner cross-validation if ``inner_splitter`` 
        is None.
    outer_splitter : scikit-learn splitter, optional
        A cross-validation splitter instance (e.g., ``ShuffleSplit``). It 
        takes first precedence for splitting the data. If None, a standard 
        ``train_test_split`` with a 20% test size is performed.
    inner_splitter : scikit-learn splitter, optional
        The strategy for cross-validation during tuning. If None, uses ``KFold`` 
        with ``inner_splits``.
    groups : array-like, optional
        Group labels for the samples used for splitting if a group-based 
        ``splitter`` is provided (e.g. ShuffleGroupKFold). Must be same 
        length as ``X`` and ``y``.
    ensemble_params : list of dict, optional
        Fixed hyperparameters for base models when ``model_class`` is a list, in 
        case of tuning a ``VotingRegressor``.
    metrics : dict, optional
        A dictionary where keys are metric names and values are callable 
        scoring functions. The first metric is used for optimization.
        Defaults to ``_REGRESSION_METRICS``.
    return_features : bool, default=True
        Return the list of feature names used after preprocessing.
    pipeline_factory : callable, optional
        A user-defined function that takes ``(model, **kwargs)`` and returns 
        a ``sklearn.pipeline.Pipeline``. If provided, this overrides 
        ``feature_selection``.
    feature_selection : bool, default=True
        Whether to apply automated feature selection (CorrelationSelector) 
        before fitting the model. Only used if ``pipeline_factory`` is None.
    mu : float, default=0
        Penalty factor for the objective function to balance performance and stability.
        Used in equation: score = mean_score -/+ mu * std_score
    direction : {'maximize', 'minimize'}, default='maximize'
        Whether to optimize for a higher or lower metric value.
    n_trials : int, default=50
        The number of optimization iterations for Optuna.
    random_state : int, default=42
        Seed for reproducibility.
    **kwargs : dict
        Additional keyword arguments. Common keys include:
        
        threshold : float, default=0.68
            The correlation threshold for ``CorrelationSelector`` if 
            ``feature_selection`` is True.
        others :
            Arguments passed directly to the ``pipeline_factory``.

    Returns
    -------
    results : dict
        Consolidated results containing:
        - ``'metrics'``: Performance scores on the held-out test set.
        - ``'y_predict'``: Predictions made on the test set.
        - ``'y_true'``: Actual target values for the test set.
        - ``'features'``: List of features used (if ``return_features`` is True).
        - ``'params'``: The best hyperparameters found during the optimization phase.

    Examples
    --------
        from sklearn.ensemble import RandomForestRegressor
        def rf_search_space(trial):
            return {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                }
        return rf_search_space
        results = train_test_tunning(X, y, RandomForestRegressor, space_search=rf_search_space, n_trials=100) 
        print(results['metrics']['r2'])
    """
    # Validate correct input format
    _validate_tuning_inputs(model_class, ensemble_params, n_folds=None)

    # Implement an 80/20 shuffle split by default or use a provided scikit-learn splitter
    if outer_splitter is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)
    else:
        # Extract indices from the first fold of the provided splitter
        train_index, test_index = next(outer_splitter.split(X, y=None, groups=groups))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Create Inner split: default to KFold or use provided inner splitter logic
    if inner_splitter is  None:
        inner_cv_splitter = KFold(n_splits = inner_splits, shuffle = False)
    else:
        inner_cv_splitter = inner_splitter

    # Ensure group labels are correctly sliced for the inner training subset
    groups_train = groups.iloc[train_index] if groups is not None else None

    # Application of hyperparameter tuning: execute Optuna study on the training subset
    study_model = run_study(X_train, y_train, model_class, inner_cv_splitter, space_search, metrics = metrics, 
                            pipeline_factory=pipeline_factory, feature_selection=feature_selection, groups=groups_train, 
                            mu = mu, ensemble_params=ensemble_params, direction=direction, random_state=random_state, 
                            n_trials=n_trials, **kwargs)
    # Obtain the best parameters discovered during the optimization process
    best_params = study_model.best_params 

    # Model Initialization: supports single estimators or VotingRegressor ensembles using the best parameters
    if isinstance(model_class, list):
        # Convert weights to list in correct model order
        best_params = [float(best_params[f'weight_{i+1}']) for i in range(len(model_class))]
        # params will be a list containing dictionary for each model
        model = _get_voting_regressor_instance(model_class, params_list = ensemble_params, 
                                            weights = best_params, random_state = random_state)
    else:
        # Instantiate the single model with the optimized hyperparameters
        model = _get_model_instance(model_class, params = best_params, random_state = random_state)

    # Pipeline Construction: custom factory, feature selection, or default pipeline
    if pipeline_factory:
        pipeline = pipeline_factory(model, **kwargs)
        pipeline.set_output(transform="pandas")  
        _validate_pipeline(pipeline)

    elif feature_selection: 
        thresh = kwargs.get('threshold', 0.68)
        pipeline = feature_pipeline(model, threshold = thresh)
    else: 
        pipeline = model_pipeline(model)
    
    # Fit the workflow on the training subset
    pipeline.fit(X_train, y_train)

    # Get feature names after preprocessing
    preprocess_pipe = _get_pre_model_pipe(pipeline)
    features = preprocess_pipe.get_feature_names_out() if return_features else None

    # Prediction on the test subset
    y_predict = pipeline.predict(X_test)

    # Metric scores
    metrics = _calculate_metrics(y_test, y_predict, metrics)

    return _create_cv_results_dict(metrics, y_predict, y_test, features, best_params)





def nested_cv_tunning(
        X, y, model_class, 
        outer_splitter, inner_splitter, 
        space_search, 
        groups = None,
        ensemble_params = None,
        metrics = _REGRESSION_METRICS,
        return_features = True,
        pipeline_factory = None, 
        feature_selection = False, 
        mu = 0, direction = 'maximize', n_trials = 50,
        random_state = 42, 
        **kwargs
        ):
    """
    Perform nested cross-validation to tune and evaluate model performance.

    This function implements a full nested CV loop:
    1. Outer Loop: Splits data into training/testing folds for unbiased evaluation.
    2. Inner Loop: For each outer fold, an Optuna study is run on the training 
       partition to find optimal hyperparameters.
    3. Evaluation: The best model for each fold is tested on the held-out outer 
       test partition.

    Parameters
    ----------
    X, y : pandas.DataFrame, pandas.Series
        The design matrix and target vector.
    outer_splitter : scikit-learn splitter
        The strategy for the outer evaluation loop.
    inner_splitter : scikit-learn splitter
        The strategy for the inner tuning loop.
    space_search : callable
        A function that defines the hyperparameter search space. It must 
        accept an ``optuna.trial.Trial`` object and return a dictionary 
        of sampled parameters.
    groups : array-like, optional
        Group labels for the samples used for splitting if a group-based 
        ``splitter`` is provided (e.g. ShuffleGroupKFold). Must be same 
        length as ``X`` and ``y``.
    ensemble_params : list of dict or list of lists of dicts, optional
        Fixed hyperparameters for base models (used for ensembles). If a list 
        of lists is provided, it extracts params per fold for nested evaluation.
    metrics : dict, optional
        A dictionary where keys are metric names and values are callable 
        scoring functions. The first metric is used for optimization.
        Defaults to ``_REGRESSION_METRICS``.
    return_features : bool, default=True
        Return the list of feature names used after preprocessing.
    pipeline_factory : callable, optional
        A user-defined function that takes ``(model, **kwargs)`` and returns 
        a ``sklearn.pipeline.Pipeline``. If provided, this overrides 
        ``feature_selection``.
    feature_selection : bool, default=False
        Whether to apply automated feature selection (CorrelationSelector) 
        before fitting the model. Only used if ``pipeline_factory`` is None.
    mu : float, default=0
        Penalty factor for the objective function to balance performance and stability.
        Used in equation: score = mean_score -/+ mu * std_score
    direction : {'maximize', 'minimize'}, default='maximize'
        Optimization direction for the tuning objective.
    n_trials : int, default=50
        Number of trials for each Optuna study.
    random_state : int, default=42
        Seed for reproducibility.
    **kwargs : dict
        Additional keyword arguments. Common keys include:
        
        threshold : float, default=0.68
            The correlation threshold for ``CorrelationSelector`` if 
            ``feature_selection`` is True.
        others :
            Arguments passed directly to the ``pipeline_factory``.

    Returns
    -------
    results : dict
        Standardized results containing metrics per fold, out-of-fold 
        predictions, tracked features, and the best params found per fold.
    """
    # Validate correct input format
    _validate_tuning_inputs(model_class, ensemble_params, n_folds=outer_splitter.get_n_splits())

    # Store metric scores
    metric_scores = {name: [] for name in metrics.keys()}
    # Store predictions
    y_predict = pd.Series(np.zeros(len(y)), index=y.index)
    # Store features
    features_per_fold = [] if return_features else None
    # Store optimized hyperparameters per fold 
    best_params = []

    # Outer CV folds: iterate through the evaluation splits
    for i, (train_index, test_index) in enumerate(outer_splitter.split(X, y = None, groups = groups)):
        # Slice groups to match the current outer training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Slice groups to match the current outer training and testing sets
        groups_train= groups.iloc[train_index] if groups is not None else None

        # Parameter Handling: extract the ensemble models parameters for the current fold
        params_fold = []
        if ensemble_params is not None and isinstance(ensemble_params, list):
            for model_param in ensemble_params:
                params_fold.append(model_param[i])
        else:
            params_fold = ensemble_params
        
        # Application of hyperparameter tuning: execute Optuna study on the current training fold
        study_model = run_study(X_train, y_train, model_class, inner_splitter, space_search, metrics=metrics, 
                                pipeline_factory=pipeline_factory, feature_selection=feature_selection, groups=groups_train, 
                                mu = mu, ensemble_params=params_fold, direction=direction, random_state=random_state, n_trials=n_trials, 
                                **kwargs)
        print(f"Completed Study for Outer Fold: {i}")

        # Obtain best parameters discovered in inner loop
        current_params = study_model.best_params 
        # Gather best parameters
        best_params.append(current_params)

        # Model Initialization: supports single estimators or VotingRegressor ensembles 
        if isinstance(model_class, list):
            # Extract weight suggestions into an ordered list for the ensemble
            current_params = [float(current_params[f'weight_{i+1}']) for i in range(len(model_class))]
            model = _get_voting_regressor_instance(model_class, params_list = params_fold, 
                                                weights = current_params, random_state = random_state)
        else:
            # Instantiate the single model with the optimized hyperparameters for this fold
            model = _get_model_instance(model_class, params = current_params, random_state = random_state)
        
        # Pipeline Construction: custom factory, feature selection, or default pipeline
        if pipeline_factory:
            pipeline = pipeline_factory(model, **kwargs)
            pipeline.set_output(transform="pandas")  
            _validate_pipeline(pipeline)
            
        elif feature_selection: 
            thresh = kwargs.get('threshold', 0.68)
            pipeline = feature_pipeline(model, threshold = thresh)
        else: 
            pipeline = model_pipeline(model)
        
        # Fit the workflow on the training subset
        pipeline.fit(X_train, y_train)

        # Get feature names after preprocessing if return_features
        preprocess_pipe = _get_pre_model_pipe(pipeline)
        if return_features: 
            # Save selected features
            features = preprocess_pipe.get_feature_names_out()
            features_per_fold.append(features)

        # Prediction on the test subset
        predictions = pipeline.predict(X_test)
        
        # Append Metric Scores
        fold_metrics = _calculate_metrics(y_test, predictions, metrics)
        for name, value in fold_metrics.items():
            metric_scores[name].append(value)
        
        # Save predictions 
        y_predict.iloc[test_index] = predictions
        
    return _create_cv_results_dict(metric_scores, y_predict, y, features_per_fold, best_params)