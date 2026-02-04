import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ._model_utils import _REGRESSION_METRICS, _validate_inputs, _get_cv_params
from ._model_utils import _get_model_instance, _get_voting_regressor_instance
from ._model_utils import _calculate_metrics, _create_cv_results_dict

from ..preprocessing.pipeline import model_pipeline, feature_pipeline
from ..preprocessing._pipeline_utils import  _validate_pipeline, _get_pre_model_pipe



def train_test_analysis(
        X, y, model_class, 
        params = None, weights = None,
        splitter = None, groups = None,
        metrics = _REGRESSION_METRICS,
        return_features = True,
        pipeline_factory = None,
        feature_selection = True, 
        random_state = 42, 
        **kwargs
        ):
    """
    Perform a single train-test split evaluation for a given model or ensemble.

    This function handles data splitting, model instantiation (including 
    VotingRegressor), and evaluation. It can use a default pipeline, 
    automated feature selection, or a custom user-provided pipeline factory.

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
    params : dict or list of dicts, optional
        Hyperparameters for the model. If ``model_class`` is a list, ``params`` 
        must be a list of dictionaries corresponding to each model in the 
        same order.
    weights : array-like of shape (n_models,), optional
        Weights for the ``VotingRegressor``. Only used if ``model_class`` is a list.
        List length must match number of models.
    splitter : scikit-learn splitter, optional
        A cross-validation splitter instance (e.g., ``ShuffleSplit``). It 
        takes first precedence for splitting the data. If None, a standard 
        ``train_test_split`` with a 20% test size is performed.
    groups : array-like of shape (n_samples,), optional
        Group labels for the samples used for splitting if a group-based 
        ``splitter`` is provided (e.g. ShuffleGroupKFold). Must be same 
        length as ``X`` and ``y``.
    metrics : dict, optional
        A dictionary where keys are metric names and values are callable 
        scoring functions. Defaults to ``_REGRESSION_METRICS``.
    return_features : bool, default=True
        Return the list of feature names used after preprocessing.
    pipeline_factory : callable, optional
        A user-defined function that takes ``(model, **kwargs)`` and returns 
        a ``sklearn.pipeline.Pipeline``. If provided, this overrides 
        ``feature_selection``.
    feature_selection : bool, default=False
        Whether to apply automated feature selection (CorrelationSelector) 
        before fitting the model. Only used if ``pipeline_factory`` is None.
    random_state : int, default=42
        Seed for reproducibility across splits and model initialization.
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
        A dictionary containing the performance results:
        
        - ``'metrics'``: Calculated scores for the test set.
        - ``'y_predict'``: Model predictions.
        - ``'y_test'``: True test values.
        - ``'features'``: List of selected feature names (if applicable).

    Examples
    --------
        from sklearn.ensemble import RandomForestRegressor
        results = train_test_analysis(X, y, RandomForestRegressor, feature_selection=True)
        print(results['metrics']['r2'])
    """
    # Validate correct input format
    _validate_inputs(model_class, params, n_folds = None)

    # Implement an 80/20 shuffle split by default or use a provided scikit-learn splitter
    if splitter is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)
    else:
        # Extract indices from the first fold of the provided splitter
        train_index, test_index = next(splitter.split(X, y=None, groups=groups))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Model Initialization: supports single estimators or VotingRegressor ensembles
    if isinstance(model_class, list):
        model = _get_voting_regressor_instance(model_class, params_list = params, weights = weights, random_state = random_state)
    else:
        model = _get_model_instance(model_class, params = params, random_state = random_state)
        
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
    metrics_results = _calculate_metrics(y_test, y_predict, metrics)

    return _create_cv_results_dict(metrics_results, y_predict, y_test, features)




def cv_analysis(
        X, y, model_class, 
        splitter, groups = None, 
        params = None, weights = None,
        metrics = _REGRESSION_METRICS,
        return_features = True,
        pipeline_factory = None, 
        feature_selection = True, 
        random_state = 42, 
        **kwargs
        ):
    """
    Perform cross-validated evaluation for a model or voting ensemble.

    Executes a multi-fold cross-validation loop. For each fold, it handles model 
    instantiation, pipeline construction (with optional feature selection), 
    fitting, and out-of-fold prediction storage.

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
    splitter : scikit-learn splitter
        The cross-validation strategy instance (e.g., ``KFold`` or ``GroupKFold``).
    groups : array-like of shape (n_samples,), optional
        Group labels for the samples used for splitting if a group-based 
        ``splitter`` is provided (e.g. ShuffleGroupKFold). Must be same 
        length as ``X`` and ``y``.
    params : dict, list of dict, or list of list of dict, optional
        Hyperparameters for the model(s). The expected structure depends on 
        the ``model_class`` and the validation strategy:

        - **Single Model:**
            - ``dict``: Use the same hyperparameters for all folds.
            - ``list of dict``: Length must match the number of folds. Each 
              element contains parameters for a specific fold (Nested CV).

        - **Voting Regressor (list of models):**
            - ``list of dict``: Length must match the number of models. Each 
              dictionary provides fixed parameters for a specific model across 
              all folds.
            - ``list of list of dict``: A 2D list with shape (n_models, n_folds). 
              Allows parameters to vary both per model and per fold.

    weights : array-like of shape (n_models,), optional
        Weights for the ``VotingRegressor``. Only used if ``model_class`` is a list.
    metrics : dict, optional
        A dictionary mapping metric names to callable scoring functions. 
        Defaults to ``_REGRESSION_METRICS``.
    return_features : bool, default=True
        Return  list of lists with feature names used after preprocessing
        in each fold.
    pipeline_factory : callable, optional
        A user-defined function that takes ``(model, **kwargs)`` and returns 
        a ``sklearn.pipeline.Pipeline``. If provided, this overrides 
        ``feature_selection``.
    feature_selection : bool, default=False
        Whether to apply automated feature selection (CorrelationSelector) 
        before fitting the model. Only used if ``pipeline_factory`` is None.
    random_state : int, default=42
        Seed for reproducibility across splits and model initialization.
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
        A standardized results dictionary containing:
        
        - ``'metrics'``: Lists of scores per fold and aggregated statistics.
        - ``'y_predict'``: Out-of-fold predictions for the entire dataset.
        - ``'y_true'``: The original target values (aligned with predictions).
        - ``'features'``: List of lists of feature names used in each fold.

    Examples
    --------
        from sklearn.model_selection import KFold
        from xgboost import XGBRegressor
        cv = KFold(n_splits=3)
        results = cv_analysis(X, y, XGBRegressor, cv, feature_selection=True)
    """

    # Validate correct input format
    # _validate_cv_inputs(model_class, splitter, params)
    _validate_inputs(model_class, params, n_folds = getattr(splitter, "n_splits", None))
    
    # Store metric scores
    metric_results = {name: [] for name in metrics.keys()}
    # Store predictions
    y_predict = pd.Series(np.zeros(len(y)), index=y.index)
    # Store features
    features_per_fold = [] if return_features else None

    # Custom cross-validation loop: iterate through training and testing folds
    for i, (train_index, test_index) in enumerate(splitter.split(X, y = None, groups = groups)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
        # Get model parameters for this fold
        current_params = _get_cv_params(i,model_class, params)

        # Model Initialization: supports single estimators or VotingRegressor ensembles
        if isinstance(model_class, list):
            model = _get_voting_regressor_instance(model_class, params_list = current_params, 
                                                   weights = weights, random_state = random_state)
        else:
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
            features = preprocess_pipe.get_feature_names_out()
            features_per_fold.append(features)

        # Prediction on the test subset
        predictions = pipeline.predict(X_test)
        # Metric scores
        fold_metrics = _calculate_metrics(y_test, predictions, metrics)
        for name, value in fold_metrics.items():
            metric_results[name].append(value)
        
        # Save predictions 
        y_predict.iloc[test_index] = predictions

    return _create_cv_results_dict(metric_results, y_predict, y, features_per_fold)