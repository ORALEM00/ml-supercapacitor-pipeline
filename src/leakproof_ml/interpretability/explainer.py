import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from ._explainer_utils import _shap_analysis

from ..modeling._model_utils import _REGRESSION_METRICS, _validate_inputs, _get_cv_params
from ..modeling._model_utils import _get_model_instance, _get_voting_regressor_instance
from ..modeling._model_utils import _calculate_metrics, _create_cv_results_dict

from ..preprocessing.pipeline import model_pipeline
from ..preprocessing._pipeline_utils import  _validate_pipeline, _get_pre_model_pipe



def train_test_interpretability(
        X, y, model_class, 
        method = "permutation", 
        features_to_use = None, 
        pi_n_repeats=30, 
        shap_background_size=None, 
        params = None,  weights = None,
        splitter = None, groups = None,
        metrics = _REGRESSION_METRICS,
        pipeline_factory = None,   
        random_state = 42, 
        **kwargs
        ):
    """
    Evaluate model performance and calculate feature importance on a test split.

    This function integrates model evaluation with interpretability tools 
    (SHAP or Permutation Importance). It ensures that importance is calculated 
    based on the features actually processed by the pipeline steps.

    Parameters
    ----------
    X : pandas.DataFrame of shape (n_samples, n_features)
        The design matrix.
    y : pandas.Series or array-like of shape (n_samples,)
        The target vector.
    model_class : type or list of types
        The estimator class to instantiate. If a list, a ``VotingRegressor`` 
        is created.
    method : {'permutation', 'shap'}, default='permutation'
        The interpretability method to use.
    features_to_use : list of str, optional
        Subset of features to keep from the original dataframe before analysis.
    pi_n_repeats : int, default=30
        Number of times to permute a feature for ``permutation_importance``.
    shap_background_size : int, optional
        Number of samples from the training set to use as SHAP background. 
        Implemented for ``KernelExplainer`` to reduce computation time.
    params : dict or list of dicts, optional
        Hyperparameters for the model. If ``model_class`` is a list, ``params`` 
        must be a list of dictionaries corresponding to each model in the 
        same order.
    weights : array-like, optional
        Weights for the ``VotingRegressor``. Only used if ``model_class`` is a list.
        List length must match number of models.
    splitter : scikit-learn splitter, optional
        A cross-validation splitter instance (e.g., ``ShuffleSplit``). It 
        takes first precedence for splitting the data. If None, a standard 
        ``train_test_split`` with a 20% test size is performed.
    groups : array-like, optional
        Group labels for the samples used for splitting if a group-based 
        ``splitter`` is provided (e.g. ShuffleGroupKFold). Must be same 
        length as ``X`` and ``y``.
    metrics : dict, optional
        A dictionary where keys are metric names and values are callable 
        scoring functions. Defaults to ``_REGRESSION_METRICS``.
    pipeline_factory : callable, optional
        A user-defined function that takes ``(model, **kwargs)`` and returns 
        a ``sklearn.pipeline.Pipeline``.
    random_state : int, default=42
        Seed for reproducibility across splits and importance calculations.
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
        A results dictionary containing performance metrics and:
        - If 'permutation': ``importance_mean``, ``importance_std``, and ``features``.
        - If 'shap': ``shap_values``, ``X_test`` (transformed), and ``features``.

    Examples
    --------
        from sklearn.ensemble import RandomForestRegressor
        results = train_test_interpretability(X, y, RandomForestRegressor, method="shap")
    """
    # Validate correct input format
    _validate_inputs(model_class, params, n_folds = None)
    
    # Subset features if specified
    if features_to_use:
        features = list(features_to_use)
        X = X[features]

    # Create dictionary to return results
    results_dict = {'method': method}
        
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

    else: 
        pipeline = model_pipeline(model)

    # Fit the workflow on the training subset
    pipeline.fit(X_train, y_train)

    # Get preprocessing pipeline
    preprocess_pipe = _get_pre_model_pipe(pipeline)
    # Get feature names after preprocessing
    feature_names = preprocess_pipe.get_feature_names_out()

    # Transform the data with same steps of pipeline for interpretability
    X_t_train = preprocess_pipe.transform(X_train)
    X_t_test = preprocess_pipe.transform(X_test)

    # Interpretability Method Application
    if (method == "permutation"):
        # Calculate Permutation Importance on the test set
        perm = permutation_importance(pipeline.named_steps['model'], X_t_test, y_test, 
                                      n_repeats=pi_n_repeats, random_state=random_state, 
                                      n_jobs=-1)
        
        # Structure results into a sorted dataframe
        importance_df = pd.DataFrame({
            'features': feature_names,
            'importance_mean': perm.importances_mean,
            'importance_std': perm.importances_std
        })

        # Sort and add color
        importance_df = importance_df.sort_values(by='importance_mean', ascending=False)

        results_dict = {'importance_mean': importance_df['importance_mean'].tolist(),
                       'importance_std': importance_df['importance_std'].tolist(),
                       'features': importance_df['features'].tolist()} 
            
    elif (method == "shap"):
        # Calculate SHAP values
        sv = _shap_analysis(pipeline, X_t_train, X_t_test, shap_background_size, random_state)
        
        results_dict = {'shap_values': np.array(sv), 
                        'X_test': np.array(X_t_test), 
                        'features': feature_names}
        
    else:
        raise ValueError("Selected method do not exist, please change to shap or permutation")
    
    # Prediction on the test subset
    y_predict = pipeline.predict(X_test)
    # Metrics Scores
    metrics = _calculate_metrics(y_test, y_predict, metrics)

    return _create_cv_results_dict(metrics, y_predict, y_test, **results_dict)



def cv_interpretability(
        X, y, model_class,  
        splitter, 
        method = "permutation", groups = None, 
        features_to_use = None,
        pi_n_repeats=30, 
        shap_background_size=None,
        params = None, weights = None,
        metrics = _REGRESSION_METRICS, 
        pipeline_factory = None,  
        random_state = 42, 
        **kwargs):
    """
    Perform cross-validated interpretability analysis (SHAP or Permutation Importance).

    This function executes a full cross-validation loop to calculate feature 
    importance. It aggregates results across all folds to provide a global 
    estimate of model behavior, ensuring that every sample in the dataset 
    is used for the final explanation. This function does not support
    feature selection within the pipeline.

    Parameters
    ----------
    X, y : pandas.DataFrame, pandas.Series
        The design matrix and target vector.
    splitter : scikit-learn splitter
        The cross-validation strategy for the analysis.
    method : {'permutation', 'shap'}, default='permutation'
        The interpretability method to apply.
    groups : array-like, optional
        Group labels for group-based cross-validation.
    features_to_use : list of str, optional
        Subset of features to keep from the original dataframe before analysis.
    pi_n_repeats : int, default=30
        Number of permutations per feature (for 'permutation' method).
    shap_background_size : int, optional
        Number of training samples to use as background for SHAP KernelExplainer.
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

    weights : array-like, optional
        Weights for the ``VotingRegressor``. Only used if ``model_class`` is a list.
    metrics : dict, optional
        A dictionary mapping metric names to callable scoring functions. 
        Defaults to ``_REGRESSION_METRICS``.
    pipeline_factory : callable, optional
        A user-defined function that takes ``(model, **kwargs)`` and returns 
        a ``sklearn.pipeline.Pipeline``. For this function, feature selection 
        within the pipeline is not supported.
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
        A standardized results dictionary containing:
        - Out-of-fold metrics and predictions.
        - Aggregated importance data (Mean/Std for permutation, concatenated values for SHAP).

    Examples
    --------
        from sklearn.model_selection import KFold
        from xgboost import XGBRegressor
        cv = KFold(n_splits=3)
        results = cv_interpretability(X, y, XGBRegressor, cv, method="shap")
    """
    # Validate correct input format
    # _validate_cv_inputs(model_class, splitter, params)
    _validate_inputs(model_class, params, n_folds = getattr(splitter, "n_splits", None))
    
    # Subset features if specified
    if features_to_use:
        features = list(features_to_use)
        X = X[features]

    # Store metric scores
    metric_scores = {name: [] for name in metrics.keys()}
    # Store predictions
    y_predict = pd.Series(np.zeros(len(y)), index=y.index)
    # Store results per fold
    shap_values_list = []
    permutation_list = []
    X_test_list = [] 
    # Create dictionary to return results
    results_dict = {'method': method}
    # To store feature names after preprocessing
    feature_names = None

    # Custom cross-validation loop
    for i, (train_index, test_index) in enumerate(splitter.split(X, y = None, groups = groups)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Get model parameters for this fold
        current_params = _get_cv_params(i, model_class, params)

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

        else: 
            pipeline = model_pipeline(model)

        # Fit the workflow on the training subset
        pipeline.fit(X_train, y_train)
        # Get preprocessing pipeline
        preprocess_pipe = _get_pre_model_pipe(pipeline)
        # Get feature names after preprocessing
        if feature_names is None:
            feature_names = preprocess_pipe.get_feature_names_out() 

        # Transform the data with same steps of pipeline for interpretability
        X_t_train = preprocess_pipe.transform(X_train)
        X_t_test = preprocess_pipe.transform(X_test)
        
        # Interpretability Method Application
        if (method == "permutation"):
            # Calculate permutation importance for the current fold
            perm = permutation_importance(pipeline.named_steps['model'], X_t_test, y_test, 
                                          n_repeats=pi_n_repeats, random_state=random_state, 
                                          n_jobs=-1)
            permutation_list.append(perm.importances_mean)
            
        elif (method == "shap"):  
            # Calculate SHAP values for the current fold    
            sv = _shap_analysis(pipeline, X_t_train, X_t_test, shap_background_size, random_state)
            
            # Store fold-specific SHAP values and transformed test data for global concatenation
            shap_values_list.append(np.array(sv))
            X_test_list.append(np.array(X_t_test))
            
        else:
            raise ValueError(
                "Selected method is not currently implemented, please change to shap or permutation"
            )
        
        # Get predictions
        predictions = pipeline.predict(X_test)
        y_predict.iloc[test_index] = predictions

        # Calculate metrics
        fold_metrics = _calculate_metrics(y_test, predictions, metrics)
        for name, value in fold_metrics.items():
            metric_scores[name].append(value)
            
    # Aggregate interpretability results across folds
    if (method == "permutation"):
        # Aggregate permutation results: calculate mean and std across all folds
        permutation_array = np.array(permutation_list)
        importance_df = pd.DataFrame({
            'features': feature_names,
            'importance_mean': permutation_array.mean(axis = 0),
            'importance_std': permutation_array.std(axis = 0)
        }).sort_values(by='importance_mean', ascending=False)

        results_dict = {'importance_mean': importance_df['importance_mean'].tolist(),
                       'importance_std': importance_df['importance_std'].tolist(),
                       'features': importance_df['features'].tolist()} 
    
    # Concatenate all CV results to be plotted
    if (method == "shap"):
        # Concatenate SHAP values from all folds into a single global array
        shap_values_full = np.concatenate(shap_values_list, axis=0)
        X_test_full = np.concatenate(X_test_list, axis=0)
        
        results_dict = {'shap_values': np.array(shap_values_full), 
                        'X_test': np.array(X_test_full), 
                        'features': feature_names}

    return _create_cv_results_dict(metric_scores, y_predict, y,  **results_dict)