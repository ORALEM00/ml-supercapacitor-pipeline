import warnings


def _shap_analysis(pipeline, X_train, X_test, background_size, random_state):
    """
    Compute SHAP values for a fitted model within a preprocessing pipeline.

    This internal utility function applies SHAP-based interpretability to the
    trained model contained in a scikit-learn ``Pipeline``. The analysis is
    performed on data that has already been transformed by the preprocessing
    steps of the pipeline, ensuring that explanations correspond to the actual
    feature representation seen by the model.

    The function use ``shap.Explainer`` for fast, model-specific
    explanations (e.g., ``TreeExplainer`` or ``LinearExplainer``). If the model
    type is not supported or automatic detection fails, it falls back to
    ``KernelExplainer``, issuing a warning due to its higher computational cost.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A fitted scikit-learn pipeline. The final step must be a predictive
        estimator exposing a ``predict`` method and is assumed to be named
        ``'model'``.
    X_train : pandas.DataFrame or numpy.ndarray of shape (n_train_samples, n_features)
        The transformed training data used as background for SHAP value
        estimation. This data is assumed to be the output of the pipeline
        preprocessing steps.
    X_test : pandas.DataFrame or numpy.ndarray of shape (n_test_samples, n_features)
        The transformed test data for which SHAP values are computed.
    background_size : int or None
        Number of samples from ``X_train`` to use as background data when
        ``KernelExplainer`` is employed. If ``None``, all training samples are
        used. Reducing this value can significantly decrease computation time
        at the cost of higher variance in SHAP estimates.
    random_state : int
        Random seed used when subsampling the background data for
        ``KernelExplainer``.

    Returns
    -------
    shap_values : numpy.ndarray
        Array of SHAP values with shape
        ``(n_test_samples, n_features)``, representing the contribution of each
        transformed feature to the model prediction for each test sample.

    Notes
    -----
    - SHAP values are computed in the *transformed feature space*, not on the
      raw input features. As a result, explanations reflect the learned feature
      representation produced by the preprocessing pipeline.
    - For tree-based models, SHAP values are computed using path-dependent
      expectations, which are robust to feature correlations introduced by
      preprocessing.
    - For model-agnostic explanations (``KernelExplainer``), SHAP relies on
      background data sampling and may produce high-variance attributions in
      small-data or highly correlated feature settings. 
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "The 'shap' library is required for SHAP analysis. "
            "Please install it via 'pip install shap'."
        )

    # Attempt to use the fast SHAP Explainer
    try:
        explainer = shap.Explainer(pipeline.named_steps['model'], X_train, seed=random_state)
        sv = explainer.shap_values(X_test)
    # Fallback to KernelExplainer for ensembles or custom estimators
    except:
        warnings.warn(
            "Falling back to SHAP KernelExplainer. "
            "This may be slower for large datasets or complex models."
            "Runtime can be reduced by setting `shap_background_size`.",
            UserWarning
        )

        # Wrapper function
        def model_predict(data):
            return pipeline.named_steps['model'].predict(data)
        
        # Reduce background data size if requested to optimize execution time
        if (background_size is not None) and (background_size < X_train.shape[0]):
            # Use a subset of the training data as background
            X_train = shap.sample(X_train, background_size, random_state=random_state)

        explainer = shap.KernelExplainer(model_predict, X_train, seed=random_state)
        sv = explainer.shap_values(X_test)

    return sv