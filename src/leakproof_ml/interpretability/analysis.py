import numpy as np



def get_stable_features(features, threshold=0.5):
    """
    Identify features that consistently appear across multiple cross-validation folds.

    In nested cross-validation, feature selection (like Correlation Selection) 
    is performed inside each fold. This results in different sets of features 
    per fold. This function identifies "stable" features—those that were 
    selected frequently enough to meet the specified threshold.

    Parameters
    ----------
    features : list of list of str
        A nested list where each inner list contains the names of features 
        selected in a specific fold. 
        Example: [['feat_A', 'feat_B'], ['feat_A', 'feat_C'], ['feat_A']]
    threshold : float, default=0.5
        The minimum fraction of folds (from 0.0 to 1.0) a feature must 
        appear in to be considered stable. For example, with 5 folds and 
        a 0.6 threshold, a feature must appear in at least 3 folds.

    Returns
    -------
    stable_features : list of str
        An alphabetically sorted list of feature names that met or exceeded 
        the selection frequency threshold.

    """
    selected_features = np.array([f for sublist in features for f in sublist])
    unique, counts = np.unique(selected_features, return_counts=True)
    
    n_folds = len(features)
    # Round number of folds based on threhold
    min_appearances = int(n_folds * threshold)
    
    stable_features = [str(col) for col, count in zip(unique, counts) if count >= min_appearances]
    return stable_features