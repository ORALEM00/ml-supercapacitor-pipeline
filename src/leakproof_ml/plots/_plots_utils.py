import numpy as np
import pandas as pd

# Returns shap results in same form as permutation importance
def _shap_barPlot_dictionary(shap_dict):
    """
    Convert raw SHAP values into a global importance summary dictionary.

    Calculates the mean absolute value of SHAP contributions across all 
    samples to represent global feature importance. This enables SHAP 
    results to be plotted using the same logic as Permutation Importance.

    Parameters
    ----------
    shap_dict : dict
        A dictionary containing:
        - 'shap_values': ndarray of shape (n_samples, n_features).
        - 'features': list of feature names.

    Returns
    -------
    results_dict : dict
        A dictionary with 'importance_mean', 'importance_std', and 'features', 
        sorted by descending importance.
    """
    importance_df = pd.DataFrame({
        'importance_mean': np.abs(shap_dict['shap_values']).mean(axis=0),
        'importance_std': np.abs(shap_dict['shap_values']).std(axis=0),
        'features': shap_dict['features']
    }).sort_values("importance_mean", ascending=False)

    results_dict = {
         'importance_mean': importance_df['importance_mean'].tolist(),
         'importance_std': importance_df['importance_std'].tolist(),
         'features': importance_df['features'].tolist() 
    }
    return results_dict



def _align_interpretability_dicts(*dicts, key_features="features", 
                                  key_means="importance_mean", key_stds="importance_std"):
    """
    Synchronize multiple interpretability dictionaries to a common feature set.

    This function ensures that all provided dictionaries:
    1. Contain the same list of features.
    2. Maintain the same feature order (defined by the first dictionary).
    3. Fill missing features with 0.0 for both mean and standard deviation.

    Parameters
    ----------
    *dicts : tuple of dicts
        Variable number of dictionaries to align in-place.
    key_features : str, default="features"
        The key used to access feature names.
    key_means : str, default="importance_mean"
        The key used to access the mean importance values.
    key_stds : str, default="importance_std"
        The key used to access the standard deviation values.

    Returns
    -------
    dicts : tuple
        The modified dictionaries (alignment is performed in-place).
    """

    # Establish reference feature order from the first dictionary
    all_features = list(dicts[0][key_features])

    # Identify any extra features present in other dictionaries
    for d in dicts[1:]:
        for f in d[key_features]:
            if f not in all_features:
                all_features.append(f)

    # Reconstruct each dictionary to match the global feature set
    for d in dicts:
        # Create  lookup maps for the current dictionary's data
        feature_map_mean = dict(zip(d[key_features], d[key_means]))
        feature_map_std = dict(zip(d[key_features], d[key_stds]))

        # Reconstruct in the order of all_features
        aligned_means = [feature_map_mean.get(f, 0.0) for f in all_features]
        aligned_stds  = [feature_map_std.get(f, 0.0) for f in all_features]

        # Update the dictionary with aligned data
        d[key_features] = all_features
        d[key_means] = aligned_means
        d[key_stds] = aligned_stds

    return dicts