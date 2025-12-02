import shap
import numpy as np

from src.utils.io_utils import load_results_from_json
from src.visualizations.interpretability_plotting import plot_interpretability_bar, interpretability_comparison_plot


""" results = load_results_from_json(f"raw_interpretability_results\XGBRegressor\pi\pi_randomCV.json")


print(results.keys()) """


# shap.summary_plot(np.array(results["shap_values"]), np.array(results['X_test']), results['features'])

""" plot_interpretability_bar(results, 
                           "SHAP Feature Importance for XGBRegressor", 'shap_feature_importance_XGBRegressor.png',
                           method="permutation") """

# Using SHAP method
""" results_shap = load_results_from_json(f"raw_interpretability_results\XGBRegressor\shap\shap_randomCV.json")
plot_interpretability_bar(results_shap, 
                           "SHAP Feature Importance for XGBRegressor", 'shap_feature_importance_XGBRegressor.png',
                           method="shap") """

results ={}

# Load all results
for method in ["pi", "shap"]:
    for key in ['trainTest', 'trainTest_removed', 'randomCV', 'randomCV_removed', 'groupedCV', 'groupedCV_removed']:
        res = load_results_from_json(f"raw_interpretability_results/XGBRegressor/{method}/{method}_{key}.json")
        results[f"{method}_{key}"] = res

interpretability_comparison_plot(
    results['shap_groupedCV_removed'], results['shap_groupedCV'],
    results['shap_randomCV_removed'], results['shap_randomCV'],
    results['shap_trainTest_removed'], results['shap_trainTest'], 
    filename='final_shap_comparison_6bar.png',
    method='shap', title='Comprehensive SHAP Importance Comparison for XGBRegressor'
    )

interpretability_comparison_plot(
    results['pi_groupedCV_removed'], results['pi_groupedCV'],
    results['pi_randomCV_removed'], results['pi_randomCV'],
    results['pi_trainTest_removed'], results['pi_trainTest'], 
    filename='final_pi_comparison_6bar.png',
    method='permutation', title='Comprehensive Permutation Importance Comparison for XGBRegressor'
    )