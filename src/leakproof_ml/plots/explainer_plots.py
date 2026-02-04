import numpy as np
import matplotlib.pyplot as plt

from ._plots_utils import _shap_barPlot_dictionary, _align_interpretability_dicts


def plot_interpretability_bar(data, title, method = "perm", filename = None):
    """
    Generate a horizontal bar plot for feature importance.

    Visualizes mean importance scores with associated error bars (standard deviation). 
    Supports both Permutation Importance and SHAP values by automatically 
    converting SHAP results into a global importance format.

    Parameters
    ----------
    data : dict
        A dictionary containing 'features', 'importance_mean', and 'importance_std'.
    title : str
        The title of the plot.
    method : {'perm', 'shap'}, default='perm'
        The interpretability method used. If 'shap', raw values are converted 
        to global importance magnitudes.
    filename : str, optional
        Path to save the plot. If None, the plot is displayed interactively.

    Returns
    -------
    None
    """
    # Convert raw SHAP values into a global importance summary dictionary
    if method == "shap":
        data = _shap_barPlot_dictionary(data)
        
    feature_names = data['features']
    means = data['importance_mean']
    stds = data['importance_std']
    
    num_features = len(feature_names)
    
    # Increase height for readability based on number of features
    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, ax = plt.subplots(figsize=(10, max(5, num_features * 0.4))) 

    y = np.arange(num_features)
    bar_height = 0.6
    
    # Plot the bars
    ax.barh(y, means, bar_height,
           xerr=stds, capsize=5,
           color='#1f77b4', edgecolor='black')

    # Add plot details
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Mean Permutation Importance (Decrease in Score)", fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels(feature_names, fontsize=12)
    ax.set_xlim(left=0) # Importance should start at zero
    ax.invert_yaxis()

    # Customize the grid and spines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.grid(axis='y', visible=False)
    
    # Add text labels for the mean values next to the bars
    for i in range(num_features):
        offset = (means[0] + stds[0]) * 0.005
        x_pos = means[i] + stds[i] + offset
        ax.text(x_pos, y[i],
                f"{means[i]:.3f}",
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()




def _plot_three_bars(data, title, filename = None):
    """
    Generate a high-density vertical grouped bar plot for feature importance.

    Designed for scientific papers, this function compares three methodologies 
    simultaneously across all features. It uses dynamic scaling, 90-degree 
    text rotations, and precise offsets to maintain clarity even when 
    visualizing dozens of features.

    Parameters
    ----------
    data : dict
        Dictionary containing the aligned importance data. Expected keys:
        - 'feature_names': list of str.
        - 'm1_with_means', 'm1_with_stds', 'm1_without_means', etc.: 
          Lists of importance scores for each methodology/outlier state.
        - 'm1_with_label', etc.: Strings for the legend entries.
    title : str
        The main title of the plot.
    filename : str, optional
        Path to save the plot. If None, the plot is displayed interactively.

    Returns
    -------
    None
    """
  
    feature_names = data['feature_names']
    num_features = len(feature_names)
    
    # Style settings
    plt.style.use('seaborn-v0_8-whitegrid')

    # Parameters for dynamic layout adjustments
    
    # Bar Width and Separation
    bar_width = 0.3 
    separation_factor = 0.08
    
    # Horizontal Spacing
    gap_factor = 0.8 
    group_span = 6 * bar_width 
    x_step = group_span + gap_factor 
    x = np.arange(num_features) * x_step

    # Dynamic Figure Size
    fig_width_factor = 0.75 # Adjust based on feature count
    fig, ax = plt.subplots(figsize=(max(10, num_features * fig_width_factor), 7)) 

    # Offset Calculation: Positions 6 bars symmetrically around each feature center
    effective_bar_width = bar_width + separation_factor 
    offsets = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) * effective_bar_width

    # Color Palette
    color1_dark, color1_light = '#1f77b4', '#aec7e8'  # Azul
    color2_dark, color2_light = '#ff7f0e', '#ffbb78'  # Naranja
    color3_dark, color3_light = '#2ca02c', '#98df8a'  # Verde

   # Plots 6 bars per feature, including error caps for standard deviation

    # M1 - With Outliers (Dark Blue/Grey)
    ax.bar(x + offsets[0], data['m1_with_means'], bar_width, yerr=data['m1_with_stds'], capsize=3, linewidth=0.4, 
            label=data['m1_with_label'], color=color1_dark, edgecolor='black', error_kw=dict(lw=0.7, capsize=2, capthick=0.7))
            
    # M1 - Without Outliers (Light Blue/Grey)
    ax.bar(x + offsets[1], data['m1_without_means'], bar_width, yerr=data['m1_without_stds'], capsize=3,linewidth=0.4, 
            label=data['m1_without_label'], color=color1_light, edgecolor='black', error_kw=dict(lw=0.7, capsize=2, capthick=0.7))
            
    # M2 - With Outliers (Dark Red/Brown)
    ax.bar(x + offsets[2], data['m2_with_means'], bar_width, yerr=data['m2_with_stds'], capsize=3,linewidth=0.4, 
            label=data['m2_with_label'], color=color2_dark, edgecolor='black', error_kw=dict(lw=0.7, capsize=2, capthick=0.7))

    # M2 - Without Outliers (Light Red/Brown)
    ax.bar(x + offsets[3], data['m2_without_means'], bar_width, yerr=data['m2_without_stds'], capsize=3,linewidth=0.4, 
            label=data['m2_without_label'], color=color2_light, edgecolor='black', error_kw=dict(lw=0.7, capsize=2, capthick=0.7))
            
    # M3 - With Outliers (Dark Green/Olive)
    ax.bar(x + offsets[4], data['m3_with_means'], bar_width, yerr=data['m3_with_stds'], capsize=3,linewidth=0.4, 
            label=data['m3_with_label'], color=color3_dark, edgecolor='black', error_kw=dict(lw=0.7, capsize=2, capthick=0.7))
            
    # M3 - Without Outliers (Light Green/Olive)
    ax.bar(x + offsets[5], data['m3_without_means'], bar_width, yerr=data['m3_without_stds'], capsize=3,linewidth=0.4, 
            label=data['m3_without_label'], color=color3_light, edgecolor='black',error_kw=dict(lw=0.7, capsize=2, capthick=0.7))

    # Add plot details 
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("Mean Permutation Importance (Decrease in $R^2$ Score)", fontsize=12) 
    ax.set_xticks(x)
    # Rotate x-axis labels for readability
    ax.set_xticklabels(feature_names, rotation=45, ha='right')

    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', visible = False) 
    ax.grid(axis='x', visible=False) 
    
    # Add text labels above each bar for mean importance values
    all_means = [data['m1_with_means'], data['m1_without_means'], data['m2_with_means'], 
                 data['m2_without_means'], data['m3_with_means'], data['m3_without_means']]
    all_stds = [data['m1_with_stds'], data['m1_without_stds'], data['m2_with_stds'], 
                data['m2_without_stds'], data['m3_with_stds'], data['m3_without_stds']]
    
    for i in range(num_features):
        # Vertical divider for feature groups
        ax.axvline(x[i] - x_step/2, color="gray", linewidth=0.5, alpha=0.6)
        for j in range(6):
            mean = all_means[j][i]
            std = all_stds[j][i]
            x_pos = x[i] + offsets[j] # Central position of the bar
            offset = (all_means[0][0] + all_stds[0][0]) * 0.02
            y_pos = mean + std + offset # Y position above the error bar
            
            # Only add label if mean importance is greater than 0.00
            if mean > 0.00:
                ax.text(x_pos, y_pos,
                        f"{mean:.3f}",
                        ha='center', va='bottom', fontsize=8, rotation=90) 
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()



def interpretability_comparison_plot(m1_removed, m1, m2_removed, m2, m3_removed, m3, method = 'perm',
    title = None, filename = None):
    """
    Coordinate the alignment and plotting of six distinct interpretability result sets.

    This function aligns feature sets across six dictionaries (representing 
    three methodologies with/without outliers) and generates a grouped vertical 
    bar plot for comparative analysis.

    Parameters
    ----------
    m1_removed, m1 : dict
        Results for Methodology 1 (e.g., Simple Split) without and with outliers.
    m2_removed, m2 : dict
        Results for Methodology 2 (e.g., Random CV) without and with outliers.
    m3_removed, m3 : dict
        Results for Methodology 3 (e.g., Grouped CV) without and with outliers.
    method : {'perm', 'shap'}, default='perm'
        The interpretability method used.
    title : str, optional
        The main title for the comparison plot.
    filename : str, optional
        Path to save the resulting high-resolution image.
    """
    # Ensure all dictionaries are in global importance format
    if method == "shap":
        m1_removed = _shap_barPlot_dictionary(m1_removed)
        m1 = _shap_barPlot_dictionary(m1)
        m2_removed = _shap_barPlot_dictionary(m2_removed)
        m2 = _shap_barPlot_dictionary(m2)
        m3_removed = _shap_barPlot_dictionary(m3_removed)
        m3 = _shap_barPlot_dictionary(m3)

    # Collect and align the six result dictionaries
    aligned_dicts = _align_interpretability_dicts(m1_removed, m1, m2_removed,m2, m3_removed, m3)
    
    # Define labels based on the user's execution structure
    method_labels = ['Grouped CV', 'Random CV','Simple Split']
    
    # Structure the data for the plot_three_pi_comparison function
    plot_data = {
        'feature_names': aligned_dicts[0]['features'], # All dicts now share the same feature order
        
        # M1: Simple Split / Train-Test
        'm1_with_label': f"{method_labels[0]} (Without Outliers)",
        'm1_with_means': aligned_dicts[0]['importance_mean'],
        'm1_with_stds': aligned_dicts[0]['importance_std'],
        'm1_without_label': f"{method_labels[0]} (With Outliers)",
        'm1_without_means': aligned_dicts[1]['importance_mean'],
        'm1_without_stds': aligned_dicts[1]['importance_std'],
        
        # M2: Random CV
        'm2_with_label': f"{method_labels[1]} (Without Outliers)",
        'm2_with_means': aligned_dicts[2]['importance_mean'],
        'm2_with_stds': aligned_dicts[2]['importance_std'],
        'm2_without_label': f"{method_labels[1]} (With Outliers)",
        'm2_without_means': aligned_dicts[3]['importance_mean'],
        'm2_without_stds': aligned_dicts[3]['importance_std'],
        
        # M3: Grouped CV
        'm3_with_label': f"{method_labels[2]} (Without Outliers)",
        'm3_with_means': aligned_dicts[4]['importance_mean'],
        'm3_with_stds': aligned_dicts[4]['importance_std'],
        'm3_without_label': f"{method_labels[2]} (With Outliers)",
        'm3_without_means': aligned_dicts[5]['importance_mean'],
        'm3_without_stds': aligned_dicts[5]['importance_std'],
    }
    
    # Generate plot
    _plot_three_bars(plot_data, title, filename)