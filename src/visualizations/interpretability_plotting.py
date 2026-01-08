import numpy as np
import matplotlib.pyplot as plt

from src.utils.interpretability_utils import _shap_barPlot_dictionary, _align_interpretability_dicts


def plot_interpretability_bar(data, title, method = "perm", filename = None):
    """
    Generates a professional horizontal bar plot for a single set of 
    Permutation Importance scores (mean and std).

    Args:
        data (dict): A dictionary containing the PI results.
            Expected format (e.g., from cv_interpretability):
            {
                'feature_names': list of strings (sorted by importance),
                'importance_mean': list of floats,
                'importance_std': list of floats
            }
        title (str): The title of the plot.
        filename (str): The name of the file to save the plot.
    """
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
    Genera un gráfico de barras VERTICAL comparando la Importancia de Permutación 
    de manera compacta y dinámica para ser incluido en un paper científico.
    
    Las etiquetas de las características del eje X se rotan 90 grados (verticalmente)
    para ahorrar espacio.
    """
    feature_names = data['feature_names']
    num_features = len(feature_names)
    
    # Estilo profesional para paper
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- PARÁMETROS DINÁMICOS Y DE AJUSTE CLAVE ---
    
    # 1. Ancho de la barra (bar_width)
    bar_width = 0.3 # Ahora es el ancho de cada barra (antes era altura). Punto de control clave 1.
    separation_factor = 0.08
    
    # 2. Espaciado Horizontal (x_step)
    # Define la separación entre los grupos de 6 barras. 
    gap_factor = 0.8 # Reducido para mayor compacidad horizontal. Punto de control clave 2.
    group_span = 6 * bar_width 
    x_step = group_span + gap_factor 
    x = np.arange(num_features) * x_step # Posiciones centrales de cada grupo de características

    # 3. Tamaño de la figura (figsize)
    # Ajusta la compacidad horizontal. Altura fija para la importancia. Punto de control clave 3.
    fig_width_factor = 0.75 # Factor por característica (ajustar este valor para hacer más compacto/separado)
    # Figura más ancha que alta, adecuada para muchas características en el eje X
    fig, ax = plt.subplots(figsize=(max(10, num_features * fig_width_factor), 7)) 

    # 4. Offsets para las 6 barras (centradas alrededor de x)
    # Se aplican al eje X para separar las 6 barras dentro de un grupo
    effective_bar_width = bar_width + separation_factor # Nuevo ancho efectivo que incluye el espacio
    offsets = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) * effective_bar_width
    
    # --- FIN PARÁMETROS DINÁMICOS ---

    # Define color palettes
    color1_dark, color1_light = '#1f77b4', '#aec7e8'  # Azul
    color2_dark, color2_light = '#ff7f0e', '#ffbb78'  # Naranja
    color3_dark, color3_light = '#2ca02c', '#98df8a'  # Verde

    # --- Trazado de las 6 barras (VERTICALES) para cada característica ---

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

    # Add plot details (Formato científico)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Eje Y: Métrica de importancia (con LaTeX)
    ax.set_ylabel("Mean Permutation Importance (Decrease in $R^2$ Score)", fontsize=12) 
    
    # Colocar los ticks (nombres de las características) en el centro de cada grupo
    ax.set_xticks(x)
    # ¡ROTACIÓN DE 90 GRADOS para las etiquetas del eje X!
    ax.set_xticklabels(feature_names, rotation=45, ha='right')

    
    # Leyenda limpia y visible
    ax.legend(fontsize=10, loc='upper right')
    
    ax.set_ylim(bottom=0)
    
    # Personalizar la cuadrícula
    ax.grid(axis='y', visible = False) # Cuadrícula horizontal
    ax.grid(axis='x', visible=False) # Quitar cuadrícula vertical
    
    # Añadir etiquetas de texto para los valores de la media (encima de la barra)
    all_means = [data['m1_with_means'], data['m1_without_means'], data['m2_with_means'], 
                 data['m2_without_means'], data['m3_with_means'], data['m3_without_means']]
    all_stds = [data['m1_with_stds'], data['m1_without_stds'], data['m2_with_stds'], 
                data['m2_without_stds'], data['m3_with_stds'], data['m3_without_stds']]
    
    for i in range(num_features):
        ax.axvline(x[i] - x_step/2, color="gray", linewidth=0.5, alpha=0.6)
        for j in range(6):
            mean = all_means[j][i]
            std = all_stds[j][i]
            x_pos = x[i] + offsets[j] # Posición central de la barra en X
            offset = (all_means[0][0] + all_stds[0][0]) * 0.02
            y_pos = mean + std + offset # Posición en Y (por encima de la barra + error)
            
            # Solo etiquetar barras que tienen una importancia media significativa
            if mean > 0.00:
                ax.text(x_pos, y_pos,
                        f"{mean:.3f}",
                        ha='center', va='bottom', fontsize=8, rotation=90) # Rotación 90 grados para ahorrar espacio vertical
    
    # Ajuste de diseño para manejar la rotación de las etiquetas y la compacidad
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()



def interpretability_comparison_plot(m1_removed, m1, m2_removed, m2, m3_removed, m3, method = 'perm',
    title = None, filename = None):
    """
    Handles the alignment, structuring, and plotting of the six PI result dictionaries.
    
    NOTE: The input dictionaries are modified in place by the alignment function.
    """
    if method == "shap":
        m1_removed = _shap_barPlot_dictionary(m1_removed)
        m1 = _shap_barPlot_dictionary(m1)
        m2_removed = _shap_barPlot_dictionary(m2_removed)
        m2 = _shap_barPlot_dictionary(m2)
        m3_removed = _shap_barPlot_dictionary(m3_removed)
        m3 = _shap_barPlot_dictionary(m3)

    # 1. Collect and align the six result dictionaries
    aligned_dicts = _align_interpretability_dicts(m1_removed, m1, m2_removed,m2, m3_removed, m3)
    
    # 2. Define labels based on the user's execution structure
    method_labels = ['Grouped CV', 'Random CV','Simple Split']
    
    # 3. Structure the data for the plot_three_pi_comparison function
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
    
    # 4. Generate the plot
    _plot_three_bars(plot_data, title, filename)