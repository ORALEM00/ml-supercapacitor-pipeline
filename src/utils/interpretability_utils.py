import numpy as np
import pandas as pd

from src.modeling.pipeline_transform import model_pipeline, feature_pipeline
from src.utils.model_utils import _get_model_instance



#To get correct order of features, since the preprocessing step changes the order

def _get_final_feature_names(X, y, model_class, random_state = 42):

    # No need of parameters since we only want the model instance
    model = _get_model_instance(model_class, params=None, random_state=random_state)
    pipeline = feature_pipeline(X, model)
    pipeline.fit(X,y)
    
    preprocessor = pipeline.named_steps['preprocessor']

    original_columns = X.columns
    
    # Get the names of the continuous features that were scaled
    feature_names = preprocessor.get_feature_names_out()
    
    return feature_names


# Returns shap results in same form as permutation importance
def _shap_barPlot_dictionary(shap_dict):
    # Create df to sort values in correct descending order
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
    Alinea múltiples diccionarios de interpretabilidad para que:
    - Tengan las mismas features en el mismo orden.
    - El orden está dado por el primer diccionario.
    - Si faltan features en alguno, se agregan al final con valor 0.0 en mean y std.
    
    Modifica los diccionarios en sitio.
    """

    # 1. Empezamos con el orden del primer diccionario
    all_features = list(dicts[0][key_features])

    # 2. Revisar si hay features extra en los demás diccionarios
    for d in dicts[1:]:
        for f in d[key_features]:
            if f not in all_features:
                all_features.append(f)

    # 3. Reconstruir todos los diccionarios con ese orden
    for d in dicts:
        # Crear mapeos para este diccionario
        feature_map_mean = dict(zip(d[key_features], d[key_means]))
        feature_map_std = dict(zip(d[key_features], d[key_stds]))

        # Reconstruir en el orden de all_features
        aligned_means = [feature_map_mean.get(f, 0.0) for f in all_features]
        aligned_stds  = [feature_map_std.get(f, 0.0) for f in all_features]

        # Mutar directamente
        d[key_features] = all_features
        d[key_means] = aligned_means
        d[key_stds] = aligned_stds

    return dicts