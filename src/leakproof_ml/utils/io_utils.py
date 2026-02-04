import pandas as pd
import os
import json



def _make_json_serializable(obj):
    """
    Recursively convert non-serializable objects into standard Python types.

    Converts NumPy arrays/scalars, Pandas Series/DataFrames, and other 
    specialized objects into JSON-compatible formats (lists, dicts, floats).

    Parameters
    ----------
    obj : any
        The object to be converted for JSON serialization.

    Returns
    -------
    serializable_obj : any
        The converted object in a standard Python type.
    """
    import numpy as np
    import pandas as pd

    if isinstance(obj, (np.ndarray, pd.Series)): 
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):  
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif obj is None:
        return None
    # Fallback: attempt direct serialization or convert to string representation
    else:
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            # Captures objects like GaussianProcess kernels or Optuna study objects
            return str(obj)
    

def save_results_as_json(results_dict, base_path, model_name, stage, filename):
    """
    Save a result dictionary to a structured directory as a JSON file.

    Creates a hierarchical folder structure: base_path/model_name/stage/filename.json.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing metrics, predictions, and model metadata.
    base_path : str
        The root directory for all experimental results.
    model_name : str
        The name of the model (used as a primary subdirectory).
    stage : str
        The experimental phase (e.g., 'baseline', 'tuned', 'feature_selection').
    filename : str
        The output filename (the .json extension is added automatically).

    Returns
    -------
    None
    """
    # Directory Management: construct and create the destination path
    model_dir = os.path.join(base_path, model_name, stage)
    os.makedirs(model_dir, exist_ok=True)
    file_path = os.path.join(model_dir, f"{filename}.json")

    # Convert results to JSON-serializable format
    serializable_dict = _make_json_serializable(results_dict)

    # Write the JSON file
    with open(file_path, "w") as f:
        json.dump(serializable_dict, f, indent=4)

    print(f"Results saved to {file_path}")


def load_results_from_json(path):
    """
    Load and reconstruct results from a JSON file.

    Restores specific keys (y_predict, y_true) into Pandas Series to 
    maintain compatibility with downstream evaluation functions.

    Parameters
    ----------
    path : str
        The full system path to the .json file.

    Returns
    -------
    data : dict
        The reconstructed dictionary of results.
    """
    # Read the JSON file
    with open(path, "r") as f:
        data = json.load(f)

    # Restore specific keys to Pandas Series
    if "y_predict" in data:
        data["y_predict"] = pd.Series(data["y_predict"])
    if "y_true" in data:
        data["y_true"] = pd.Series(data["y_true"])

    return data