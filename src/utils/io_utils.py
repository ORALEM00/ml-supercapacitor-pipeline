import numpy as np
import pandas as pd
import os
import json


def _make_json_serializable(obj):
    """
    Converts NumPy, Pandas, and other non-serializable objects into standard
    Python types that can be safely serialized into a JSON format.
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
    elif isinstance(obj, (np.integer, np.floating)):  # handles all NumPy numeric types
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif obj is None:
        return None
    else:
        # Final fallback for objects like GaussianProcess kernels, Optuna types, etc.
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)
    

def save_results_as_json(results_dict, base_path, model_name, stage, filename):
    """
    Save a dictionary to a structured JSON file.

        Parameters:
            results_dict (dict): Dictionary containing results to save
            base_path (str): Parent folder where results will be stored
            model_name (str): The name of the model to create folder
            stage (str): Subdirectory name (e.g. 'baseline', 'tuned')
            filename (str): The output file name (without .json extension)

        Return:
            JSON file
    """
    model_dir = os.path.join(base_path, model_name, stage)
    os.makedirs(model_dir, exist_ok=True)
    file_path = os.path.join(model_dir, f"{filename}.json")

    serializable_dict = _make_json_serializable(results_dict)

    with open(file_path, "w") as f:
        json.dump(serializable_dict, f, indent=4)

    print(f"Results saved to {file_path}")


def load_results_from_json(path):
    """
    Load results from JSON file in path.
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Convert lists back into arrays or Series if needed
    # data["R2_score"] = np.array(data["R2_score"])
    # data["MAE_score"] = np.array(data["MAE_score"])
    # data["RMSE_score"] = np.array(data["RMSE_score"])
    data["y_predict"] = pd.Series(data["y_predict"])
    data["y_true"] = pd.Series(data["y_true"])

    return data