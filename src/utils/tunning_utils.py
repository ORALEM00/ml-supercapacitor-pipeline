import numpy as np
import optuna 
from optuna.samplers import TPESampler
from functools import partial
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel, ConstantKernel

from src.modeling.model_functions import cv_analysis


def _models_params(model_name, trial):
    """
    Gives the search space for hyperparameter tunning using the Optuna library
    for a given model
    """
    if model_name == 'LinearRegression':
        # No hyperparameters to tune; it's a deterministic model
        return {}

    elif model_name == 'Ridge':
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)
        }

    elif model_name == 'Lasso':
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True)
        }

    elif model_name == 'ElasticNet':
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)
        }

    elif model_name == 'SVR':
        return {
            "C": trial.suggest_float("C", 0.1, 100.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
            "degree": trial.suggest_int("degree", 2, 5) if trial.params.get("kernel") == "poly" else 3,
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
        }

    elif model_name == 'GaussianProcessRegressor':
        # --- SELECCIÓN DEL KERNEL BASE ---
        kernel_choice = trial.suggest_categorical("kernel", ["RBF", "Matern", "RationalQuadratic"])
        
        # Parámetros Comunes para la reconstrucción
        params = {
            "length_scale": trial.suggest_float("length_scale", 1e-6, 100.0, log=True),
            # KERNEL DE ESCALA (ConstantKernel)
            "C_value": trial.suggest_float("C_value", 1e-4, 1000.0, log=True),
            # KERNEL DE RUIDO (WhiteKernel)
            "WK_sigma": trial.suggest_float("WK_sigma", 1e-6, 1e-1, log=True),
            # Parámetros Directos de GPR
            "alpha": trial.suggest_float("alpha", 1e-10, 1e-2, log=True),
            "normalize_y": trial.suggest_categorical("normalize_y", [True, False]),
            "n_restarts_optimizer": trial.suggest_int("n_restarts_optimizer", 5, 20),
            "kernel": kernel_choice # Mantenemos el string de elección para la reconstrucción
        }
        
        # --- KERNEL BASE ESPECÍFICO ---
        if kernel_choice == "Matern":
            # nu como flotante continuo para optimización
            params["Matern_nu"] = trial.suggest_float("Matern_nu", 0.5, 3.0) 
            
        elif kernel_choice == "RationalQuadratic":
            params["RQ_alpha"] = trial.suggest_float("RQ_alpha", 1e-4, 10.0, log=True)

        return params

    elif model_name == 'RandomForestRegressor':
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
        }
    
    elif model_name == 'XGBRegressor': 
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        }
    
    elif model_name == 'CatBoostRegressor':
        return {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 254)
        }

    elif model_name == 'LGBMRegressor':
        return {
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
        }

    elif model_name == 'MLPRegressor':
        return {
            "hidden_layer_sizes": trial.suggest_int("hidden_layer_sizes", 10, 100),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
            "solver": trial.suggest_categorical("solver", ["adam", "lbfgs"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "max_iter": trial.suggest_int("max_iter", 500, 5000),
            "tol": 1e-4
        }
    
    else:
        raise ValueError(f"Unknown model: {model_name}")



# Objetive function to be tunned
def objective_cv(trial, df, model_class, cv_splitter,  mu,  random_state):
    """
    cv_type -> grouped or random
    """
    model_name = model_class.__name__
    
    # Parameters to explore
    params = _models_params(model_name, trial)

    # For Gaussian Kernel
    if model_name == 'GaussianProcessRegressor':
        params = _reconstruct_gpr_params(params)

    cv_scores = cv_analysis(df, model_class, cv_splitter, feature_selection = True, params = params)
    
    # Aggregate values
    mean_score = np.mean(cv_scores['R2_score'])
    std_score = np.std(cv_scores['R2_score'])

    return mean_score - mu * std_score


# Optuna study function to obtain the hyperparameters
def run_study(df, model_class, cv_splitter, mu = 0, random_state = 42, 
              n_trials = 50): 
    # Wrapped objective function to single trial parameter
    objective = partial(objective_cv, df=df, model_class = model_class, 
                        cv_splitter = cv_splitter, mu = mu, random_state = random_state)
    
    # Optuna study
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print("Best Score:", study.best_value)
    print("Best Params:", study.best_params)
    
    return study

def _reconstruct_gpr_params(optuna_best_params: dict) -> dict:
    """
    Toma el diccionario best_params de Optuna, filtra las claves auxiliares 
    (como 'length_scale', 'C_value', etc.), y reconstruye el objeto Kernel 
    complejo que GaussianProcessRegressor necesita para la instanciación.
    
    Esta versión detecta el tipo de kernel base (RBF, Matern, RQ) basándose 
    en la presencia de sus hiperparámetros únicos en el diccionario, 
    ignorando la posible contaminación de la clave 'kernel'.
    """
    
    # --- 1. Filtrar y Renombrar Parámetros Directos de GPR ---
    
    # Se extraen los parámetros directos que GPR espera
    clean_params = {
        "alpha": optuna_best_params.get('alpha'), 
        "normalize_y": optuna_best_params.get("normalize_y"),
        # Usamos .get() por si el usuario no incluyó estos en su Optuna objective
        "n_restarts_optimizer": optuna_best_params.get("n_restarts_optimizer", 10), 
    }
    
    # --- 2. Detección y Reconstrucción del Objeto Kernel Complejo ---
    
    # Parámetros comunes que DEBEN estar presentes en todos los kernels de este framework
    length_scale = optuna_best_params.get("length_scale")
    constant_value = optuna_best_params.get("C_value")
    sigma_noise = optuna_best_params.get("WK_sigma")
    
    if length_scale is None or constant_value is None or sigma_noise is None:
        raise KeyError("One of the core kernel parameters (length_scale, C_value, or WK_sigma) is missing. Check your Optuna objective definition.")

    # 2a. Detección del Kernel Base por Hiperparámetros Únicos
    if 'Matern_nu' in optuna_best_params:
        # Detected Matern Kernel
        nu = optuna_best_params["Matern_nu"]
        base_kernel = Matern(length_scale=length_scale, nu=nu)
        
    elif 'RQ_alpha' in optuna_best_params:
        # Detected RationalQuadratic Kernel
        alpha_rq = optuna_best_params["RQ_alpha"]
        base_kernel = RationalQuadratic(length_scale=length_scale, alpha=alpha_rq)
        
    else:
        # Default: RBF Kernel
        base_kernel = RBF(length_scale=length_scale)

    # 2b. Combinación Kernels (Constante * Base + Ruido Blanco)
    constant_kernel = ConstantKernel(constant_value=constant_value)
    white_kernel = WhiteKernel(noise_level=sigma_noise)
    
    # Final Kernel Composition: C * Base + White
    final_kernel = constant_kernel * base_kernel + white_kernel
    
    # --- 3. Añadir el kernel reconstruido a los parámetros finales ---
    clean_params["kernel"] = final_kernel
    
    # Asegurar que las claves esenciales estén presentes
    if clean_params.get("alpha") is None or clean_params.get("normalize_y") is None:
         raise KeyError("Alpha or normalize_y parameter missing from Optuna results.")

    return clean_params