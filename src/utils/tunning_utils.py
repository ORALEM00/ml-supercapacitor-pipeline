import numpy as np
import optuna 
from optuna.samplers import TPESampler
from functools import partial
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel, ConstantKernel

from src.modeling.model_functions import cv_analysis


# Objetive function to be tunned
def objective_cv(trial, X, y, model_class, cv_splitter, groups,  space_search, mu,  random_state):
    params = space_search(trial)

    cv_scores = cv_analysis(X, y, model_class, cv_splitter, groups = groups, feature_selection = True, params = params, random_state= random_state)
    
    # Aggregate values
    mean_score = np.mean(cv_scores['R2_score'])
    std_score = np.std(cv_scores['R2_score'])

    return mean_score - mu * std_score

# Objective function in case of Voting Regressor
def objective_vr_cv(trial, X, y, model_class_list, cv_splitter, groups, space_search, model_params, mu, random_state):
    # Search space of weights
    weights = space_search(trial)['weights']

    cv_scores = cv_analysis(X, y, model_class_list, cv_splitter, groups = groups, feature_selection = True, 
                            params = model_params, weights=weights, random_state= random_state)
    
    # Aggregate values
    mean_score = np.mean(cv_scores['R2_score'])
    std_score = np.std(cv_scores['R2_score'])

    return mean_score - mu * std_score


# Optuna study function to obtain the hyperparameters
def run_study(X, y, model_class, cv_splitter,  space_search, mu = 0, groups = None, params = None,  random_state = 42, 
              n_trials = 50): 
    # For Voting Regressor case
    if isinstance(model_class, list):
        # Wrapped objective function to single trial parameter
        objective = partial(objective_vr_cv,  X = X, y = y, model_class_list = model_class, groups = groups,
                            cv_splitter = cv_splitter, space_search = space_search, 
                            model_params = params, mu = mu, random_state = random_state)
    # For single model case
    else:
        # Wrapped objective function to single trial parameter
        objective = partial(objective_cv, X = X, y = y, model_class = model_class, groups = groups,
                            cv_splitter = cv_splitter, space_search = space_search, mu = mu, 
                            random_state = random_state)
        
    # Optuna study
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print("Best Score:", study.best_value)
    print("Best Params:", study.best_params)
    
    return study

