
def params_space_search(model_name):
    if model_name == "Ridge":
        def ridge_search_space(trial):
            return {
                "alpha": trial.suggest_float("alpha", 1e-4, 1000.0, log=True),
                "solver": trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr", "sag", "saga"])
            }
        return ridge_search_space
        
    if model_name == "Lasso":
        def lasso_search_space(trial):
            return {
                    "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True)
                }
        return lasso_search_space
        
    if model_name == "ElasticNet":
        def elastic_net_search_space(trial):
            return {
                    "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True),
                    "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)
                }
        return elastic_net_search_space

    if model_name == "SVR":
        def svr_search_space(trial):
            return {
                    "C": trial.suggest_float("C", 0.1, 100.0, log=True),
                    "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
                    "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
                    "degree": trial.suggest_int("degree", 2, 5) if trial.params.get("kernel") == "poly" else 3,
                    "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
                }
        return svr_search_space


    if model_name == "RandomForestRegressor":
        def rf_search_space(trial):
            return {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
                }
        return rf_search_space

    if model_name == "XGBRegressor":
        def xgb_search_space(trial):
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
        return xgb_search_space

    if model_name == "CatBoostRegressor":
        def cat_search_space(trial):
            return {
                    "depth": trial.suggest_int("depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "iterations": trial.suggest_int("iterations", 200, 1000),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                    "border_count": trial.suggest_int("border_count", 32, 254)
                }
        return cat_search_space


    if model_name == "LGBMRegressor":
        def lgbm_search_space(trial):
            return {
                    "num_leaves": trial.suggest_int("num_leaves", 16, 64),
                    "max_depth": trial.suggest_int("max_depth", 2, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
                }
        return lgbm_search_space

    if model_name == "MLPRegressor":
        def mlp_search_space(trial):

            # Define candidate hidden layer patterns
            hidden_options = [
                (32,), (64,), (128,),          # Single layer
                (64, 32), (32, 16), (128, 64), # Two layers
                (64, 32, 16),                  # Three layers (small)
            ]

            return {
                "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes", hidden_options
                ),

                # Best-performing activations for tabular regression
                "activation": trial.suggest_categorical(
                    "activation", ["relu", "tanh"]
                ),

                # Adam is almost always better; lbfgs stays as fallback
                "solver": trial.suggest_categorical(
                    "solver", ["adam"]
                ),

                # Strong regularization recommended
                "alpha": trial.suggest_float("alpha", 1e-4, 1e-1, log=True),

                # Small step sizes perform better
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True),

                "learning_rate": trial.suggest_categorical(
                    "learning_rate", ["adaptive", "invscaling"]
                ),

                # Moderate number of iterations
                "max_iter": trial.suggest_int("max_iter", 300, 1200),

                # Early stopping MUST be used
                "early_stopping": True,

                # Small batch sizes improve generalization
                "batch_size": trial.suggest_categorical(
                    "batch_size", [8, 16, 32]
                ),

                "tol": 1e-4,
            }
        return mlp_search_space
    
    if model_name.startswith("VotingRegressor"):
        def voting_search_space(trial):
            return {
                "weights": [
                    trial.suggest_float("weight_1", 0.0, 1.0),
                    trial.suggest_float("weight_2", 0.0, 1.0),
                    trial.suggest_float("weight_3", 0.0, 1.0)
                ]
            }
        return voting_search_space
