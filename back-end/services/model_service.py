import pandas as pd
from training.runner import ModelRunner
from feature_set import FeatureSetManager

def process_models_request(raw_data_list, model_list, hyperparameters, feature_sets_file, verbose=False):
    """
    Process a models request with the given data and configurations.
    
    Args:
        raw_data_list (list): List of dictionaries representing time series data
        model_list (list): List of model configurations
        hyperparameters (dict): Hyperparameters including train/val/test split
        feature_sets_file (str): Path to feature sets configuration file
        verbose (bool): Whether to enable verbose logging
    
    Returns:
        dict: Results dictionary with success status, predictions, stats, etc.
    
    Raises:
        ValueError: If input validation fails
        Exception: If all models fail to process
    """
    if verbose:
        print(f"Running process_models_request for: {model_list}")
        
    # Input validation
    if not model_list:
        raise ValueError("No models provided")
    if not raw_data_list:
        raise ValueError("No data provided")
    
    # Convert raw data to DataFrame
    try:
        raw_data = pd.DataFrame(raw_data_list)
    except Exception as e:
        raise ValueError(f"Invalid data format: {str(e)}")
    
    # Validate DataFrame has required structure
    if raw_data.empty:
        raise ValueError("Data is empty")
    if 'date' not in raw_data.columns:
        raise ValueError("Data must contain 'date' column")
    
    # Ensure 'price' is numeric
    # for col in ["price"]:  # or any required numeric columns
    #     if col in raw_data.columns:
    #         raw_data[col] = pd.to_numeric(raw_data[col], errors="coerce")

    # if raw_data["price"].isna().any():
    #     raise ValueError("Invalid numeric values found in 'price' column")


    # Initialize feature set manager
    try:
        feature_set_manager = FeatureSetManager(feature_sets_file)
    except Exception as e:
        raise ValueError(f"Failed to load feature sets: {str(e)}")
    
    # Initialize results structure
    results = {
        "success": False,
        "dates": {},
        "actual": {},
        "predictions": {},
        "stats": {},
        "errors": {}
    }
    # results = {
    #     "success": False,
    #     "dates": None,
    #     "actual": None,
    #     "predictions": {},
    #     "stats": {},
    #     "errors": {}
    # }
    
    # Create ModelRunner
    try:
        model_runner = ModelRunner(
            raw_data=raw_data,
            feature_set_manager=feature_set_manager,
            hyperparameters=hyperparameters,
            verbose=verbose
        )
    except Exception as e:
        raise ValueError(f"Failed to initialize ModelRunner: {str(e)}")
    
    successful_models = 0
    
    # Process each model
    for model_config in model_list:
        model_name = model_config.get("name", "unknown")

        model_type = model_config.get('model')
        feature_set = model_config.get('featureSet', '')
        
        if model_type == 'baseline' and not feature_set.startswith('Baseline_'):
            raise ValueError(f"Baseline model must use a Baseline_ feature set, got: {feature_set}")
        if model_type != 'baseline' and feature_set.startswith('Baseline_'):
            raise ValueError(f"Non-baseline models cannot use Baseline_ feature sets")

        try:
            if verbose:
                print(f"Processing model: {model_name}")
            
            # Run single model
            # print("DEBUG about to run model:", model_name)
            model_results = model_runner.run_single_model(model_config)
            # print("DEBUG result for", model_name, ":", type(model_results), model_results)

            # Store per-model dates and actuals
            if model_results.get("dates"):
                results["dates"][model_name] = model_results["dates"]
            if model_results.get("actual"):
                results["actual"][model_name] = model_results["actual"]

            # Merge predictions and stats as before
            if "predictions" in model_results:
                results["predictions"].update(model_results["predictions"])
            if "stats" in model_results:
                results["stats"].update(model_results["stats"])

            # Set shared fields only once (from first successful model)
            # if results["dates"] is None and model_results.get("dates"):
            #     results["dates"] = model_results["dates"]
            # if results["actual"] is None and model_results.get("actual"):
            #     results["actual"] = model_results["actual"]
            
            # # Merge model-specific results
            # if "predictions" in model_results:
            #     results["predictions"].update(model_results["predictions"])
            # if "stats" in model_results:
            #     results["stats"].update(model_results["stats"])
            
            successful_models += 1
            
        except Exception as e:
            error_msg = str(e)
            if verbose:
                print(f"Error processing model {model_name}: {error_msg}")
            
            # Record error but continue with other models
            results["predictions"][model_name] = []
            results["stats"][model_name] = {"error": error_msg}
            results["errors"][model_name] = error_msg
    
    # Determine overall success
    results["success"] = successful_models > 0
    
    # Add metadata about processing
    results["metadata"] = {
        "total_models": len(model_list),
        "successful_models": successful_models,
        "failed_models": len(model_list) - successful_models,
        "cache_info": model_runner.get_cache_info()
    }
    
    if successful_models == 0:
        return {
            "status": "error",
            "error": "All models failed to process",
            "results": []
        }
    
    return results
