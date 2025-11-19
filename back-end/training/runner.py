import numpy as np
import pandas as pd

from models.base import ModelConfig
from models.factory import ModelFactory
from training.trainer import ModelTrainer
from process_data import DataProcessor

class ModelRunner:
    """
    Handles running multiple models with shared data processing.
    This reduced redundant data processing when running multiple models.
    """

    def __init__(self, raw_data, feature_set_manager, hyperparameters, verbose=False):
        self.raw_data = raw_data
        self.feature_set_manager = feature_set_manager
        self.hyperparameters = hyperparameters
        self.verbose = verbose

        # Add caching for processed data
        self._data_cache = {}

        # self.train_ratio, self.val_ratio, self.test_ratio = hyperparameters.get("train_val_test_split", [0.8, 0.1, 0.1])
        
        ratios = hyperparameters.get("train_val_test_split", [0.8, 0.1, 0.1])
        if len(ratios) != 3:
            raise ValueError("train_val_test_split must have exactly 3 elements (train, val, test)")
        self.train_ratio, self.val_ratio, self.test_ratio = ratios

    def run_models(self, model_list):
        """
        Run all models, grouping by data configuration for efficiency.
        """
        
        # Group models by data processing requirements
        model_groups = self._group_models_by_data_config(model_list)

        all_results = {
            "dates": None,
            "actual": None,
            "predictions": {},
            "stats": {}
        }        

        for config_key, models in model_groups.items():
            if self.verbose:
                print(f"Processing group: {config_key} with {len(models)} models")

            # Process data once for this group
            ml_data = self._process_data_for_config(config_key)
            processed_data = ml_data.get_data()

            # Train all models in this group
            for model_config in models:
                try:
                    model_results = self._train_single_model(
                        model_config, ml_data, processed_data
                    )

                    # Merge results
                    if all_results["dates"] is None:
                        all_results["dates"] = model_results["dates"]
                        all_results["actual"] = model_results["actual"]

                    all_results["predictions"].update(model_results["predictions"])
                    all_results["stats"].update(model_results["stats"])

                except Exception as e:
                    model_name = model_config.get("name", "unknown")
                    print(f"Error with model {model_name}: {e}")

                    all_results["predictions"][model_name] = []
                    all_results["stats"][model_name] = {"error": str(e)}
                    if all_results["dates"] is None:
                        all_results["dates"] = []
                        all_results["actual"] = []

        return all_results

    def run_single_model(self, model_config):
        """
        Run a single model (supports both batch and individual processing).
        Uses caching to avoid redundant data processing.
        """
        model_name = model_config["model"]
        display_name = model_config.get("name", model_name)
        
        if self.verbose:
            print(f"Running single model: {display_name}")
        
        # Get data processing parameters
        feature_set_name = model_config.get("featureSet")
        forecast_period = model_config.get("forecastPeriod", 1)
        input_width = model_config.get("inputWidth", 1)
        normalise = model_config.get("normalise", False)
        
        # Create config key for caching
        config_key = (feature_set_name, forecast_period, input_width, normalise)
        
        # Get processed data (potentially from cache)
        ml_data = self._get_processed_data_cached(config_key)
        processed_data = ml_data.get_data()
        
        # Train and evaluate model
        model_results = self._train_single_model(model_config, ml_data, processed_data)
        
        return model_results
    
    def _get_processed_data_cached(self, config_key):
        """
        Get processed data, using cache when possible.
        """
        
        if config_key in self._data_cache:
            if self.verbose:
                print(f"Using cached data for {config_key}")
            return self._data_cache[config_key]
        
        # Process data if not cached
        ml_data = self._process_data_for_config(config_key)
        
        # Cache the result
        self._data_cache[config_key] = ml_data
        
        if self.verbose:
            print(f"Processed and cached data for {config_key}")
        
        return ml_data

    def _group_models_by_data_config(self, model_list):
        """
        Group models by their data processing requirements.
        """
        groups = {}

        for model in model_list:
            config_key = (
                model.get("featureSet"),
                model.get("forecastPeriod", 1),
                model.get("inputWidth", 1),
                model.get("normalise", False)
            )

            if config_key not in groups:
                groups[config_key] = []

            groups[config_key].append(model)

        return groups

    def _process_data_for_config(self, config_key):
        """
        Process data for a specific configuration.
        """
        feature_set_name, forecast_period, input_width, normalise = config_key
        feature_set = self.feature_set_manager.get_feature_set(feature_set_name)

        return DataProcessor(
            raw_data=self.raw_data,
            feature_set=feature_set,
            forecast_period=forecast_period,
            input_width=input_width,
            normalise=normalise,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio
        )
    
    def _train_single_model(self, model_config, ml_data, processed_data):
        """
        Train a single model with pre-processed data.
        """

        model_name = model_config["model"]
        display_name = model_config.get("name", model_name)

        # Create model configuration
        model_params = model_config.get("params", {})
        training_params = {k: v for k, v in model_params.items()
                           if k in ['epochs', 'batch_size', 'optimizer', 'loss', 'learning_rate']}
        architecture_params = {k: v for k, v in model_params.items()
                               if k not in ['epochs', 'batch_size', 'optimizer', 'loss', 'learning_rate']}

        config = ModelConfig(
            model_params=architecture_params,
            **training_params
        )

        # Ensure the window matches this model's forecast period (defensive)
        forecast_period = model_config.get("forecastPeriod", 1)
        # Re-request processed_data with explicit forecast_period (safe no-op if same)
        if hasattr(ml_data, "forecast_period") and forecast_period != ml_data.forecast_period:
            processed_data = ml_data.get_data(forecast_period=forecast_period)

        # --- Compute input/output shapes from processed_data first ---
        x_shape = None
        try:
            x_train_obj = processed_data.get('x_train')
            y_train_obj = processed_data.get('y_train')
            # If x_train is a DataFrame or numpy array get shape directly
            if hasattr(x_train_obj, "shape"):
                x_shape = x_train_obj.shape
            else:
                # Fallback: try to infer from window dataset shape if provided
                # (window datasets are in processed_data['train'], trainer will still re-infer)
                x_shape = None
        except Exception:
            x_shape = None

        if x_shape and len(x_shape) == 3:
            input_shape = (x_shape[1], x_shape[2])
        elif x_shape and len(x_shape) == 2:
            # x_train DataFrame: (n_samples, n_features)
            input_shape = (x_shape[1],)  # (n_features,)
        else:
            # Unknown shape: let trainer/model infer shapes from the tf.data dataset
            input_shape = None

        # Determine output size robustly
        if hasattr(y_train_obj, "shape"):
            if len(y_train_obj.shape) == 1:
                output_size = 1
            elif len(y_train_obj.shape) == 2:
                output_size = y_train_obj.shape[1]
            else:
                output_size = None
        else:
            output_size = None

        # Now populate feature set metadata reliably (use processed_data directly)
        feature_names = None
        n_features = None
        if hasattr(processed_data.get('x_train'), 'columns'):
            # pandas DataFrame
            feature_names = list(processed_data['x_train'].columns)
            n_features = processed_data['x_train'].shape[1]
        elif x_shape and len(x_shape) >= 2:
            # numpy array
            n_features = x_shape[1]
        # attach metadata

        # config.model_params["feature_set_metadata"] = {
        #     "n_features": int(n_features) if n_features is not None else None,
        #     "feature_names": feature_names
        # }

        if isinstance(config.model_params, dict):
            config.model_params["feature_set_metadata"] = {
                'n_features': int(n_features) if n_features is not None else None,
                'feature_names': feature_names
            }

        # config.feature_set_metadata = {
        #     'n_features': int(n_features) if n_features is not None else None,
        #     'feature_names': feature_names
        # }

        # Train model
        trainer = ModelTrainer(config, verbose=self.verbose)

        # If input_shape/output_size are None, ModelTrainer.train_model will infer from the windowed dataset
        trainer.train_model(model_name, processed_data, input_shape, output_size)

        # Show structure
        if self.verbose:
            trainer.show_model(return_string=False, include_weights=True, include_values=True)

        results = trainer.evaluate_model(processed_data)
        print("DEBUG results metrics:", results['metrics'])

        # Post-process results
        y_pred = np.array(results['predictions'])
        y_true = np.array(results['actuals'])

        # Reverse normalisation if needed
        if ml_data.get_normalise():
            target = ml_data.get_target()
            y_pred = ml_data.inverse_transform(y_pred, target)
            y_true = ml_data.inverse_transform(y_true, target)

        # Prepare dates 
        dates_test = self._get_test_dates(ml_data)

        print(f"Model: {model_name}, Forecast Period: {forecast_period}")
        print(f"Predictions shape: {y_pred.shape}")
        print(f"First 5 predictions: {y_pred[:5]}")
        print(f"Dates: {dates_test[:5]}")

        return {
            "dates": dates_test,
            "actual": y_true.tolist(),
            "predictions": {display_name: y_pred.tolist()},
            "stats": {display_name: results['metrics']}
        }
    

    def _get_test_dates(self, ml_data):
        """
        Get test dates for the given data configuration.
        """
        try:
            # Use DataProcessor's logic so dates align with WindowGenerator labels
            dates = ml_data.get_label_timestamps(split="test", forecast_period=getattr(ml_data, "forecast_period", None))
            # Ensure list of strings (fallback if timestamps are pandas.Timestamps)
            return [str(d) for d in dates]
        except Exception:
            # Fallback: approximate using raw_data split indices (legacy behaviour)
            if "date" not in self.raw_data:
                return []
            date_series = pd.to_datetime(self.raw_data["date"], errors="coerce")
            train_len = int(len(self.raw_data) * self.train_ratio)
            val_len = int(len(self.raw_data) * self.val_ratio)
            start_idx = train_len + val_len
            return pd.to_datetime(date_series.iloc[start_idx:]).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

        # if "date" not in self.raw_data:
        #     return []

        # date_series = pd.to_datetime(self.raw_data["date"], errors="coerce")
        # train_len = int(len(self.raw_data) * self.train_ratio)
        # val_len = int(len(self.raw_data) * self.val_ratio)
        # start_idx = train_len + val_len
        # return pd.to_datetime(date_series.iloc[start_idx:]).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    
    def get_cache_info(self):
        """
        Get information about cached data for debugging.
        """
        return {
            "cached_configurations": len(self._data_cache),
            "cache_keys": list(self._data_cache.keys())
        }
    
    def clear_cache(self):
        """
        Clear the data processing cache.
        Useful for memory management or when raw data changes.
        """
        self._data_cache.clear()
        if self.verbose:
            print("Data processing cache cleared")

    def get_cache_memory_usage(self):
        """
        Estimate memory usage of cached data (rough estimate).
        """
        total_size = 0
        for config_key, ml_data in self._data_cache.items():
            try:
                # Rough estimate - not exact but gives an idea
                processed_data = ml_data.get_data()
                for key, data in processed_data.items():
                    if isinstance(data, np.ndarray):
                        total_size += data.nbytes
            except:
                pass

        return {
            "estimated_bytes": total_size,
            "estimated_mb": round(total_size / (1024 * 1024), 2),
            "cached_configs": len(self._data_cache)
        }

