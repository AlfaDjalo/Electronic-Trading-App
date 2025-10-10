import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass (frozen=True)
class ModelConfig:
    """
    Configuration for model training and architecture.
    """
    epochs: int = 20
    batch_size: int = 32
    optimizer: str = "adam"
    learning_rate: Optional[float] = None
    loss: str = "mse"
    metrics: list = None

    model_params: Dict[str, Any] = None

    # Callbacks
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5

    def __post_init__(self):
        if self.metrics is None:
            object.__setattr__(self, "metrics", ["mae"])
        if self.model_params is None:
            object.__setattr__(self, "model_params", {})

class BaseModel(ABC):
    """
    Base class for all ML models.
    """
    def __init__(self, name:str, config: ModelConfig):
        self.name = name
        self.config = config
        self.model = None
        self.history = None
        self.is_built = False
        self.is_compiled = False

    @abstractmethod
    def build_architecture(self, input_shape: Tuple[int, ...], output_size: int) -> tf.keras.Model:
        """
        Build and return the model architecture.
        """
        pass

    def build(self, input_shape: Tuple[int, ...], output_size: int):
        """
        Build the model if not already build.
        """
        if not self.is_built:
            self.model = self.build_architecture(input_shape, output_size)
            self.is_built = True

    def compile(self):
        """
        Compile the model if not already compiled.
        """
        if not self.is_built:
            raise ValueError("Model must be built before compilation")
        
        if not self.is_compiled:
            print("Learning Rate:", self.config.learning_rate)
            if isinstance(self.config.optimizer, str):
                if self.config.learning_rate is not None:
                    optimizer = tf.keras.optimizers.get({
                        "class_name": self.config.optimizer,
                        "config": {"learning_rate": self.config.learning_rate}
                    })
                else:
                    optimizer = tf.keras.optimizers.get(self.config.optimizer)
                        
            self.model.compile(
                optimizer=optimizer,
                loss=self.config.loss,
                metrics=self.config.metrics
            )
            self.is_compiled = True

    def get_callbacks(self) -> list:
        """
        Get default callbacks for training.
        """
        callbacks = []

        if self.config.early_stopping_patience > 0:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1
                )
            )

        if self.config.reduce_lr_patience > 0:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,
                    patience=self.config.reduce_lr_patience,
                    min_lr=1e-6,
                    verbose=1
                )
            )

        return callbacks
    
    def fit(self, x_train, y_train, x_val=None, y_val=None, verbose=1):
        """
        Train the model.
        """
        if not self.is_compiled:
            raise ValueError("Model must be compiled before training")
        
        validation_data = (x_val, y_val) if x_val is not None and y_val is not None else None

        self.history = self.model.fit(
            x_train, y_train,
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=self.get_callbacks(),
            verbose=verbose
        )
        return self.history

    def predict(self, x, verbose=0):
        """
        Make predictions.
        """
        if not self.is_built:
            raise ValueError("Model must be built before prediction")
        return self.model.predict(x)
    
    def evaluate(self, x, y, verbose=0):
        """
        Evaluate the model.
        """
        if not self.is_built:
            raise ValueError("Model must be built before evaluation")
        return self.model.evaluate(x, y, verbose=verbose)

