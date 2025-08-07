"""Base model class for all recommendation models."""

import abc
import os
import tensorflow as tf
import numpy as np
from src.Config.settings import config
from typing import Dict, List, Tuple, Optional, Any


class BaseRecommendationModel(abc.ABC):
    """Abstract base class"""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.is_trained = False
        self.history = None
        self.config = config

        # Model parameters
        self.embedding_dim = kwargs.get('embedding_dim', config.model.embedding_dim)
        self.hidden_dims = kwargs.get('hidden_dims', config.model.hidden_dims)
        self.dropout_rate = kwargs.get('dropout_rate', config.model.dropout_rate)
        self.learning_rate = kwargs.get('learning_rate', config.model.learning_rate)

    @abc.abstractmethod
    def build_model(self, **kwargs) -> tf.keras.Model:
        pass

    @abc.abstractmethod
    def prepare_training_data(self, interactions: np.ndarray, **kwargs) -> Tuple:
        pass

    def compile_model(self, optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
                      loss: Optional[str] = None, metrics: Optional[List[str]] = None):
        if self.model is None:
            raise ValueError("Model must be built before compilation.")

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        if loss is None:
            loss = 'binary_crossentropy'

        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall']

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, train_data: Tuple, validation_data: Optional[Tuple] = None,
              epochs: Optional[int] = None, batch_size: Optional[int] = None,
              callbacks: Optional[List] = None) -> tf.keras.callbacks.History:
        if self.model is None:
            raise ValueError("Model must be built and compiled before training.")

        epochs = epochs or config.model.epochs
        batch_size = batch_size or config.model.batch_size

        if callbacks is None:
            callbacks = self._get_default_callbacks()

        self.history = self.model.fit(
            train_data[0], train_data[1],
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.is_trained = True
        return self.history

    def predict(self, input_data: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")

        batch_size = batch_size or config.model.batch_size
        return self.model.predict(input_data, batch_size=batch_size)

    def recommend(self, user_id: int, item_features: Optional[np.ndarray] = None,
                  top_k: Optional[int] = None, exclude_seen: bool = True) -> List[Tuple[int, float]]:
        top_k = top_k or config.top_k

        # This is a generic implementation - subclasses should override
        if item_features is None:
            raise NotImplementedError("Subclasses must implement recommendation logic")

        scores = self.predict(item_features)
        top_items = np.argsort(scores)[-top_k:][::-1]

        return [(item_id, scores[item_id]) for item_id in top_items]

    def save_model(self, filepath: Optional[str] = None):
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")

        if filepath is None:
            filepath = os.path.join(config.training.model_save_dir, f"{self.name}.h5")

        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")

    def get_embeddings(self, entity_type: str = 'user') -> Optional[np.ndarray]:
        if not self.is_trained:
            return None

        # Look for embedding layers
        for layer in self.model.layers:
            if entity_type in layer.name.lower() and 'embedding' in layer.name.lower():
                return layer.get_weights()[0]

        return None

    def _get_default_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.model.early_stopping_patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(config.training.checkpoint_dir, f"{self.name}_best.h5"),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
        ]

        # Add TensorBoard logging
        log_dir = os.path.join(config.training.log_dir, self.name)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        )

        return callbacks

    def summary(self):
        if self.model is None:
            print("Model not built yet.")
        else:
            self.model.summary()

    def get_config(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'embedding_dim': self.embedding_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'is_trained': self.is_trained
        }