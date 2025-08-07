import abc
import os
import tensorflow as tf
import numpy as np
from src.Config.settings import config
from typing import Dict, List, Tuple, Optional, Any
import joblib
import json


class BaseRecommendationModel(abc.ABC):
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

        epochs = epochs or self.config.model.epochs
        batch_size = batch_size or self.config.model.batch_size

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

        batch_size = batch_size or self.config.model.batch_size
        return self.model.predict(input_data, batch_size=batch_size)

    def recommend(self, user_id: int, item_candidates: Optional[np.ndarray] = None,
                  top_k: Optional[int] = None, exclude_seen: bool = True) -> List[Tuple[int, float]]:
        top_k = top_k or self.config.top_k

        if item_candidates is None:
            raise NotImplementedError("Subclasses must implement recommendation logic")

        # Create user-item pairs for prediction
        user_items = np.column_stack([
            np.full(len(item_candidates), user_id),
            item_candidates
        ])

        # Get predictions
        scores = self.predict(user_items).flatten()

        # Get top-k items
        top_indices = np.argsort(scores)[-top_k:][::-1]
        recommendations = [(item_candidates[i], scores[i]) for i in top_indices]

        return recommendations

    def save_model(self, filepath: str):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")

        # Create the dir if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save the Keras model
        self.model.save(filepath)

        # Save additional metadata
        metadata = {
            'name': self.name,
            'embedding_dim': self.embedding_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'is_trained': self.is_trained
        }

        metadata_path = filepath.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load the Keras model
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True

        # Load metadata if available
        metadata_path = filepath.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.name = metadata.get('name', self.name)
            self.embedding_dim = metadata.get('embedding_dim', self.embedding_dim)
            self.hidden_dims = metadata.get('hidden_dims', self.hidden_dims)
            self.dropout_rate = metadata.get('dropout_rate', self.dropout_rate)
            self.learning_rate = metadata.get('learning_rate', self.learning_rate)

        print(f"Model loaded from {filepath}")

    def _get_default_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        callbacks = []

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.model.early_stopping_patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)

        # Model checkpoint
        checkpoint_path = os.path.join(
            self.config.training.checkpoint_dir,
            f'{self.name}_best_model.h5'
        )
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        )
        callbacks.append(model_checkpoint)

        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        callbacks.append(reduce_lr)

        # TensorBoard logging
        log_dir = os.path.join(self.config.training.log_dir, self.name)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        callbacks.append(tensorboard)

        return callbacks

    def _generate_negative_samples(self, positive_interactions: np.ndarray,
                                   negative_ratio: int = 4) -> np.ndarray:
        users = positive_interactions[:, 0]
        items = positive_interactions[:, 1]

        # Get unique users and items
        unique_users = np.unique(users)
        unique_items = np.unique(items)

        # Create the positive interaction set for fast lookup
        positive_set = set(zip(users, items))

        negative_samples = []

        for user in unique_users:
            # Get the number of positive interactions for this user
            user_positives = np.sum(users == user)
            num_negatives = user_positives * negative_ratio

            # Generate negative samples for this user
            user_negatives = 0
            attempts = 0
            max_attempts = num_negatives * 10  # Prevent infinite loop

            while user_negatives < num_negatives and attempts < max_attempts:
                # Randomly sample items
                negative_items = np.random.choice(unique_items, size=num_negatives - user_negatives)

                for item in negative_items:
                    if (user, item) not in positive_set:
                        negative_samples.append([user, item, 0])
                        user_negatives += 1
                        if user_negatives >= num_negatives:
                            break

                attempts += 1

        return np.array(negative_samples)

    def get_model_summary(self) -> str:
        if self.model is None:
            return "Model not built yet."

        # Capture model summary
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)

    def get_embeddings(self, entity_type: str = 'user') -> Optional[np.ndarray]:
        if not self.is_trained:
            raise ValueError("Model must be trained before extracting embeddings.")

        # This is a generic implementation - subclasses should override
        # to provide specific embedding extraction logic
        layer_name = f'{entity_type}_embedding'

        try:
            embedding_layer = self.model.get_layer(layer_name)
            return embedding_layer.get_weights()[0]
        except ValueError:
            print(f"Layer '{layer_name}' not found in model.")
            return None

    def evaluate_model(self, test_data: Tuple) -> Dict[str, float]:
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")

        results = self.model.evaluate(test_data[0], test_data[1], verbose=0)

        # Create a dictionary of metric names and values
        metric_names = self.model.metrics_names
        return dict(zip(metric_names, results))
