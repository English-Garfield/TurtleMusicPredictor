import tensorflow as tf
import numpy as np
from typing import Tuple, List, Optional
from src.models.baseModel import BaseRecommendationModel


class NeuralCollaborativeFiltering(BaseRecommendationModel):
    def __init__(self, num_users: int, num_items: int, **kwargs):
        super().__init__("neural_cf", **kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.gmf_embedding_dim = kwargs.get('gmf_embedding_dim', self.embedding_dim)
        self.mlp_embedding_dim = kwargs.get('mlp_embedding_dim', self.embedding_dim)
        self.alpha = kwargs.get('alpha', 0.5)  # Weight for combining GMF and MLP

    def build_model(self, alpha: float = None) -> tf.keras.Model:
        if alpha is not None:
            self.alpha = alpha

        # Input layers
        user_input = tf.keras.layers.Input(shape=(), name='user_id', dtype=tf.int32)
        item_input = tf.keras.layers.Input(shape=(), name='item_id', dtype=tf.int32)

        # GMF Component
        gmf_user_embedding = tf.keras.layers.Embedding(
            self.num_users, self.gmf_embedding_dim,
            embeddings_initializer='normal',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
            name='gmf_user_embedding'
        )(user_input)
        gmf_item_embedding = tf.keras.layers.Embedding(
            self.num_items, self.gmf_embedding_dim,
            embeddings_initializer='normal',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
            name='gmf_item_embedding'
        )(item_input)

        gmf_user_vec = tf.keras.layers.Flatten(name='gmf_user_flatten')(gmf_user_embedding)
        gmf_item_vec = tf.keras.layers.Flatten(name='gmf_item_flatten')(gmf_item_embedding)

        # Element-wise product for GMF
        gmf_output = tf.keras.layers.Multiply(name='gmf_multiply')([gmf_user_vec, gmf_item_vec])

        # MLP Component
        mlp_user_embedding = tf.keras.layers.Embedding(
            self.num_users, self.mlp_embedding_dim,
            embeddings_initializer='normal',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
            name='mlp_user_embedding'
        )(user_input)
        mlp_item_embedding = tf.keras.layers.Embedding(
            self.num_items, self.mlp_embedding_dim,
            embeddings_initializer='normal',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
            name='mlp_item_embedding'
        )(item_input)

        mlp_user_vec = tf.keras.layers.Flatten(name='mlp_user_flatten')(mlp_user_embedding)
        mlp_item_vec = tf.keras.layers.Flatten(name='mlp_item_flatten')(mlp_item_embedding)

        # Concatenate user and item embeddings
        mlp_vector = tf.keras.layers.Concatenate(name='mlp_concat')([mlp_user_vec, mlp_item_vec])

        # MLP layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            mlp_vector = tf.keras.layers.Dense(
                hidden_dim,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(1e-6),
                name=f'mlp_layer_{i}'
            )(mlp_vector)
            mlp_vector = tf.keras.layers.BatchNormalization(name=f'mlp_bn_{i}')(mlp_vector)
            mlp_vector = tf.keras.layers.Dropout(
                self.dropout_rate,
                name=f'mlp_dropout_{i}'
            )(mlp_vector)

        # Combine GMF and MLP
        combined = tf.keras.layers.Concatenate(name='final_concat')([gmf_output, mlp_vector])

        # Final prediction layer
        output = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer='lecun_uniform',
            name='prediction'
        )(combined)

        self.model = tf.keras.Model(
            inputs=[user_input, item_input],
            outputs=output,
            name='neural_collaborative_filtering'
        )

        return self.model

    def prepare_training_data(self, interactions: np.ndarray,
                              negative_sampling: bool = True) -> Tuple:
        # Filter positive interactions (assuming rating > 0 means positive)
        if interactions.shape[1] >= 3:  # Has rating column
            positive_interactions = interactions[interactions[:, 2] > 0]
        else:
            # If no rating column, assume all interactions are positive
            positive_interactions = interactions
            # Add a rating column with 1s
            positive_interactions = np.column_stack([positive_interactions, np.ones(len(positive_interactions))])

        if negative_sampling:
            # Generate negative samples
            negative_samples = self._generate_negative_samples(
                positive_interactions,
                self.config.data.negative_sampling_ratio
            )

            # Combine positive and negative samples
            all_interactions = np.vstack([positive_interactions, negative_samples])
        else:
            all_interactions = positive_interactions

        # Shuffle the data
        np.random.shuffle(all_interactions)

        # Prepare input features (user_id, item_id) and labels
        user_ids = all_interactions[:, 0].astype(int)
        item_ids = all_interactions[:, 1].astype(int)
        labels = all_interactions[:, 2].astype(float)

        # Convert to binary labels (1 for positive, 0 for negative)
        labels = (labels > 0).astype(int)

        return [user_ids, item_ids], labels

    def recommend(self, user_id: int, item_candidates: Optional[np.ndarray] = None,
                  top_k: Optional[int] = None, exclude_seen: bool = True,
                  seen_items: Optional[set] = None) -> List[Tuple[int, float]]:
        top_k = top_k or self.config.top_k

        if item_candidates is None:
            # Generate candidates from all items
            item_candidates = np.arange(self.num_items)

        # Exclude seen items if requested
        if exclude_seen and seen_items:
            item_candidates = np.array([item for item in item_candidates if item not in seen_items])

        if len(item_candidates) == 0:
            return []

        # Create user-item pairs for prediction
        user_ids = np.full(len(item_candidates), user_id)

        # Get predictions
        scores = self.predict([user_ids, item_candidates]).flatten()

        # Get top-k items
        if len(scores) < top_k:
            top_k = len(scores)

        top_indices = np.argsort(scores)[-top_k:][::-1]
        recommendations = [(int(item_candidates[i]), float(scores[i])) for i in top_indices]

        return recommendations

    def get_user_embedding(self, user_id: int, component: str = 'both') -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before extracting embeddings.")

        embeddings = []

        if component in ['gmf', 'both']:
            gmf_layer = self.model.get_layer('gmf_user_embedding')
            gmf_embedding = gmf_layer(tf.constant([user_id]))[0].numpy()
            embeddings.append(gmf_embedding)

        if component in ['mlp', 'both']:
            mlp_layer = self.model.get_layer('mlp_user_embedding')
            mlp_embedding = mlp_layer(tf.constant([user_id]))[0].numpy()
            embeddings.append(mlp_embedding)

        if len(embeddings) == 1:
            return embeddings[0]
        else:
            return np.concatenate(embeddings)

    def get_item_embedding(self, item_id: int, component: str = 'both') -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before extracting embeddings.")

        embeddings = []

        if component in ['gmf', 'both']:
            gmf_layer = self.model.get_layer('gmf_item_embedding')
            gmf_embedding = gmf_layer(tf.constant([item_id]))[0].numpy()
            embeddings.append(gmf_embedding)

        if component in ['mlp', 'both']:
            mlp_layer = self.model.get_layer('mlp_item_embedding')
            mlp_embedding = mlp_layer(tf.constant([item_id]))[0].numpy()
            embeddings.append(mlp_embedding)

        if len(embeddings) == 1:
            return embeddings[0]
        else:
            return np.concatenate(embeddings)

    def predict_user_item(self, user_id: int, item_id: int) -> float:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")

        prediction = self.model.predict([np.array([user_id]), np.array([item_id])])
        return float(prediction[0][0])

    def get_similar_users(self, user_id: int, top_k: int = 10, component: str = 'both') -> List[Tuple[int, float]]:
        if not self.is_trained:
            raise ValueError("Model must be trained before finding similar users.")

        target_embedding = self.get_user_embedding(user_id, component)

        similarities = []
        for other_user in range(self.num_users):
            if other_user != user_id:
                other_embedding = self.get_user_embedding(other_user, component)
                # Cosine similarity
                similarity = np.dot(target_embedding, other_embedding) / (
                        np.linalg.norm(target_embedding) * np.linalg.norm(other_embedding)
                )
                similarities.append((other_user, float(similarity)))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_similar_items(self, item_id: int, top_k: int = 10, component: str = 'both') -> List[Tuple[int, float]]:
        if not self.is_trained:
            raise ValueError("Model must be trained before finding similar items.")

        target_embedding = self.get_item_embedding(item_id, component)

        similarities = []
        for other_item in range(self.num_items):
            if other_item != item_id:
                other_embedding = self.get_item_embedding(other_item, component)
                # Cosine similarity
                similarity = np.dot(target_embedding, other_embedding) / (
                        np.linalg.norm(target_embedding) * np.linalg.norm(other_embedding)
                )
                similarities.append((other_item, float(similarity)))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
