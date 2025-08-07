import tensorflow as tf
import numpy as np
from typing import Tuple, List, Optional
from src.models.baseModel import BaseRecommendationModel


class NeuralCollaborativeFiltering(BaseRecommendationModel):
    """Neural Collaborative Filtering model combining GMF and MLP."""

    def __init__(self, num_users: int, num_items: int, **kwargs):
        super().__init__("neural_cf", **kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.gmf_embedding_dim = kwargs.get('gmf_embedding_dim', self.embedding_dim)
        self.mlp_embedding_dim = kwargs.get('mlp_embedding_dim', self.embedding_dim)

    def build_model(self, alpha: float = 0.5) -> tf.keras.Model:
        # Input layers
        user_input = tf.keras.layers.Input(shape=(), name='user_id')
        item_input = tf.keras.layers.Input(shape=(), name='item_id')

        # GMF Component
        gmf_user_embedding = tf.keras.layers.Embedding(
            self.num_users, self.gmf_embedding_dim,
            name='gmf_user_embedding'
        )(user_input)
        gmf_item_embedding = tf.keras.layers.Embedding(
            self.num_items, self.gmf_embedding_dim,
            name='gmf_item_embedding'
        )(item_input)

        gmf_user_vec = tf.keras.layers.Flatten()(gmf_user_embedding)
        gmf_item_vec = tf.keras.layers.Flatten()(gmf_item_embedding)

        # Element-wise product for GMF
        gmf_output = tf.keras.layers.Multiply()([gmf_user_vec, gmf_item_vec])

        # MLP Component
        mlp_user_embedding = tf.keras.layers.Embedding(
            self.num_users, self.mlp_embedding_dim,
            name='mlp_user_embedding'
        )(user_input)
        mlp_item_embedding = tf.keras.layers.Embedding(
            self.num_items, self.mlp_embedding_dim,
            name='mlp_item_embedding'
        )(item_input)

        mlp_user_vec = tf.keras.layers.Flatten()(mlp_user_embedding)
        mlp_item_vec = tf.keras.layers.Flatten()(mlp_item_embedding)

        # Concatenate user and item embeddings
        mlp_vector = tf.keras.layers.Concatenate()([mlp_user_vec, mlp_item_vec])

        # MLP layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            mlp_vector = tf.keras.layers.Dense(
                hidden_dim,
                activation='relu',
                name=f'mlp_layer_{i}'
            )(mlp_vector)
            mlp_vector = tf.keras.layers.Dropout(
                self.dropout_rate,
                name=f'mlp_dropout_{i}'
            )(mlp_vector)

        # Combine GMF and MLP
        combined = tf.keras.layers.Concatenate()([gmf_output, mlp_vector])

        # Final prediction layer
        output = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
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
        positive_interactions = interactions[interactions[:, 2] > 0]

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

        # Prepare inputs
        users = all_interactions[:, 0].astype(np.int32)
        items = all_interactions[:, 1].astype(np.int32)
        ratings = (all_interactions[:, 2] > 0).astype(np.float32)

        return ([users, items], ratings)

    def _generate_negative_samples(self, positive_interactions: np.ndarray,
                                   ratio: int) -> np.ndarray:
        # Create set of positive (user, item) pairs
        positive_pairs = set(
            zip(positive_interactions[:, 0], positive_interactions[:, 1])
        )

        negative_samples = []

        for user_id, item_id, _ in positive_interactions:
            for _ in range(ratio):
                # Sample random item
                neg_item = np.random.randint(0, self.num_items)

                # Ensure it's not a positive interaction
                while (user_id, neg_item) in positive_pairs:
                    neg_item = np.random.randint(0, self.num_items)

                negative_samples.append([user_id, neg_item, 0])

        return np.array(negative_samples)

    def recommend(self, user_id: int, exclude_items: Optional[List[int]] = None,
                  top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        top_k = top_k or self.config.top_k
        exclude_items = exclude_items or []

        # Generate candidate items
        candidate_items = [i for i in range(self.num_items) if i not in exclude_items]

        if not candidate_items:
            return []

        # Prepare input data
        users = np.full(len(candidate_items), user_id, dtype=np.int32)
        items = np.array(candidate_items, dtype=np.int32)

        # Predict scores
        scores = self.model.predict([users, items], verbose=0).flatten()

        # Get top-k recommendations
        top_indices = np.argsort(scores)[-top_k:][::-1]

        recommendations = [
            (candidate_items[idx], float(scores[idx]))
            for idx in top_indices
        ]

        return recommendations

    def get_user_embedding(self, user_id: int, component: str = 'both') -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        embeddings = []

        if component in ['gmf', 'both']:
            gmf_layer = self.model.get_layer('gmf_user_embedding')
            gmf_embedding = gmf_layer(np.array([user_id]))[0].numpy()
            embeddings.append(gmf_embedding)

        if component in ['mlp', 'both']:
            mlp_layer = self.model.get_layer('mlp_user_embedding')
            mlp_embedding = mlp_layer(np.array([user_id]))[0].numpy()
            embeddings.append(mlp_embedding)

        if len(embeddings) == 1:
            return embeddings[0]
        else:
            return np.concatenate(embeddings)

    def get_item_embedding(self, item_id: int, component: str = 'both') -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        embeddings = []

        if component in ['gmf', 'both']:
            gmf_layer = self.model.get_layer('gmf_item_embedding')
            gmf_embedding = gmf_layer(np.array([item_id]))[0].numpy()
            embeddings.append(gmf_embedding)

        if component in ['mlp', 'both']:
            mlp_layer = self.model.get_layer('mlp_item_embedding')
            mlp_embedding = mlp_layer(np.array([item_id]))[0].numpy()
            embeddings.append(mlp_embedding)

        if len(embeddings) == 1:
            return embeddings[0]
        else:
            return np.concatenate(embeddings)

    def compute_similarity(self, user_id1: int, user_id2: int,
                           component: str = 'both') -> float:
        emb1 = self.get_user_embedding(user_id1, component)
        emb2 = self.get_user_embedding(user_id2, component)

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))