import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional


class CollaborativeFiltering:
    def __init__(self, ratings: pd.DataFrame, k: int = 10):
        """
        Initialize collaborative filtering model.

        Args:
            ratings: User-item ratings matrix (users as rows, items as columns)
            k: Number of similar users/items to consider
        """
        if ratings.empty:
            raise ValueError("Ratings matrix cannot be empty")
        if k <= 0:
            raise ValueError("k must be positive")

        self.ratings = ratings.fillna(0)
        self.k = min(k, len(ratings) - 1)  # Ensure k doesn't exceed available users
        self.normalized_ratings = self._normalize_ratings()

        # User-based similarity
        self.user_similarity = cosine_similarity(self.normalized_ratings)
        self.user_sim_df = pd.DataFrame(
            self.user_similarity,
            index=ratings.index,
            columns=ratings.index
        )

        # Item-based similarity
        self.item_similarity = cosine_similarity(self.ratings.T)
        self.item_sim_df = pd.DataFrame(
            self.item_similarity,
            index=self.ratings.columns,
            columns=self.ratings.columns
        )

    def _normalize_ratings(self) -> pd.DataFrame:
        """Normalize ratings by subtracting user mean."""
        user_means = self.ratings.mean(axis=1)
        return self.ratings.sub(user_means, axis=0).fillna(0)

    def recommend(self, user_id, n: int = 5) -> List[str]:
        """
        User-based collaborative filtering recommendations.

        Args:
            user_id: ID of user to recommend for
            n: Number of recommendations to return

        Returns:
            List of recommended item IDs
        """
        if user_id not in self.ratings.index:
            raise ValueError(f"User {user_id} not found in ratings matrix")

        user_ratings = self.ratings.loc[user_id]
        similar_users = self.user_sim_df[user_id].drop(user_id).nlargest(self.k)

        if similar_users.empty or similar_users.sum() == 0:
            # Fallback to popular items
            return self._get_popular_items(user_ratings, n)

        # Calculate weighted ratings
        similar_user_ratings = self.ratings.loc[similar_users.index]
        weighted_ratings = similar_user_ratings.T.dot(similar_users) / similar_users.sum()

        # Filter out items user has already rated
        unrated_items = user_ratings == 0
        recommendations = weighted_ratings[unrated_items].sort_values(ascending=False)

        return recommendations.head(n).index.tolist()

    def item_based_recommend(self, user_id, n: int = 5) -> List[str]:
        """
        Item-based collaborative filtering recommendations.

        Args:
            user_id: ID of user to recommend for
            n: Number of recommendations to return

        Returns:
            List of recommended item IDs
        """
        if user_id not in self.ratings.index:
            raise ValueError(f"User {user_id} not found in ratings matrix")

        user_ratings = self.ratings.loc[user_id]

        # Calculate weighted ratings based on item similarity
        weighted_ratings = self.item_sim_df.dot(user_ratings)
        item_sim_sums = self.item_sim_df.sum(axis=1)

        # Avoid division by zero
        weighted_ratings = weighted_ratings.div(item_sim_sums.replace(0, np.nan))

        # Filter out items user has already rated
        unrated_items = user_ratings == 0
        recommendations = weighted_ratings[unrated_items].sort_values(ascending=False)

        return recommendations.head(n).index.tolist()

    def _get_popular_items(self, user_ratings: pd.Series, n: int) -> List[str]:
        """Fallback to most popular unrated items."""
        unrated_items = user_ratings == 0
        item_popularity = self.ratings.sum(axis=0)
        popular_unrated = item_popularity[unrated_items].sort_values(ascending=False)
        return popular_unrated.head(n).index.tolist()

    def get_user_similarity(self, user1_id, user2_id) -> float:
        """Get similarity score between two users."""
        if user1_id not in self.ratings.index or user2_id not in self.ratings.index:
            raise ValueError("One or both users not found")
        return self.user_sim_df.loc[user1_id, user2_id]

    def get_item_similarity(self, item1_id, item2_id) -> float:
        """Get similarity score between two items."""
        if item1_id not in self.ratings.columns or item2_id not in self.ratings.columns:
            raise ValueError("One or both items not found")
        return self.item_sim_df.loc[item1_id, item2_id]