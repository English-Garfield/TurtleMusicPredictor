import os
import sys
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Add src to the path for the imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Config.settings import config
from data.dataLoader import DataLoader
from models.neuralCF import NeuralCollaborativeFiltering
from models.matrixFactorization import MatrixFactorization
from models.deepLearning import DeepMusicModel
from evaluation.metrics import RecommendationEvaluator
from utils.preprocessor import AudioPreprocessor
from api.spotifyAPI import SpotifyAPI
from recommendation.engine import RecommendationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TurtleMusicPredictor:

    def __init__(self):
        self.config = config
        self.data_loader = None
        self.models = {}
        self.recommendation_engine = None
        self.evaluator = RecommendationEvaluator()
        self.audio_processor = AudioPreprocessor()
        self.spotify_api = SpotifyAPI()

    def initialize(self):
        logger.info("Initializing TurtleMusicPredictor...")

        # Initialize data loader
        self.data_loader = DataLoader(self.config.training.data_dir)

        # Initialize recommendation engine
        self.recommendation_engine = RecommendationEngine()

        logger.info("Initialization complete!")

    def load_data(self, data_path: str = None) -> Dict[str, Any]:
        logger.info("Loading training data...")

        if data_path is None:
            data_path = self.config.training.data_dir

        # Load interaction data
        interactions = self.data_loader.load_interactions(data_path)

        # Load audio features if available
        audio_features = self.data_loader.load_audio_features(data_path)

        # Load metadata
        metadata = self.data_loader.load_metadata(data_path)

        return {
            'interactions': interactions,
            'audio_features': audio_features,
            'metadata': metadata
        }

    def train_models(self, data: Dict[str, Any]):
        logger.info("Training recommendation models...")

        interactions = data['interactions']
        audio_features = data.get('audio_features')

        # Get dimensions
        num_users = len(interactions['user_id'].unique())
        num_items = len(interactions['item_id'].unique())

        # Train Neural Collaborative Filtering
        logger.info("Training Neural Collaborative Filtering model...")
        ncf_model = NeuralCollaborativeFiltering(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=self.config.model.embedding_dim
        )
        ncf_model.build_model()
        ncf_model.compile_model()

        # Prepare training data
        train_data = ncf_model.prepare_training_data(interactions.values)

        # Train the model
        ncf_model.train(train_data)
        self.models['neural_cf'] = ncf_model

        # Train Matrix Factorization
        logger.info("Training Matrix Factorization model...")
        mf_model = MatrixFactorization(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=self.config.model.embedding_dim
        )
        mf_model.build_model()
        mf_model.compile_model()

        mf_train_data = mf_model.prepare_training_data(interactions.values)
        mf_model.train(mf_train_data)
        self.models['matrix_factorization'] = mf_model

        # Train Deep Learning model if audio features available
        if audio_features is not None:
            logger.info("Training Deep Learning model...")
            dl_model = DeepMusicModel(
                num_users=num_users,
                num_items=num_items,
                audio_feature_dim=audio_features.shape[1]
            )
            dl_model.build_model()
            dl_model.compile_model()

            dl_train_data = dl_model.prepare_training_data(
                interactions.values, audio_features
            )
            dl_model.train(dl_train_data)
            self.models['deep_learning'] = dl_model

        logger.info("Model training complete!")

    def evaluate_models(self, test_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        logger.info("Evaluating models...")

        results = {}
        interactions = test_data['interactions']

        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")

            # Generate predictions
            predictions = model.predict(interactions[['user_id', 'item_id']].values)

            # Calculate metrics
            metrics = self.evaluator.evaluate(
                interactions.values,
                predictions,
                k=self.config.top_k
            )

            results[model_name] = metrics
            logger.info(f"{model_name} metrics: {metrics}")

        return results

    def get_recommendations(self, user_id: int, top_k: int = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.config.top_k

        if not self.models:
            raise ValueError("No models trained. Please train models first.")

        # Use the recommendation engine to combine multiple models
        recommendations = self.recommendation_engine.recommend(
            user_id=user_id,
            models=self.models,
            top_k=top_k
        )

        # Enrich with Spotify metadata if available
        enriched_recommendations = []
        for rec in recommendations:
            track_info = self.spotify_api.get_track_info(rec['item_id'])
            enriched_rec = {
                **rec,
                'track_info': track_info
            }
            enriched_recommendations.append(enriched_rec)

        return enriched_recommendations

    def save_models(self):
        logger.info("Saving models...")

        for model_name, model in self.models.items():
            save_path = os.path.join(
                self.config.training.model_save_dir,
                f"{model_name}.h5"
            )
            model.save_model(save_path)
            logger.info(f"Saved {model_name} to {save_path}")

    def load_models(self):
        logger.info("Loading models...")

        model_dir = self.config.training.model_save_dir
        for filename in os.listdir(model_dir):
            if filename.endswith('.h5'):
                model_name = filename[:-3]  # Remove .h5 extension
                model_path = os.path.join(model_dir, filename)

                # Load based on the model type
                if model_name == 'neural_cf':
                    model = NeuralCollaborativeFiltering(0, 0)  # Will be updated when loading
                elif model_name == 'matrix_factorization':
                    model = MatrixFactorization(0, 0)
                else:
                    continue

                model.load_model(model_path)
                self.models[model_name] = model
                logger.info(f"Loaded {model_name} from {model_path}")


def main():
    """Main function to run the music recommendation system."""
    print("🐢 Welcome to TurtleMusicPredictor! 🎵")

    # Initialize the system
    predictor = TurtleMusicPredictor()
    predictor.initialize()

    # Check if we have pre-trained models
    model_dir = config.training.model_save_dir
    if os.path.exists(model_dir) and os.listdir(model_dir):
        print("Found existing models. Loading...")
        predictor.load_models()
    else:
        print("No existing models found. Training new models...")

        # Load data
        data = predictor.load_data()

        if not data['interactions'].empty:
            # Train models
            predictor.train_models(data)

            # Save models
            predictor.save_models()

            # Evaluate models
            evaluation_results = predictor.evaluate_models(data)
            print("Evaluation Results:")
            for model_name, metrics in evaluation_results.items():
                print(f"{model_name}: {metrics}")
        else:
            print("No interaction data found. Please add training data to continue.")
            return

    # Interactive mode
    while True:
        print("\n" + "=" * 50)
        print("What would you like to do?")
        print("1. Get recommendations for a user")
        print("2. Train models with new data")
        print("3. Evaluate models")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            try:
                user_id = int(input("Enter user ID: "))
                top_k = int(input(f"Number of recommendations (default {config.top_k}): ") or config.top_k)

                recommendations = predictor.get_recommendations(user_id, top_k)

                print(f"\nTop {len(recommendations)} recommendations for user {user_id}:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. Item {rec['item_id']} (Score: {rec['score']:.3f})")
                    if 'track_info' in rec and rec['track_info']:
                        track = rec['track_info']
                        print(f"   🎵 {track.get('name', 'Unknown')} by {track.get('artist', 'Unknown Artist')}")

            except ValueError:
                print("Invalid user ID. Please enter a number.")
            except Exception as e:
                print(f"Error getting recommendations: {e}")

        elif choice == '2':
            data_path = input("Enter path to new training data (press Enter for default): ").strip()
            if not data_path:
                data_path = None

            try:
                data = predictor.load_data(data_path)
                predictor.train_models(data)
                predictor.save_models()
                print("Models retrained successfully!")
            except Exception as e:
                print(f"Error training models: {e}")

        elif choice == '3':
            try:
                data = predictor.load_data()
                results = predictor.evaluate_models(data)
                print("\nEvaluation Results:")
                for model_name, metrics in results.items():
                    print(f"\n{model_name.upper()}:")
                    for metric, value in metrics.items():
                        print(f"  {metric}: {value:.4f}")
            except Exception as e:
                print(f"Error evaluating models: {e}")

        elif choice == '4':
            print("Thanks for using TurtleMusicPredictor! 🐢🎵")
            break

        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == '__main__':
    main()