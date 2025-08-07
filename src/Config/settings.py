"""Config Settings"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AudioConfig:
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 13
    segment_length: int = 30  # seconds


@dataclass
class ModelConfig:
    embedding_dim: int = 64
    hidden_dims: List[int] = None
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 100
    early_stopping_patience: int = 10

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]


@dataclass
class DataConfig:
    min_interactions: int = 5
    test_split: float = 0.2
    validation_split: float = 0.1
    negative_sampling_ratio: int = 4
    max_sequence_length: int = 50


@dataclass
class APIConfig:
    spotify_client_id: str = os.getenv('SPOTIFY_CLIENT_ID', '')
    spotify_client_secret: str = os.getenv('SPOTIFY_CLIENT_SECRET', '')
    lastfm_api_key: str = os.getenv('LASTFM_API_KEY', '')
    lastfm_api_secret: str = os.getenv('LASTFM_API_SECRET', '')


@dataclass
class TrainingConfig:
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    model_save_dir: str = 'saved_models'
    data_dir: str = 'data'
    use_mixed_precision: bool = True
    distribute_strategy: Optional[str] = None  # 'mirrored', 'multi_worker_mirrored'


class Config:

    def __init__(self):
        self.audio = AudioConfig()
        self.model = ModelConfig()
        self.data = DataConfig()
        self.api = APIConfig()
        self.training = TrainingConfig()

        # Recommendation parameters
        self.top_k: int = 50
        self.diversity_weight: float = 0.1
        self.novelty_weight: float = 0.05
        self.context_weight: float = 0.2

        # Evaluation parameters
        self.eval_metrics: List[str] = [
            'precision', 'recall', 'ndcg', 'diversity',
            'novelty', 'coverage', 'popularity_bias'
        ]

        # Create directories
        self._create_directories()

    def _create_directories(self):
        directories = [
            self.training.checkpoint_dir,
            self.training.log_dir,
            self.training.model_save_dir,
            self.training.data_dir
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def update_config(self, config_dict: Dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), object) and hasattr(getattr(self, key), '__dict__'):
                    # Update nested config
                    nested_config = getattr(self, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(self, key, value)


config = Config()