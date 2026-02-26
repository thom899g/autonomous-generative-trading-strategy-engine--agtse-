"""
Configuration management for AGTSE.
Centralizes all configurable parameters with validation.
"""
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available generative model types"""
    TRANSFORMER = "transformer"
    GAN = "gan"
    LSTM = "lstm"
    VAE = "vae"


class MarketEnvType(Enum):
    """Available market environment types"""
    HISTORICAL_SIMULATION = "historical"
    REINFORCEMENT = "rl"
    HYBRID = "hybrid"


@dataclass
class DatabaseConfig:
    """Firebase database configuration"""
    project_id: str = os.getenv("FIREBASE_PROJECT_ID", "agtse-default")
    credentials_path: str = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_credentials.json")
    firestore_collection: str = "trading_strategies"
    realtime_db_url: Optional[str] = os.getenv("FIREBASE_REALTIME_DB_URL")
    
    def validate(self) -> bool:
        """Validate database configuration"""
        if not os.path.exists(self.credentials_path):
            logger.error(f"Firebase credentials file not found: {self.credentials_path}")
            return False
        return True


@dataclass
class ModelConfig:
    """Generative model configuration"""
    model_type: ModelType = ModelType.TRANSFORMER
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout_rate: float = 0.1
    sequence_length: int = 100
    feature_dim: int = 20  # OHLCV + indicators
    batch_size: int = 32
    learning_rate: float = 1e-4
    checkpoint_dir: str = "checkpoints"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters as dict"""
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "sequence_length": self.sequence_length,
            "feature_dim": self.feature_dim
        }


@dataclass
class SimulationConfig:
    """Market simulation configuration"""
    env_type: MarketEnvType = MarketEnvType.HISTORICAL_SIMULATION
    initial_balance: float = 10000.0
    transaction_cost: float = 0.001  # 0.1%
    max_position_size: float = 0.1  # 10% of portfolio
    risk_free_rate: float = 0.02  # 2% annual
    data_source: str = "binance"  # ccxt-compatible exchange
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframe: str = "1h"
    lookback_window: int = 1000
    validation_split: float = 0.2
    test_split: float = 0.1


@dataclass
class RLConfig:
    """Reinforcement learning configuration"""
    algorithm: str = "PPO"  # PPO, DQN, A2C
    gamma: float = 0.99  # Discount factor
    entropy_coeff: float = 0.01  # Entropy regularization
    learning_rate: float = 3e-4
    num_epochs: int = 10
    clip_param: float = 0.2
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    rollout_length: int = 2048
    num_workers: int = 4


@dataclass
class AGTSEConfig:
    """Main AGTSE configuration"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    
    # System settings
    log_level: str = "INFO"
    results_dir: str = "results"
    max_strategies: int = 1000
    evolution_iterations: int = 100
    parallel_simulations: int = 8
    
    @classmethod
    def from_file(cls, config_path: str) -> "AGTSEConfig":
        """Load configuration from JSON file"""
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return cls._from_dict(config_dict)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Error loading config: {e}")
            raise ValueError(f"Invalid configuration file: {e}")
    
    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "AGTSEConfig":
        """Create config from dictionary"""
        # Convert string enums to Enum objects
        if "model" in config_dict:
            if "model_type" in config_dict["model"]:
                config_dict["model"]["model_type"] = ModelType(config_dict["model"]["model_type"])
        
        if "simulation" in config_dict:
            if "env_type" in config_dict["simulation"]:
                config_dict["simulation"]["env_type"] = MarketEnvType(
                    config_dict["simulation"]["env_type"]
                )
        
        return cls(
            database=DatabaseConfig(**config_dict.get("database", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            simulation=SimulationConfig(**config_dict.get("simulation", {})),
            rl=RLConfig(**config_dict.get("rl", {})),
            log_level=config_dict.get("log_level", "INFO"),
            results_dir=config_dict.get("results_dir", "results"),
            max_strategies=config_dict.get("max_strategies", 1000),
            evolution_iterations=config_dict.get("evolution_iterations", 100),
            parallel_simulations=config_dict.get("parallel_simulations", 8)
        )
    
    def validate(self) -> bool:
        """Validate entire configuration"""
        try:
            # Validate database config
            if not self.database.validate():
                return False
            
            # Validate paths
            os.makedirs(self.results_dir, exist_ok=True)
            os.makedirs(self.model.checkpoint_dir, exist_ok=True)
            
            # Validate numerical ranges
            if