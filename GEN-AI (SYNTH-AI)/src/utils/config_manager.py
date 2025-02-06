import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
from functools import lru_cache

@dataclass
class ModelConfig:
    input_dim: int
    hidden_dims: list
    output_dim: int
    dropout_rate: float
    learning_rate: float

@dataclass
class DataConfig:
    raw_data_path: str
    processed_data_path: str
    batch_size: int
    num_workers: int

@dataclass
class TrainingConfig:
    num_epochs: int
    device: str
    checkpoint_dir: str
    save_frequency: int

class ConfigManager:
    def __init__(self, config_path: str = "config/"):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        self.config_cache = {}
        
        # Create config directory if it doesn't exist
        self.config_path.mkdir(parents=True, exist_ok=True)
        
        # Default configurations
        self.default_config = {
            "model": {
                "input_dim": 100,
                "hidden_dims": [256, 512, 1024],
                "output_dim": 50,
                "dropout_rate": 0.3,
                "learning_rate": 0.001
            },
            "data": {
                "raw_data_path": "data/raw",
                "processed_data_path": "data/processed",
                "batch_size": 32,
                "num_workers": 4
            },
            "training": {
                "num_epochs": 100,
                "device": "cuda",
                "checkpoint_dir": "checkpoints/",
                "save_frequency": 10
            },
            "environment": {
                "seed": 42,
                "debug_mode": False,
                "log_level": "INFO"
            }
        }

    @lru_cache(maxsize=32)
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration from file with caching."""
        config_file = self.config_path / f"{config_name}.yaml"
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"Loaded configuration from {config_file}")
            else:
                config = self.default_config.copy()
                self.save_config(config_name, config)
                self.logger.info(f"Created new configuration at {config_file}")
            
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            return self.default_config.copy()

    def save_config(self, config_name: str, config: Dict[str, Any]) -> bool:
        """Save configuration to file."""
        try:
            config_file = self.config_path / f"{config_name}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            self.logger.info(f"Saved configuration to {config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            return False

    def update_config(self, config_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update specific configuration values."""
        config = self.load_config(config_name)
        
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    d[k] = deep_update(d[k], v)
                else:
                    d[k] = v
            return d

        updated_config = deep_update(config, updates)
        self.save_config(config_name, updated_config)
        # Clear the cache for this config
        self.load_config.cache_clear()
        return updated_config

    def get_model_config(self, config_name: str) -> ModelConfig:
        """Get model-specific configuration."""
        config = self.load_config(config_name)
        return ModelConfig(**config['model'])

    def get_data_config(self, config_name: str) -> DataConfig:
        """Get data-specific configuration."""
        config = self.load_config(config_name)
        return DataConfig(**config['data'])

    def get_training_config(self, config_name: str) -> TrainingConfig:
        """Get training-specific configuration."""
        config = self.load_config(config_name)
        return TrainingConfig(**config['training'])

    def export_config(self, config_name: str, export_path: str) -> bool:
        """Export configuration to JSON format."""
        try:
            config = self.load_config(config_name)
            export_file = Path(export_path) / f"{config_name}_config.json"
            
            with open(export_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            self.logger.info(f"Exported configuration to {export_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {str(e)}")
            return False

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure and values."""
        required_sections = ['model', 'data', 'training', 'environment']
        
        try:
            # Check for required sections
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section: {section}")
            
            # Validate model config
            model_config = config['model']
            assert isinstance(model_config['input_dim'], int)
            assert isinstance(model_config['hidden_dims'], list)
            assert isinstance(model_config['output_dim'], int)
            assert 0 <= model_config['dropout_rate'] <= 1
            
            # Validate data config
            assert os.path.exists(config['data']['raw_data_path'])
            assert config['data']['batch_size'] > 0
            
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False