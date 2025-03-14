"""
Default configuration for StyleGAN watermarking.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class ModelConfig:
    """Configuration for models."""
    # StyleGAN2 configuration
    stylegan2_url: str = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl"
    stylegan2_local_path: str = "ffhq70k-paper256-ada.pkl"
    img_size: int = 256
    
    # Watermarking configuration
    key_length: int = 4
    selected_indices: List[int] = field(default_factory=lambda: list(range(32)))
    key_mapper_seed: Optional[int] = None  # Specific seed for KeyMapper initialization
    

@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 16
    total_iterations: int = 100000
    lr: float = 1e-4
    lambda_lpips: float = 1.0
    log_interval: int = 1
    checkpoint_interval: int = 10000


@dataclass
class DecoderConfig:
    """Configuration for decoder model."""
    image_size: int = 256
    channels: int = 3
    output_dim: int = 4


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "nccl"
    init_method: str = "env://"


@dataclass
class EvaluateConfig:
    """Configuration for evaluation."""
    batch_size: int = 16
    num_samples: int = 1000
    num_vis_samples: int = 10
    evaluation_mode: str = 'both'


@dataclass
class AttackConfig:
    """Configuration for attacks against the watermarking."""
    # General attack settings
    batch_size: int = 16
    num_samples: int = 100
    
    # PGD attack parameters
    pgd_alpha: float = 0.01  # Step size
    pgd_steps: int = 100     # Number of PGD iterations
    pgd_epsilon: float = 1.0  # Maximum perturbation
    
    # Surrogate training parameters
    surrogate_lr: float = 1e-4
    surrogate_batch_size: int = 32
    surrogate_epochs: int = 1
    surrogate_num_samples: int = 10000
    num_surrogate_models: int = 5
    
    # Attack evaluation
    log_interval: int = 10
    visualization_samples: int = 5


@dataclass
class Config:
    """Main configuration class."""
    # Output path
    output_dir: str = "results"
    
    # Checkpoint path for resuming training
    checkpoint_path: Optional[str] = None
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    
    # Meta configuration
    seed: Optional[int] = None
    
    def update_from_args(self, args):
        """Update config from command line arguments."""
        for key, value in vars(args).items():
            # First check attack config since we're in attack mode
            if hasattr(self.attack, key):
                setattr(self.attack, key, value)
            elif hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.model, key):
                setattr(self.model, key, value)
            elif hasattr(self.training, key):
                setattr(self.training, key, value)
            elif hasattr(self.decoder, key):
                setattr(self.decoder, key, value)
            elif hasattr(self.distributed, key):
                setattr(self.distributed, key, value)
            elif hasattr(self.evaluate, key):
                setattr(self.evaluate, key, value)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config() 