"""
Default configuration for StyleGAN watermarking.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union
import ast


@dataclass
class ModelConfig:
    """Model configuration."""
    stylegan2_url: str = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl"
    stylegan2_local_path: str = "ffhq70k-paper256-ada.pkl"
    img_size: int = 256
    image_pixel_set_seed: int = 42
    image_pixel_count: int = 32


@dataclass
class DecoderConfig:
    """Decoder configuration."""
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 2048, 1024, 512, 256])
    activation: str = "gelu"
    dropout_rate: float = 0.3
    num_residual_blocks: int = 3
    use_spectral_norm: bool = True
    use_layer_norm: bool = True
    use_attention: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    total_iterations: int = 100000
    lr: float = 1e-4
    log_interval: int = 1
    checkpoint_interval: int = 10000


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "nccl"
    init_method: str = "env://"


@dataclass
class EvaluateConfig:
    """Configuration for evaluation."""
    def __init__(self):
        # Basic evaluation settings
        self.num_samples: int = 1000
        self.batch_size: int = 16
        self.output_dir: str = "evaluation_results"
        
        # Enable timing logs
        self.enable_timing_logs: bool = True

    def update_from_args(self, args, mode='train'):
        """Update config from command line arguments."""
        if mode == 'evaluate':
            # Update evaluation settings
            if hasattr(args, 'num_samples'):
                self.num_samples = args.num_samples
            if hasattr(args, 'batch_size'):
                self.batch_size = args.batch_size
            if hasattr(args, 'output_dir'):
                self.output_dir = args.output_dir


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
    model: ModelConfig = field(default_factory=ModelConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    output_dir: str = "results"
    checkpoint_path: Optional[str] = None
    seed: Optional[int] = None
    
    def update_from_args(self, args, mode='train'):
        """Update config from command line arguments."""
        # Model configuration
        if hasattr(args, 'stylegan2_url'):
            self.model.stylegan2_url = args.stylegan2_url
        if hasattr(args, 'stylegan2_local_path'):
            self.model.stylegan2_local_path = args.stylegan2_local_path
        if hasattr(args, 'img_size'):
            self.model.img_size = args.img_size
        if hasattr(args, 'image_pixel_set_seed'):
            self.model.image_pixel_set_seed = args.image_pixel_set_seed
        if hasattr(args, 'image_pixel_count'):
            self.model.image_pixel_count = args.image_pixel_count
        
        # Mode-specific configuration
        if mode == 'train':
            if hasattr(args, 'batch_size'):
                self.training.batch_size = args.batch_size
            if hasattr(args, 'total_iterations'):
                self.training.total_iterations = args.total_iterations
            if hasattr(args, 'lr'):
                self.training.lr = args.lr
            if hasattr(args, 'log_interval'):
                self.training.log_interval = args.log_interval
            if hasattr(args, 'checkpoint_interval'):
                self.training.checkpoint_interval = args.checkpoint_interval
        elif mode == 'evaluate':
            # Update evaluation-specific parameters
            if hasattr(args, 'num_samples'):
                self.evaluate.num_samples = args.num_samples
            if hasattr(args, 'batch_size'):
                self.evaluate.batch_size = args.batch_size
                
        elif mode == 'attack':
            # Update attack-specific parameters
            if hasattr(args, 'batch_size'):
                self.attack.batch_size = args.batch_size
            if hasattr(args, 'num_samples'):
                self.attack.num_samples = args.num_samples
            if hasattr(args, 'pgd_alpha'):
                self.attack.pgd_alpha = args.pgd_alpha
            if hasattr(args, 'pgd_steps'):
                self.attack.pgd_steps = args.pgd_steps
            if hasattr(args, 'pgd_epsilon'):
                self.attack.pgd_epsilon = args.pgd_epsilon
            if hasattr(args, 'surrogate_lr'):
                self.attack.surrogate_lr = args.surrogate_lr
            if hasattr(args, 'surrogate_batch_size'):
                self.attack.surrogate_batch_size = args.surrogate_batch_size
            if hasattr(args, 'surrogate_epochs'):
                self.attack.surrogate_epochs = args.surrogate_epochs
            if hasattr(args, 'surrogate_num_samples'):
                self.attack.surrogate_num_samples = args.surrogate_num_samples
            if hasattr(args, 'num_surrogate_models'):
                self.attack.num_surrogate_models = args.num_surrogate_models
            if hasattr(args, 'log_interval'):
                self.attack.log_interval = args.log_interval
            if hasattr(args, 'visualization_samples'):
                self.attack.visualization_samples = args.visualization_samples
        
        # Other configuration
        if hasattr(args, 'output_dir'):
            self.output_dir = args.output_dir
        if hasattr(args, 'checkpoint_path'):
            self.checkpoint_path = args.checkpoint_path
        if hasattr(args, 'seed'):
            self.seed = args.seed


def get_default_config() -> Config:
    """Get default configuration."""
    return Config() 