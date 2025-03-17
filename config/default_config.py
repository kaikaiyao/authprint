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
    selected_indices: Optional[Union[List[int], str]] = None
    key_mapper_seed: Optional[int] = None  # Specific seed for KeyMapper initialization
    
    # Latent-based watermarking configuration
    w_partial_set_seed: int = 42  # Seed for selecting random latent indices
    w_partial_length: int = 32  # Number of dimensions to select from the latent vector
    
    # Image-based watermarking configuration
    use_image_pixels: bool = False  # Whether to use image pixels instead of latent vectors
    image_pixel_set_seed: int = 42  # Seed for selecting random pixels
    image_pixel_count: int = 8192  # Number of pixels to select from the image


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
    
    # Negative sample evaluation options
    evaluate_neg_samples: bool = True
    
    # Pre-trained model options
    evaluate_pretrained: bool = True
    evaluate_ffhq1k: bool = True
    evaluate_ffhq30k: bool = True
    evaluate_ffhq70k_bcr: bool = True
    evaluate_ffhq70k_noaug: bool = True
    
    # Image transformation options
    evaluate_transforms: bool = True
    evaluate_truncation: bool = True
    truncation_psi: float = 2.0
    evaluate_quantization: bool = True
    evaluate_downsample: bool = True
    downsample_size: int = 128
    evaluate_jpeg: bool = False
    jpeg_quality: int = 55


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

    def _update_subconfig(self, config_obj, args_dict):
        """Update a specific sub-configuration with matching arguments."""
        for key, value in args_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
    
    def update_from_args(self, args, mode=None):
        """Update config from command line arguments.
        
        Args:
            args: Parsed command line arguments
            mode: The mode we're running in ('train', 'attack', 'evaluate'). 
                 If None, will try to infer from the arguments.
        """
        args_dict = vars(args)
        
        # Update main config first
        self._update_subconfig(self, args_dict)
        
        # Always update model config as it's shared across modes
        self._update_subconfig(self.model, args_dict)
        
        # Update specific configs based on mode
        if mode == 'train' or ('total_iterations' in args_dict):
            self._update_subconfig(self.training, args_dict)
        
        if mode == 'attack' or ('pgd_alpha' in args_dict):
            self._update_subconfig(self.attack, args_dict)
        
        # Handle evaluate mode with special case for negative samples options
        evaluate_mode = (mode == 'evaluate' or 'evaluation_mode' in args_dict)
        if evaluate_mode:
            # Update basic evaluate config
            self._update_subconfig(self.evaluate, args_dict)
            
            # Explicitly check for negative sample evaluation options
            evaluate_options = [
                'evaluate_neg_samples', 'evaluate_pretrained', 
                'evaluate_ffhq1k', 'evaluate_ffhq30k', 'evaluate_ffhq70k_bcr', 'evaluate_ffhq70k_noaug',
                'evaluate_transforms', 'evaluate_truncation', 'truncation_psi',
                'evaluate_quantization', 'evaluate_downsample', 'downsample_size',
                'evaluate_jpeg', 'jpeg_quality'
            ]
            
            for option in evaluate_options:
                if option in args_dict:
                    setattr(self.evaluate, option, args_dict[option])
        
        # These are shared across modes
        self._update_subconfig(self.decoder, args_dict)
        self._update_subconfig(self.distributed, args_dict)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config() 