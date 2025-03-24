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
    key_mapper_use_sine: bool = False  # Whether to use sine-based mapping in KeyMapper (more sensitive to input changes)
    key_mapper_sensitivity: float = 20.0  # Sensitivity parameter for sine-based mapping
    
    # Latent-based watermarking configuration
    w_partial_set_seed: int = 42  # Seed for selecting random latent indices
    w_partial_length: int = 32  # Number of dimensions to select from the latent vector
    
    # Image-based watermarking configuration
    use_image_pixels: bool = False  # Whether to use image pixels instead of latent vectors
    image_pixel_set_seed: int = 42  # Seed for selecting random pixels
    image_pixel_count: int = 8192  # Number of pixels to select from the image
    
    # Direct feature decoder mode
    direct_feature_decoder: bool = False  # Whether to train decoder directly on features instead of images


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 16
    total_iterations: int = 100000
    lr: float = 1e-4
    lambda_lpips: float = 1.0
    log_interval: int = 1
    checkpoint_interval: int = 10000
    freeze_watermarked_model: bool = False


@dataclass
class DecoderConfig:
    """Configuration for decoder model."""
    image_size: int = 256
    channels: int = 3
    output_dim: int = 4
    
    # Enhanced FeatureDecoder parameters
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 2048, 1024, 512, 256])
    activation: str = 'gelu'  # Options: 'leaky_relu', 'relu', 'gelu', 'swish', 'mish'
    dropout_rate: float = 0.3
    num_residual_blocks: int = 3
    use_spectral_norm: bool = True
    use_layer_norm: bool = True
    use_attention: bool = True


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
    evaluation_mode: str = 'batch'  # Changed from 'both' to 'batch' as default
    
    # Process monitoring options
    enable_timing_logs: bool = True  # Enable detailed timing logs for process monitoring
    
    # Visualization options
    enable_visualization: bool = False  # Master switch for all visualizations
    save_comparisons: bool = False  # Whether to save comparison visualizations
    visualization_seed: int = 42  # Random seed for consistent visualization samples
    verbose_visualization: bool = False  # Whether to log detailed per-sample information
    
    # Visualization flags for specific model types
    visualize_pretrained: bool = False  # Master switch for pretrained model visualizations
    visualize_transforms: bool = False  # Master switch for transformation visualizations
    visualize_ffhq1k: bool = False
    visualize_ffhq30k: bool = False
    visualize_ffhq70k_bcr: bool = False
    visualize_ffhq70k_noaug: bool = False
    
    # Visualization flags for specific transformations
    visualize_truncation: bool = False
    visualize_truncation_watermarked: bool = False
    visualize_quantization: bool = False
    visualize_quantization_watermarked: bool = False
    visualize_quantization_int4: bool = False
    visualize_quantization_int4_watermarked: bool = False
    visualize_quantization_int2: bool = False
    visualize_quantization_int2_watermarked: bool = False
    visualize_downsample: bool = False
    visualize_downsample_watermarked: bool = False
    visualize_jpeg: bool = False
    visualize_jpeg_watermarked: bool = False
    
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
    
    # Truncation options
    evaluate_truncation: bool = True
    evaluate_truncation_watermarked: bool = True
    truncation_psi: float = 2.0
    
    # Int8 quantization options (original)
    evaluate_quantization: bool = True
    evaluate_quantization_watermarked: bool = True
    
    # Int4 quantization options
    evaluate_quantization_int4: bool = True
    evaluate_quantization_int4_watermarked: bool = True
    
    # Int2 quantization options
    evaluate_quantization_int2: bool = True
    evaluate_quantization_int2_watermarked: bool = True
    
    # Downsampling options
    evaluate_downsample: bool = True
    evaluate_downsample_watermarked: bool = True
    downsample_size: int = 128
    
    # JPEG compression options
    evaluate_jpeg: bool = False
    evaluate_jpeg_watermarked: bool = False
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
    use_combined_surrogate_input: bool = False  # Whether to use both images and w_partial as input to surrogate decoders
    
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
                'evaluate_jpeg', 'jpeg_quality', 'visualization_seed', 'verbose_visualization',
                # Add new visualization options
                'enable_visualization', 'save_comparisons', 'visualize_pretrained', 'visualize_transforms',
                'visualize_ffhq1k', 'visualize_ffhq30k', 'visualize_ffhq70k_bcr', 'visualize_ffhq70k_noaug',
                'visualize_truncation', 'visualize_truncation_watermarked',
                'visualize_quantization', 'visualize_quantization_watermarked',
                'visualize_quantization_int4', 'visualize_quantization_int4_watermarked',
                'visualize_quantization_int2', 'visualize_quantization_int2_watermarked',
                'visualize_downsample', 'visualize_downsample_watermarked',
                'visualize_jpeg', 'visualize_jpeg_watermarked'
            ]
            
            for option in evaluate_options:
                if option in args_dict:
                    setattr(self.evaluate, option, args_dict[option])
        
        # Handle decoder special parameters
        if 'decoder_hidden_dims' in args_dict and args_dict['decoder_hidden_dims'] is not None:
            # Convert comma-separated string to list of ints
            self.decoder.hidden_dims = [int(dim) for dim in args_dict['decoder_hidden_dims'].split(',')]
        
        if 'decoder_no_spectral_norm' in args_dict:
            self.decoder.use_spectral_norm = not args_dict['decoder_no_spectral_norm']
            
        if 'decoder_no_layer_norm' in args_dict:
            self.decoder.use_layer_norm = not args_dict['decoder_no_layer_norm']
            
        if 'decoder_no_attention' in args_dict:
            self.decoder.use_attention = not args_dict['decoder_no_attention']
        
        # Update other decoder parameters
        decoder_params = {
            'activation': 'decoder_activation',
            'dropout_rate': 'decoder_dropout_rate',
            'num_residual_blocks': 'decoder_num_residual_blocks'
        }
        
        for decoder_param, arg_name in decoder_params.items():
            if arg_name in args_dict and args_dict[arg_name] is not None:
                setattr(self.decoder, decoder_param, args_dict[arg_name])
        
        # These are shared across modes
        self._update_subconfig(self.decoder, args_dict)
        self._update_subconfig(self.distributed, args_dict)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config() 