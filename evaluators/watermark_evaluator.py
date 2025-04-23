"""
Evaluator for StyleGAN watermarking.
"""
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import lpips
import numpy as np
from tqdm import tqdm
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
import torch.nn as nn

from config.default_config import Config
from models.decoder import Decoder, FeatureDecoder
from models.key_mapper import KeyMapper
from models.model_utils import clone_model, load_stylegan2_model
from utils.checkpoint import load_checkpoint
from utils.metrics import calculate_metrics, save_metrics_plots, save_metrics_text
from utils.visualization import save_visualization, save_comparison_visualization
from utils.image_transforms import (
    apply_truncation, 
    quantize_model_weights, 
    downsample_and_upsample, 
    apply_jpeg_compression
)
from utils.model_loading import load_pretrained_models


class WatermarkEvaluator:
    """
    Evaluator for StyleGAN watermarking.
    """
    def __init__(
        self,
        config: Config,
        local_rank: int,
        rank: int,
        world_size: int,
        device: torch.device
    ):
        """
        Initialize the evaluator.
        
        Args:
            config (Config): Configuration object.
            local_rank (int): Local process rank.
            rank (int): Global process rank.
            world_size (int): Total number of processes.
            device (torch.device): Device to run evaluation on.
        """
        self.config = config
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.device = device
        
        # Enable timing logs
        self.enable_timing = getattr(self.config.evaluate, 'enable_timing_logs', True)
        if self.rank == 0 and self.enable_timing:
            logging.info("Timing logs are enabled for detailed process monitoring")
        
        # Initialize timing dictionary for tracking durations
        self.timing_stats = {}
        
        # Initialize models
        self.gan_model = None
        self.watermarked_model = None
        self.decoder = None
        self.key_mapper = None
        
        # Initialize loss functions
        self.lpips_loss_fn = None
        self.bce_loss_fn = nn.BCEWithLogitsLoss()  # Add BCE loss initialization
        
        # ZCA whitening parameters
        self.use_zca_whitening = getattr(self.config.model, 'use_zca_whitening', False)
        self.zca_eps = getattr(self.config.model, 'zca_eps', 1e-5)
        self.zca_batch_size = getattr(self.config.model, 'zca_batch_size', 1000)
        self.zca_mean = None
        self.whitening_factors = None
        
        # Handle selected indices based on the approach
        self.use_image_pixels = self.config.model.use_image_pixels
        
        # Track if we're using direct feature decoder mode
        self.direct_feature_decoder = getattr(self.config.model, 'direct_feature_decoder', False)
        if self.direct_feature_decoder and self.rank == 0:
            logging.info(f"Using direct feature decoder mode: decoder takes pixel features directly as input")
            
        # Flag for direct pixel prediction mode
        self.direct_pixel_pred = getattr(self.config.model, 'direct_pixel_pred', False)
        if self.direct_pixel_pred and self.rank == 0:
            logging.info(f"Using direct pixel prediction mode: decoder will be trained to predict pixel values directly")
        
        # Multi-decoder mode setup
        self.enable_multi_decoder = getattr(self.config.evaluate, 'enable_multi_decoder', False)
        if self.enable_multi_decoder:
            if self.rank == 0:
                logging.info("Multi-decoder mode enabled")
            self.decoders = []
            self.key_mappers = []
            self.image_pixel_indices_list = []
            self.key_lengths = self.config.evaluate.multi_decoder_key_lengths
            self.key_mapper_seeds = self.config.evaluate.multi_decoder_key_mapper_seeds
            self.pixel_counts = self.config.evaluate.multi_decoder_pixel_counts
            self.pixel_seeds = self.config.evaluate.multi_decoder_pixel_seeds
            
            # Validate configurations
            if not self.config.evaluate.multi_decoder_checkpoints:
                raise ValueError("Multi-decoder mode requires checkpoint paths")
            
            num_decoders = len(self.config.evaluate.multi_decoder_checkpoints)
            if self.key_lengths and len(self.key_lengths) != num_decoders:
                raise ValueError(f"Number of key lengths ({len(self.key_lengths)}) must match number of checkpoints ({num_decoders})")
            if self.key_mapper_seeds and len(self.key_mapper_seeds) != num_decoders:
                raise ValueError(f"Number of key mapper seeds ({len(self.key_mapper_seeds)}) must match number of checkpoints ({num_decoders})")
            if self.pixel_counts and len(self.pixel_counts) != num_decoders:
                raise ValueError(f"Number of pixel counts ({len(self.pixel_counts)}) must match number of checkpoints ({num_decoders})")
            if self.pixel_seeds and len(self.pixel_seeds) != num_decoders:
                raise ValueError(f"Number of pixel seeds ({len(self.pixel_seeds)}) must match number of checkpoints ({num_decoders})")
        
        if self.use_image_pixels:
            # For image-based approach
            if not self.enable_multi_decoder:
                self.image_pixel_indices = None
                self.image_pixel_count = self.config.model.image_pixel_count
                self.image_pixel_set_seed = self.config.model.image_pixel_set_seed
                if self.rank == 0:
                    logging.info(f"Using image-based approach with {self.image_pixel_count} pixels and seed {self.image_pixel_set_seed}")
        else:
            # For latent-based approach
            self.latent_indices = None
            
            # Check if explicit indices are provided
            if hasattr(self.config.model, 'selected_indices') and self.config.model.selected_indices is not None:
                if isinstance(self.config.model.selected_indices, str):
                    indices = [int(idx) for idx in self.config.model.selected_indices.split(',')]
                else:
                    indices = self.config.model.selected_indices
                # Convert to tensor immediately
                self.latent_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
                if self.rank == 0:
                    logging.info(f"Using manually specified latent indices: {indices}")
            else:
                # We'll generate random indices later once we know the latent dimension
                # For backward compatibility, default to 32 if w_partial_length not present
                self.w_partial_length = getattr(self.config.model, 'w_partial_length', 32)
                # Default seed for backward compatibility
                self.w_partial_set_seed = getattr(self.config.model, 'w_partial_set_seed', 42)
                if self.rank == 0:
                    logging.info(f"Will generate {self.w_partial_length} latent indices with seed {self.w_partial_set_seed}")
        
        # Setup models
        self.setup_models()
        
        # Load additional pretrained models if needed
        self.pretrained_models = {}
        if hasattr(config.evaluate, 'evaluate_pretrained') and config.evaluate.evaluate_pretrained:
            self.pretrained_models = load_pretrained_models(config, device, rank)
        
        # Setup the quantized model if we're evaluating quantization
        self.quantized_model = None
        self.quantized_watermarked_model = None
        if getattr(self.config.evaluate, 'evaluate_quantization', False):
            # Setup the quantized models will happen in setup_models
            pass
    
    def _generate_pixel_indices(self) -> None:
        """
        Generate random pixel indices for image-based approach.
        """
        # Set seed for reproducibility
        np.random.seed(self.image_pixel_set_seed)
        
        # Calculate total number of pixels
        img_size = self.config.model.img_size
        channels = 3  # RGB image
        total_pixels = channels * img_size * img_size
        
        # Generate random indices (without replacement)
        if self.image_pixel_count > total_pixels:
            if self.rank == 0:
                logging.warning(f"Requested {self.image_pixel_count} pixels exceeds total pixels {total_pixels}. Using all pixels.")
            self.image_pixel_count = total_pixels
            self.image_pixel_indices = np.arange(total_pixels)
        else:
            self.image_pixel_indices = np.random.choice(
                total_pixels, 
                size=self.image_pixel_count, 
                replace=False
            )
        
        # Convert to torch tensor and move to device for faster extraction
        self.image_pixel_indices = torch.tensor(self.image_pixel_indices, dtype=torch.long, device=self.device)
        
        if self.rank == 0:
            logging.info(f"Generated {len(self.image_pixel_indices)} pixel indices with seed {self.image_pixel_set_seed}")
            # Add detailed logging of the actual indices
            logging.info(f"Selected pixel indices: {self.image_pixel_indices.tolist()}")

    def extract_image_partial(self, images: torch.Tensor, pixel_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract partial image using selected pixel indices.
        
        Args:
            images (torch.Tensor): Batch of images [batch_size, channels, height, width]
            pixel_indices (torch.Tensor, optional): Indices of pixels to extract. If None, uses self.image_pixel_indices
            
        Returns:
            torch.Tensor: Batch of flattened pixel values at selected indices
        """
        batch_size = images.shape[0]
        
        # Flatten the spatial dimensions using a view operation (more efficient than reshape)
        flattened = images.view(batch_size, -1)
        
        # Handle multi-decoder mode
        if self.enable_multi_decoder:
            # In multi-decoder mode, use the first decoder's indices as default
            default_indices = self.image_pixel_indices_list[0] if self.image_pixel_indices_list else None
            indices = pixel_indices if pixel_indices is not None else default_indices
            if indices is None:
                raise ValueError("No pixel indices available. In multi-decoder mode, either provide pixel_indices or ensure image_pixel_indices_list is populated.")
        else:
            # Use provided indices or fall back to self.image_pixel_indices
            indices = pixel_indices if pixel_indices is not None else self.image_pixel_indices
            if indices is None:
                raise ValueError("No pixel indices available. Either provide pixel_indices or ensure image_pixel_indices is set.")
        
        # Get values at selected indices: [batch_size, pixel_count] using index_select or direct indexing
        image_partial = flattened.index_select(1, indices)
        
        return image_partial
    
    def _generate_latent_indices(self, latent_dim: int) -> None:
        """
        Generate random latent indices for latent-based approach.
        
        Args:
            latent_dim (int): Dimension of the latent space.
        """
        # Set seed for reproducibility
        np.random.seed(self.w_partial_set_seed)
        
        # Generate random indices (without replacement)
        if self.w_partial_length > latent_dim:
            if self.rank == 0:
                logging.warning(f"Requested {self.w_partial_length} indices exceeds latent dimension {latent_dim}. Using all dimensions.")
            self.w_partial_length = latent_dim
            indices = np.arange(latent_dim)
        else:
            indices = np.random.choice(
                latent_dim, 
                size=self.w_partial_length, 
                replace=False
            )
        # Convert to tensor
        self.latent_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        if self.rank == 0:
            logging.info(f"Generated {len(indices)} latent indices with seed {self.w_partial_set_seed}")
            
    def setup_models(self):
        """
        Initialize and set up all models.
        """
        # Load pretrained StyleGAN2 model
        self.gan_model = load_stylegan2_model(
            self.config.model.stylegan2_url,
            self.config.model.stylegan2_local_path,
            self.device
        )
        
        # Store the latent dimension from the GAN model
        self.latent_dim = self.gan_model.z_dim
        if self.rank == 0:
            logging.info(f"Using latent dimension: {self.latent_dim}")
            
        # Create watermarked model
        self.watermarked_model = clone_model(self.gan_model)
        self.watermarked_model.eval()
        self.watermarked_model.to(self.device)
        
        # Initialize decoder(s) based on mode
        if self.enable_multi_decoder:
            # Initialize decoders list for multi-decoder mode
            self.decoders = []
            self.key_mappers = []
            self.image_pixel_indices_list = []
            
            # Get configuration for each decoder
            checkpoints = self.config.evaluate.multi_decoder_checkpoints
            num_decoders = len(checkpoints)
            
            # Validate and set default values for multi-decoder parameters
            if not self.key_lengths:
                self.key_lengths = [self.config.model.key_length] * num_decoders
            if not self.key_mapper_seeds:
                self.key_mapper_seeds = [self.config.model.key_mapper_seed] * num_decoders
            if not self.pixel_counts:
                self.pixel_counts = [self.config.model.image_pixel_count] * num_decoders
            if not self.pixel_seeds:
                self.pixel_seeds = [self.config.model.image_pixel_set_seed + i for i in range(num_decoders)]
            
            if self.rank == 0:
                logging.info(f"\nSetting up {num_decoders} decoders:")
                for i in range(num_decoders):
                    logging.info(f"\nDecoder {i+1}:")
                    logging.info(f"  Checkpoint: {checkpoints[i]}")
                    logging.info(f"  Key length: {self.key_lengths[i]}")
                    logging.info(f"  Key mapper seed: {self.key_mapper_seeds[i]}")
                    logging.info(f"  Pixel count: {self.pixel_counts[i]}")
                    logging.info(f"  Pixel seed: {self.pixel_seeds[i]}")
            
            # Generate pixel indices for each decoder
            if self.use_image_pixels:
                img_size = self.config.model.img_size
                channels = 3
                total_pixels = channels * img_size * img_size
                
                for i in range(num_decoders):
                    np.random.seed(self.pixel_seeds[i])
                    pixel_count = self.pixel_counts[i]
                    
                    if pixel_count > total_pixels:
                        if self.rank == 0:
                            logging.warning(f"Decoder {i+1}: Requested {pixel_count} pixels exceeds total pixels {total_pixels}. Using all pixels.")
                        pixel_count = total_pixels
                        indices = np.arange(total_pixels)
                    else:
                        indices = np.random.choice(total_pixels, size=pixel_count, replace=False)
                    
                    # Convert to torch tensor and move to device
                    pixel_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
                    self.image_pixel_indices_list.append(pixel_indices)
                    
                    if self.rank == 0:
                        logging.info(f"  Generated {len(pixel_indices)} pixel indices for decoder {i+1}")
                
                # Set the base image_pixel_indices to the first decoder's indices
                self.image_pixel_indices = self.image_pixel_indices_list[0]
            
            # Initialize decoders and key mappers
            for i in range(num_decoders):
                # Initialize decoder
                if self.direct_feature_decoder:
                    input_dim = self.pixel_counts[i] if self.use_image_pixels else len(self.latent_indices)
                    decoder = FeatureDecoder(
                        input_dim=input_dim,
                        output_dim=self.key_lengths[i],
                        hidden_dims=self.config.decoder.hidden_dims,
                        activation=self.config.decoder.activation,
                        dropout_rate=self.config.decoder.dropout_rate,
                        num_residual_blocks=self.config.decoder.num_residual_blocks,
                        use_spectral_norm=self.config.decoder.use_spectral_norm,
                        use_layer_norm=self.config.decoder.use_layer_norm,
                        use_attention=self.config.decoder.use_attention
                    ).to(self.device)
                else:
                    decoder_output_dim = self.pixel_counts[i] if self.direct_pixel_pred else self.key_lengths[i]
                    decoder = Decoder(
                        image_size=self.config.model.img_size,
                        channels=3,
                        output_dim=decoder_output_dim
                    ).to(self.device)
                decoder.eval()
                
                # Initialize key mapper
                input_dim = self.pixel_counts[i] if self.use_image_pixels else len(self.latent_indices)
                key_mapper = KeyMapper(
                    input_dim=input_dim,
                    output_dim=self.key_lengths[i],
                    seed=self.key_mapper_seeds[i],
                    use_sine=getattr(self.config.model, 'key_mapper_use_sine', False),
                    sensitivity=getattr(self.config.model, 'key_mapper_sensitivity', 10.0)
                ).to(self.device)
                key_mapper.eval()
                
                # Load checkpoint for this decoder
                if self.rank == 0:
                    logging.info(f"Loading checkpoint for decoder {i+1} from {checkpoints[i]}...")
                load_checkpoint(
                    checkpoint_path=checkpoints[i],
                    watermarked_model=None,  # We already loaded the watermarked model
                    decoder=decoder,
                    key_mapper=None,  # Set to None to skip loading key mapper state
                    device=self.device
                )
                
                self.decoders.append(decoder)
                self.key_mappers.append(key_mapper)
        else:
            # Initialize single decoder
            if self.direct_feature_decoder:
                input_dim = self.image_pixel_count if self.use_image_pixels else len(self.latent_indices)
                if self.rank == 0:
                    logging.info(f"Initializing enhanced FeatureDecoder with input_dim={input_dim}, output_dim={self.config.model.key_length}")
                
                self.decoder = FeatureDecoder(
                    input_dim=input_dim,
                    output_dim=self.config.model.key_length,
                    hidden_dims=self.config.decoder.hidden_dims,
                    activation=self.config.decoder.activation,
                    dropout_rate=self.config.decoder.dropout_rate,
                    num_residual_blocks=self.config.decoder.num_residual_blocks,
                    use_spectral_norm=self.config.decoder.use_spectral_norm,
                    use_layer_norm=self.config.decoder.use_layer_norm,
                    use_attention=self.config.decoder.use_attention
                ).to(self.device)
            else:
                # Determine decoder output dimension based on mode
                if self.direct_pixel_pred:
                    # For direct pixel prediction, output dimension is the number of pixels we're predicting
                    decoder_output_dim = self.image_pixel_count
                    if self.rank == 0:
                        logging.info(f"Setting decoder output dimension to {decoder_output_dim} for direct pixel prediction")
                else:
                    # Normal mode - output dimension is the binary key length
                    decoder_output_dim = self.config.model.key_length
                
                self.decoder = Decoder(
                    image_size=self.config.model.img_size,
                    channels=3,
                    output_dim=decoder_output_dim
                ).to(self.device)
            self.decoder.eval()
            
            # Initialize key mapper
            if self.rank == 0:
                logging.info(f"Initializing key mapper with seed {self.config.model.key_mapper_seed}")
            
            # Determine key mapper input dimension based on approach
            if self.use_image_pixels:
                input_dim = self.image_pixel_count
            else:
                input_dim = len(self.latent_indices)
            
            self.key_mapper = KeyMapper(
                input_dim=input_dim,
                output_dim=self.config.model.key_length,
                seed=self.config.model.key_mapper_seed,
                use_sine=getattr(self.config.model, 'key_mapper_use_sine', False),
                sensitivity=getattr(self.config.model, 'key_mapper_sensitivity', 10.0)
            ).to(self.device)
            self.key_mapper.eval()
            
            # Generate pixel indices if using image-based approach
            if self.use_image_pixels:
                self._generate_pixel_indices()
        
        # Load main checkpoint
        if self.rank == 0:
            logging.info(f"Loading checkpoint from {self.config.checkpoint_path}...")
        
        checkpoint = load_checkpoint(
            checkpoint_path=self.config.checkpoint_path,
            watermarked_model=self.watermarked_model,
            decoder=None if self.enable_multi_decoder else self.decoder,
            key_mapper=None,  # Set to None to skip loading key mapper state
            device=self.device
        )
        
        # Load ZCA parameters from checkpoint if they exist
        if 'zca_mean' in checkpoint and 'whitening_factors' in checkpoint:
            self.zca_mean = checkpoint['zca_mean']
            self.whitening_factors = checkpoint['whitening_factors']
            if self.rank == 0:
                logging.info("Loaded ZCA parameters from checkpoint")
        else:
            if self.rank == 0:
                logging.warning("No ZCA parameters found in checkpoint, will recompute if needed")
            self.zca_mean = None
            self.whitening_factors = None
            
        # Setup multiple decoders if in multi-decoder mode
        if self.enable_multi_decoder:
            # Setup multiple decoders and key mappers
            checkpoints = self.config.evaluate.multi_decoder_checkpoints
            if self.rank == 0:
                logging.info(f"Setting up {len(checkpoints)} decoders and key mappers...")
            
            for i, checkpoint_path in enumerate(checkpoints):
                # Get configuration for this decoder
                key_length = self.key_lengths[i] if self.key_lengths else self.config.model.key_length
                key_mapper_seed = self.key_mapper_seeds[i] if self.key_mapper_seeds else self.config.model.key_mapper_seed
                pixel_count = self.pixel_counts[i] if self.pixel_counts else self.config.model.image_pixel_count
                pixel_seed = self.pixel_seeds[i] if self.pixel_seeds else self.config.model.image_pixel_set_seed
                
                if self.rank == 0:
                    logging.info(f"\nSetting up decoder {i+1}/{len(checkpoints)}:")
                    logging.info(f"  Checkpoint: {checkpoint_path}")
                    logging.info(f"  Key length: {key_length}")
                    logging.info(f"  Key mapper seed: {key_mapper_seed}")
                    logging.info(f"  Pixel count: {pixel_count}")
                    logging.info(f"  Pixel seed: {pixel_seed}")
                
                # Generate pixel indices for this decoder
                if self.use_image_pixels:
                    np.random.seed(pixel_seed)
                    img_size = self.config.model.img_size
                    channels = 3
                    total_pixels = channels * img_size * img_size
                    
                    if pixel_count > total_pixels:
                        if self.rank == 0:
                            logging.warning(f"Requested {pixel_count} pixels exceeds total pixels {total_pixels}. Using all pixels.")
                        pixel_count = total_pixels
                        indices = np.arange(total_pixels)
                    else:
                        indices = np.random.choice(total_pixels, size=pixel_count, replace=False)
                    
                    # Convert to torch tensor and move to device
                    pixel_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
                    self.image_pixel_indices_list.append(pixel_indices)
                
                # Initialize decoder
                if self.direct_feature_decoder:
                    input_dim = pixel_count if self.use_image_pixels else len(self.latent_indices)
                    decoder = FeatureDecoder(
                        input_dim=input_dim,
                        output_dim=key_length,
                        hidden_dims=self.config.decoder.hidden_dims,
                        activation=self.config.decoder.activation,
                        dropout_rate=self.config.decoder.dropout_rate,
                        num_residual_blocks=self.config.decoder.num_residual_blocks,
                        use_spectral_norm=self.config.decoder.use_spectral_norm,
                        use_layer_norm=self.config.decoder.use_layer_norm,
                        use_attention=self.config.decoder.use_attention
                    ).to(self.device)
                else:
                    # Determine decoder output dimension based on mode
                    if self.direct_pixel_pred:
                        # For direct pixel prediction, output dimension is the number of pixels we're predicting
                        decoder_output_dim = pixel_count
                        if self.rank == 0:
                            logging.info(f"  Setting decoder output dimension to {decoder_output_dim} for direct pixel prediction")
                    else:
                        # Normal mode - output dimension is the binary key length
                        decoder_output_dim = key_length
                    
                    decoder = Decoder(
                        image_size=self.config.model.img_size,
                        channels=3,
                        output_dim=decoder_output_dim
                    ).to(self.device)
                decoder.eval()
                
                # Initialize key mapper
                input_dim = pixel_count if self.use_image_pixels else len(self.latent_indices)
                key_mapper = KeyMapper(
                    input_dim=input_dim,
                    output_dim=key_length,
                    seed=key_mapper_seed,
                    use_sine=getattr(self.config.model, 'key_mapper_use_sine', False),
                    sensitivity=getattr(self.config.model, 'key_mapper_sensitivity', 10.0)
                ).to(self.device)
                key_mapper.eval()
                
                # Load checkpoint for this decoder
                if self.rank == 0:
                    logging.info(f"Loading checkpoint from {checkpoint_path}...")
                load_checkpoint(
                    checkpoint_path=checkpoint_path,
                    watermarked_model=None,  # We already loaded the watermarked model
                    decoder=decoder,
                    key_mapper=None,  # Set to None to skip loading key mapper state
                    device=self.device
                )
                
                self.decoders.append(decoder)
                self.key_mappers.append(key_mapper)
        
        # Initialize LPIPS loss with the same network as in training
        self.lpips_loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.lpips_loss_fn.eval()  # Ensure it's in eval mode
        
        # Validate direct pixel prediction mode
        if self.direct_pixel_pred:
            # Ensure that direct_pixel_pred is only used with image-based approach
            if not self.use_image_pixels:
                self.direct_pixel_pred = False
                if self.rank == 0:
                    logging.warning("direct_pixel_pred can only be used with image-based approach (use_image_pixels=True). Disabling it.")
            elif self.rank == 0:
                logging.info("Validated direct pixel prediction mode: decoder will predict selected pixel values directly")
        
        # Setup quantized models - only if explicitly enabled
        self.quantized_models = {}
        self.quantized_watermarked_models = {}
        
        # Lazy initialization for quantized models
        need_int8_original = getattr(self.config.evaluate, 'evaluate_quantization', False)
        need_int8_watermarked = getattr(self.config.evaluate, 'evaluate_quantization_watermarked', False)
        need_int4_original = getattr(self.config.evaluate, 'evaluate_quantization_int4', False)
        need_int4_watermarked = getattr(self.config.evaluate, 'evaluate_quantization_int4_watermarked', False)
        need_int2_original = getattr(self.config.evaluate, 'evaluate_quantization_int2', False)
        need_int2_watermarked = getattr(self.config.evaluate, 'evaluate_quantization_int2_watermarked', False)
        
        # Only set up the models we need
        if need_int8_original:
            if self.rank == 0:
                logging.info("Setting up int8 quantized model...")
            self.quantized_models['int8'] = quantize_model_weights(self.gan_model, 'int8')
            self.quantized_models['int8'].eval()
            
        if need_int8_watermarked:
            if self.rank == 0:
                logging.info("Setting up int8 quantized watermarked model...")
            self.quantized_watermarked_models['int8'] = quantize_model_weights(self.watermarked_model, 'int8')
            self.quantized_watermarked_models['int8'].eval()
            
        if need_int4_original:
            if self.rank == 0:
                logging.info("Setting up int4 quantized model...")
            self.quantized_models['int4'] = quantize_model_weights(self.gan_model, 'int4')
            self.quantized_models['int4'].eval()
            
        if need_int4_watermarked:
            if self.rank == 0:
                logging.info("Setting up int4 quantized watermarked model...")
            self.quantized_watermarked_models['int4'] = quantize_model_weights(self.watermarked_model, 'int4')
            self.quantized_watermarked_models['int4'].eval()
            
        if need_int2_original:
            if self.rank == 0:
                logging.info("Setting up int2 quantized model...")
            self.quantized_models['int2'] = quantize_model_weights(self.gan_model, 'int2')
            self.quantized_models['int2'].eval()
            
        if need_int2_watermarked:
            if self.rank == 0:
                logging.info("Setting up int2 quantized watermarked model...")
            self.quantized_watermarked_models['int2'] = quantize_model_weights(self.watermarked_model, 'int2')
            self.quantized_watermarked_models['int2'].eval()
        
        # For backward compatibility
        if 'int8' in self.quantized_models:
            self.quantized_model = self.quantized_models['int8']
        else:
            self.quantized_model = None
            
        if 'int8' in self.quantized_watermarked_models:
            self.quantized_watermarked_model = self.quantized_watermarked_models['int8'] 
        else:
            self.quantized_watermarked_model = None
        
        # After setting up models, compute ZCA parameters if needed
        if self.use_zca_whitening:
            self.compute_zca_parameters()
    
    def compute_zca_parameters(self) -> None:
        """
        Compute ZCA whitening parameters using batches of generated images.
        This is done once at the start of evaluation using memory-efficient computation.
        """
        # If we already have ZCA parameters from the checkpoint, skip computation
        if self.zca_mean is not None and self.whitening_factors is not None:
            if self.rank == 0:
                logging.info("Using ZCA parameters from checkpoint")
            return
            
        if not self.use_zca_whitening:
            return
            
        if self.rank == 0:
            logging.info("Computing ZCA whitening parameters (no parameters found in checkpoint)...")
        
        # Use smaller batch size for synthesis to avoid memory issues
        synthesis_batch_size = 16  # Small enough to avoid memory issues
        num_batches = (self.zca_batch_size + synthesis_batch_size - 1) // synthesis_batch_size
        
        # First pass: compute mean
        sum_x = None
        total_samples = 0
        input_dim = None
        
        with torch.no_grad():
            latent_dim = self.gan_model.z_dim
            
            # First pass to compute mean
            for batch_idx in range(num_batches):
                current_batch_size = min(synthesis_batch_size, 
                                      self.zca_batch_size - batch_idx * synthesis_batch_size)
                
                if current_batch_size <= 0:
                    break
                
                # Generate latent vectors for this batch
                z = torch.randn(current_batch_size, latent_dim, device=self.device)
                
                # Generate images using watermarked model
                if hasattr(self.watermarked_model, 'module'):
                    w = self.watermarked_model.module.mapping(z, None)
                    x = self.watermarked_model.module.synthesis(w, noise_mode="const")
                else:
                    w = self.watermarked_model.mapping(z, None)
                    x = self.watermarked_model.synthesis(w, noise_mode="const")
                
                # Reshape images to 2D matrix
                x_flat = x.view(x.size(0), -1)
                
                # Store input dimension
                if input_dim is None:
                    input_dim = x_flat.size(1)
                
                # Initialize or accumulate mean
                if sum_x is None:
                    sum_x = torch.zeros(input_dim, device=self.device)
                
                sum_x += x_flat.sum(dim=0)
                total_samples += current_batch_size
                
                if self.rank == 0 and batch_idx % 10 == 0:
                    logging.info(f"Computing mean: batch {batch_idx + 1}/{num_batches}")
            
            # Compute mean
            self.zca_mean = (sum_x / total_samples).unsqueeze(0)
            
            # Free memory
            del sum_x
            torch.cuda.empty_cache()
            
            if self.rank == 0:
                logging.info("Mean computation completed. Starting variance computation...")
            
            # Second pass: compute variance for diagonal whitening
            sum_var = torch.zeros(input_dim, device=self.device)
            
            for batch_idx in range(num_batches):
                current_batch_size = min(synthesis_batch_size, 
                                      self.zca_batch_size - batch_idx * synthesis_batch_size)
                
                if current_batch_size <= 0:
                    break
                
                z = torch.randn(current_batch_size, latent_dim, device=self.device)
                
                if hasattr(self.watermarked_model, 'module'):
                    w = self.watermarked_model.module.mapping(z, None)
                    x = self.watermarked_model.module.synthesis(w, noise_mode="const")
                else:
                    w = self.watermarked_model.mapping(z, None)
                    x = self.watermarked_model.synthesis(w, noise_mode="const")
                
                x_flat = x.view(x.size(0), -1)
                x_centered = x_flat - self.zca_mean
                
                # Accumulate variance
                sum_var += torch.sum(x_centered ** 2, dim=0)
                
                if self.rank == 0 and batch_idx % 10 == 0:
                    logging.info(f"Computing variance: batch {batch_idx + 1}/{num_batches}")
            
            # Compute variance and create diagonal whitening matrix
            var = sum_var / total_samples
            
            # Add epsilon for numerical stability
            var += self.zca_eps
            
            # Store the whitening factors (sqrt of inverse variance)
            self.whitening_factors = torch.where(var > self.zca_eps,
                                               var.pow(-0.5),
                                               torch.zeros_like(var))
            
            if self.rank == 0:
                logging.info("Whitening parameters computed successfully")
            
            # Clean up
            del var, sum_var
            torch.cuda.empty_cache()
    
    def apply_zca_whitening(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ZCA whitening to input images.
        
        Args:
            x (torch.Tensor): Input images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Whitened images
        """
        if not self.use_zca_whitening:
            return x
            
        # Process in chunks to save memory
        batch_size = x.size(0)
        chunk_size = 16  # Process 16 images at a time
        x_whitened_chunks = []
        
        for i in range(0, batch_size, chunk_size):
            # Get chunk
            x_chunk = x[i:i + chunk_size]
            
            # Reshape to 2D matrix
            x_shape = x_chunk.shape
            x_flat = x_chunk.view(x_chunk.size(0), -1)
            
            # Center the data
            x_centered = x_flat - self.zca_mean
            
            # Apply diagonal whitening
            x_whitened = x_centered * self.whitening_factors
            
            # Reshape back to image format
            x_whitened = x_whitened.view(x_shape)
            x_whitened_chunks.append(x_whitened)
        
        # Concatenate chunks
        return torch.cat(x_whitened_chunks, dim=0)
    
    def process_batch(self, z, model_name=None, transformation=None):
        """
        Process a batch of latent vectors through the model pipeline.
        
        Args:
            z (torch.Tensor): Batch of latent vectors
            model_name (str, optional): Name of the model being evaluated (for logging)
            transformation (str, optional): Name of the transformation being applied
            
        Returns:
            dict: Dictionary containing batch processing results
        """
        # Generate images using either original or watermarked model
        model = self.gan_model if model_name == "original" else self.watermarked_model
        x = model(z)
        
        # Apply transformation if specified
        if transformation:
            x = self.apply_transformation(x, transformation)
        
        # Process batch differently based on mode
        if self.enable_multi_decoder:
            # Process each decoder separately
            all_features = []
            all_predictions = []
            all_keys = []
            all_metrics = []
            
            for idx, (decoder, key_mapper, pixel_indices) in enumerate(zip(self.decoders, self.key_mappers, self.image_pixel_indices_list)):
                # Extract features using current decoder's pixel indices
                features = self.extract_image_partial(x, pixel_indices)
                all_features.append(features)
                
                # Get predictions from current decoder
                if self.direct_feature_decoder:
                    predictions = decoder(features)
                else:
                    predictions = decoder(x)
                all_predictions.append(predictions)
                
                # Get keys from current key mapper
                keys = key_mapper(features)
                all_keys.append(keys)
                
                # Calculate metrics for this decoder
                pred_probs = torch.sigmoid(predictions)
                mse_distances = torch.mean(torch.pow(pred_probs - keys, 2), dim=1)
                mae_distances = torch.mean(torch.abs(pred_probs - keys), dim=1)
                
                metrics = {
                    'decoder_idx': idx,
                    'key_length': self.key_lengths[idx] if self.key_lengths else self.config.model.key_length,
                    'pixel_count': len(pixel_indices),
                    'mse_distances': mse_distances.cpu().numpy(),
                    'mae_distances': mae_distances.cpu().numpy()
                }
                all_metrics.append(metrics)
            
            return {
                'features': all_features,
                'predictions': all_predictions,
                'keys': all_keys,
                'metrics': all_metrics,
                'images': x
            }
        else:
            # Single decoder mode
            # Extract features using the base image_pixel_indices
            features = self.extract_image_partial(x)
            
            # Get predictions from decoder
            if self.direct_feature_decoder:
                predictions = self.decoder(features)
            else:
                predictions = self.decoder(x)
            
            # Get keys from key mapper
            keys = self.key_mapper(features)
            
            return {
                'features': features,
                'predictions': predictions,
                'keys': keys,
                'images': x
            }
    
    def evaluate_batch(self) -> Dict[str, float]:
        """
        Run batch evaluation to compute metrics.
        
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        if self.rank == 0:
            logging.info(f"Running batch evaluation with {self.config.evaluate.num_samples} samples...")
        
        try:
            # Set seed for reproducibility
            if hasattr(self.config.evaluate, 'seed') and self.config.evaluate.seed is not None:
                np.random.seed(self.config.evaluate.seed)
                torch.manual_seed(self.config.evaluate.seed)
                if self.rank == 0:
                    logging.info(f"Using fixed random seed {self.config.evaluate.seed} for evaluation")
            
            # Create empty accumulators
            batch_size = self.config.evaluate.batch_size
            num_samples = self.config.evaluate.num_samples
            num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
            
            # Generate all latent vectors upfront for consistency across evaluations
            all_z = torch.randn(num_samples, self.latent_dim, device=self.device)
            
            # Process differently based on whether we're in direct pixel prediction mode
            if self.direct_pixel_pred:
                # For direct pixel prediction, we need to measure pixel prediction accuracy
                if self.rank == 0:
                    logging.info(f"Evaluating in direct pixel prediction mode with {num_samples} samples and {self.image_pixel_count} pixels...")
                
                metrics = self._evaluate_direct_pixel_pred_batch(all_z, num_batches, batch_size, num_samples)
                
                if not metrics:
                    if self.rank == 0:
                        logging.error("No valid results accumulated during batch processing")
                    return {}
                
                # Save metrics
                if self.rank == 0:
                    save_metrics_text(metrics, self.config.output_dir)
                    
                    # Also create a simple visualization of pixel MSE distribution
                    try:
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(10, 6))
                        plt.hist(metrics['pixel_mse_values'], bins=50, alpha=0.75)
                        plt.title('Pixel MSE Distribution')
                        plt.xlabel('MSE')
                        plt.ylabel('Frequency')
                        plt.grid(True)
                        plt.savefig(os.path.join(self.config.output_dir, 'pixel_mse_distribution.png'), dpi=300)
                        plt.close()
                        
                        logging.info(f"Created pixel MSE distribution visualization in {self.config.output_dir}")
                    except Exception as e:
                        logging.warning(f"Failed to create pixel MSE distribution visualization: {str(e)}")
                
                return metrics
            else:
                # Normal watermarking mode - process comparison batches (original vs watermarked)
                watermarked_distances_all = []
                original_distances_all = []
                lpips_losses_all = []
                
                # Process in batches
                with torch.no_grad():
                    for i in range(num_batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, num_samples)
                        current_batch_size = end_idx - start_idx
                        
                        # Extract batch of latent vectors
                        z = all_z[start_idx:end_idx]
                        
                        # Process the comparison batch
                        batch_results = self.process_batch(z)
                        
                        if not all(k in batch_results for k in ['watermarked_mse_distances', 'original_mse_distances', 'lpips_losses']):
                            if self.rank == 0:
                                logging.error(f"Missing metrics in batch results: {batch_results.keys()}")
                            continue
                        
                        # Accumulate results
                        watermarked_distances_all.append(batch_results['watermarked_mse_distances'])
                        original_distances_all.append(batch_results['original_mse_distances'])
                        lpips_losses_all.append(batch_results['lpips_losses'])
                        
                        # Progress reporting for long running evaluation
                        if self.rank == 0 and num_batches > 10 and (i+1) % max(1, num_batches//10) == 0:
                            logging.info(f"Processed {i+1}/{num_batches} batches for original vs watermarked comparison")
                
                # Verify we have results before proceeding
                if not watermarked_distances_all or not original_distances_all or not lpips_losses_all:
                    if self.rank == 0:
                        logging.error("No valid results accumulated during batch processing")
                    return {}
                
                # Negative sample evaluation - only if enabled and we have samples to process
                negative_distances_all = {}
                if getattr(self.config.evaluate, 'evaluate_neg_samples', True):
                    negative_distances_all = self._evaluate_negative_samples(all_z)
                
                # Concatenate results across batches
                watermarked_distances = np.concatenate(watermarked_distances_all)
                original_distances = np.concatenate(original_distances_all)
                lpips_losses = np.concatenate(lpips_losses_all)
                
                # Calculate metrics
                threshold = 0.25  # Typical threshold for correct key detection
                watermarked_correct = np.sum(watermarked_distances < threshold)
                original_correct = np.sum(original_distances < threshold)
                total_samples = len(watermarked_distances)
                
                # Use MSE distances as MAE distances for backward compatibility
                watermarked_mae_distances = watermarked_distances
                original_mae_distances = original_distances
                
                # Calculate metrics
                metrics, roc_data = calculate_metrics(
                    watermarked_distances, 
                    original_distances,
                    watermarked_mae_distances, 
                    original_mae_distances,
                    watermarked_correct,
                    original_correct,
                    total_samples,
                    lpips_losses
                )
                
                # Add raw distances to metrics
                metrics['watermarked_mse_distances'] = watermarked_distances
                metrics['original_mse_distances'] = original_distances
                metrics['lpips_losses'] = lpips_losses
                
                if negative_distances_all:
                    metrics['negative_distances_all'] = negative_distances_all
                
                metrics['roc_data'] = roc_data
                
                if self.rank == 0:
                    # Save metrics
                    save_metrics_text(metrics, self.config.output_dir)
                    save_metrics_plots(metrics, roc_data, watermarked_distances, original_distances, 
                                       watermarked_mae_distances, original_mae_distances, self.config.output_dir)
                
                return metrics
                
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error in batch evaluation: {str(e)}")
                logging.error(str(e), exc_info=True)
            return {}
    
    def _evaluate_direct_pixel_pred_batch(self, all_z, num_batches, batch_size, num_samples):
        """
        Evaluate direct pixel prediction mode by measuring pixel prediction accuracy.
        
        Args:
            all_z (torch.Tensor): All latent vectors
            num_batches (int): Number of batches
            batch_size (int): Batch size
            num_samples (int): Total number of samples
            
        Returns:
            dict: Metrics dictionary for direct pixel prediction
        """
        # First compute watermarked model metrics to establish threshold
        watermarked_mse_values = []
        watermarked_mae_values = []
        
        # Process watermarked model batches
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                current_batch_size = end_idx - start_idx
                
                # Extract batch of latent vectors
                z = all_z[start_idx:end_idx]
                
                # Generate watermarked images
                if hasattr(self.watermarked_model, 'module'):
                    w = self.watermarked_model.module.mapping(z, None)
                    x = self.watermarked_model.module.synthesis(w, noise_mode="const")
                else:
                    w = self.watermarked_model.mapping(z, None)
                    x = self.watermarked_model.synthesis(w, noise_mode="const")
                
                # Extract features (real pixel values)
                features = self.extract_image_partial(x)
                true_values = features
                
                # Apply ZCA whitening to decoder input if enabled
                x_decoder = self.apply_zca_whitening(x) if self.use_zca_whitening else x
                
                # Predict pixel values
                pred_values = self.decoder(x_decoder)
                
                # Calculate metrics
                mse = torch.mean(torch.pow(pred_values - true_values, 2), dim=1).cpu().numpy()
                mae = torch.mean(torch.abs(pred_values - true_values), dim=1).cpu().numpy()
                
                watermarked_mse_values.append(mse)
                watermarked_mae_values.append(mae)
                
                # Progress reporting
                if self.rank == 0 and num_batches > 10 and (i+1) % max(1, num_batches//10) == 0:
                    logging.info(f"Processed {i+1}/{num_batches} batches for direct pixel prediction")
        
        # Combine watermarked results
        watermarked_mse_all = np.concatenate(watermarked_mse_values)
        watermarked_mae_all = np.concatenate(watermarked_mae_values)
        
        # Calculate threshold at 95% TPR for watermarked model
        sorted_mse = np.sort(watermarked_mse_all)
        sorted_mae = np.sort(watermarked_mae_all)
        threshold_idx = int(0.95 * len(sorted_mse))
        mse_threshold = sorted_mse[threshold_idx]
        mae_threshold = sorted_mae[threshold_idx]
        
        if self.rank == 0:
            logging.info(f"Computed thresholds at 95% TPR - MSE: {mse_threshold:.6f}, MAE: {mae_threshold:.6f}")
        
        # Create metrics dictionary
        metrics = {
            'pixel_mse_mean': np.mean(watermarked_mse_all),
            'pixel_mse_std': np.std(watermarked_mse_all),
            'pixel_mae_mean': np.mean(watermarked_mae_all),
            'pixel_mae_std': np.std(watermarked_mae_all),
            'pixel_mse_values': watermarked_mse_all,
            'pixel_mae_values': watermarked_mae_all,
            'mse_threshold_95tpr': mse_threshold,
            'mae_threshold_95tpr': mae_threshold
        }
        
        # Evaluate negative samples if enabled
        negative_results = {}
        if getattr(self.config.evaluate, 'evaluate_neg_samples', True):
            negative_results = self._evaluate_negative_samples_direct_pixel(all_z)
            
            # Calculate FPR at 95% TPR threshold for each negative case
            if negative_results:
                for key, result in negative_results.items():
                    mse_all = result['mse_values']
                    mae_all = result['mae_values']
                    
                    # Calculate FPR (percentage of negative samples below threshold)
                    fpr_mse = np.mean(mse_all <= mse_threshold) * 100
                    fpr_mae = np.mean(mae_all <= mae_threshold) * 100
                    
                    result['fpr_95tpr_mse'] = fpr_mse
                    result['fpr_95tpr_mae'] = fpr_mae
        
        # Add negative results to metrics
        if negative_results:
            metrics['negative_results'] = negative_results
        
        # Log summary metrics
        if self.rank == 0:
            logging.info("\nDirect Pixel Prediction Results:")
            logging.info("-" * 100)
            logging.info(f"Watermarked Model - Pixel MSE: {metrics['pixel_mse_mean']:.6f} ± {metrics['pixel_mse_std']:.6f}")
            logging.info(f"Watermarked Model - Pixel MAE: {metrics['pixel_mae_mean']:.6f} ± {metrics['pixel_mae_std']:.6f}")
            
            # Print negative sample results if available
            if negative_results:
                logging.info("\nNegative Sample Results:")
                logging.info("-" * 100)
                logging.info(f"{'Model/Transform':<40}{'Pixel MSE':>15}{'FPR@95%TPR':>15}{'Pixel MAE':>15}{'FPR@95%TPR':>15}")
                logging.info("-" * 100)
                
                for name, result in negative_results.items():
                    logging.info(f"{name:<40}{result['mse_mean']:>15.6f}{result['fpr_95tpr_mse']:>15.2f}%{result['mae_mean']:>15.6f}{result['fpr_95tpr_mae']:>15.2f}%")
            
            logging.info("-" * 100)
        
        return metrics
    
    def _evaluate_negative_samples_direct_pixel(self, all_z):
        """
        Evaluate negative samples for direct pixel prediction mode.
        
        Args:
            all_z (torch.Tensor): All latent vectors
            
        Returns:
            dict: Dictionary mapping negative sample types to their pixel prediction metrics
        """
        negative_results = {}
        batch_size = self.config.evaluate.batch_size
        num_samples = self.config.evaluate.num_samples
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        # Get the list of evaluations to run
        evaluations_to_run = []
        
        # Add pretrained model evaluations
        if getattr(self.config.evaluate, 'evaluate_pretrained', True):
            if getattr(self.config.evaluate, 'evaluate_ffhq1k', True) and 'ffhq1k' in self.pretrained_models:
                evaluations_to_run.append(('ffhq1k', None))
            
            if getattr(self.config.evaluate, 'evaluate_ffhq30k', True) and 'ffhq30k' in self.pretrained_models:
                evaluations_to_run.append(('ffhq30k', None))
            
            if getattr(self.config.evaluate, 'evaluate_ffhq70k_bcr', True) and 'ffhq70k-bcr' in self.pretrained_models:
                evaluations_to_run.append(('ffhq70k-bcr', None))
            
            if getattr(self.config.evaluate, 'evaluate_ffhq70k_noaug', True) and 'ffhq70k-noaug' in self.pretrained_models:
                evaluations_to_run.append(('ffhq70k-noaug', None))
        
        # Add transformations
        if getattr(self.config.evaluate, 'evaluate_transforms', True):
            # Truncation
            if getattr(self.config.evaluate, 'evaluate_truncation', True):
                evaluations_to_run.append((None, 'truncation_original'))
            if getattr(self.config.evaluate, 'evaluate_truncation_watermarked', True):
                evaluations_to_run.append((None, 'truncation_watermarked'))
            
            # Int8 Quantization
            if getattr(self.config.evaluate, 'evaluate_quantization', True):
                evaluations_to_run.append((None, 'quantization_original'))
            if getattr(self.config.evaluate, 'evaluate_quantization_watermarked', True):
                evaluations_to_run.append((None, 'quantization_watermarked'))
            
            # Int4 Quantization
            if getattr(self.config.evaluate, 'evaluate_quantization_int4', True):
                evaluations_to_run.append((None, 'quantization_int4_original'))
            if getattr(self.config.evaluate, 'evaluate_quantization_int4_watermarked', True):
                evaluations_to_run.append((None, 'quantization_int4_watermarked'))
            
            # Int2 Quantization
            if getattr(self.config.evaluate, 'evaluate_quantization_int2', True):
                evaluations_to_run.append((None, 'quantization_int2_original'))
            if getattr(self.config.evaluate, 'evaluate_quantization_int2_watermarked', True):
                evaluations_to_run.append((None, 'quantization_int2_watermarked'))
            
            # Downsampling
            if getattr(self.config.evaluate, 'evaluate_downsample', True):
                evaluations_to_run.append((None, 'downsample_original'))
            if getattr(self.config.evaluate, 'evaluate_downsample_watermarked', True):
                evaluations_to_run.append((None, 'downsample_watermarked'))
            
            # JPEG compression
            if getattr(self.config.evaluate, 'evaluate_jpeg', True):
                evaluations_to_run.append((None, 'jpeg_original'))
            if getattr(self.config.evaluate, 'evaluate_jpeg_watermarked', True):
                evaluations_to_run.append((None, 'jpeg_watermarked'))
        
        # Run evaluations
        total_evals = len(evaluations_to_run)
        if self.rank == 0 and total_evals > 0:
            logging.info(f"Running {total_evals} negative sample evaluations for direct pixel prediction...")
        
        with torch.no_grad():
            for idx, (model_name, transformation) in enumerate(evaluations_to_run):
                key = model_name if model_name else transformation
                
                # Skip if we've already evaluated this configuration
                if key in negative_results:
                    continue
                
                mse_per_batch = []
                mae_per_batch = []
                
                # Process in batches
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_samples)
                    
                    # Extract batch of latent vectors
                    z = all_z[start_idx:end_idx]
                    
                    # Generate negative sample images
                    if model_name is not None:
                        # Use pretrained model
                        model = self.pretrained_models[model_name]
                        
                        if hasattr(model, 'module'):
                            w = model.module.mapping(z, None)
                            x = model.module.synthesis(w, noise_mode="const")
                        else:
                            w = model.mapping(z, None)
                            x = model.synthesis(w, noise_mode="const")
                    else:
                        # Use transformation on watermarked or original model
                        use_watermarked = "watermarked" in transformation
                        source_model = self.watermarked_model if use_watermarked else self.gan_model
                        
                        if 'truncation' in transformation:
                            # For truncation, use apply_truncation with the model and latent vectors
                            truncation_psi = getattr(self.config.evaluate, 'truncation_psi', 2.0)
                            x = apply_truncation(source_model, z, truncation_psi)
                        elif 'quantization' in transformation:
                            # For quantization, use the pre-quantized models
                            precision = 'int8'
                            if 'int4' in transformation:
                                precision = 'int4'
                            elif 'int2' in transformation:
                                precision = 'int2'
                                
                            if use_watermarked:
                                model = self.quantized_watermarked_models[precision]
                            else:
                                model = self.quantized_models[precision]
                                
                            if hasattr(model, 'module'):
                                w = model.module.mapping(z, None)
                                x = model.module.synthesis(w, noise_mode="const")
                            else:
                                w = model.mapping(z, None)
                                x = model.synthesis(w, noise_mode="const")
                        else:
                            # Normal generation followed by transformation
                            if hasattr(source_model, 'module'):
                                w = source_model.module.mapping(z, None)
                                x = source_model.module.synthesis(w, noise_mode="const")
                            else:
                                w = source_model.mapping(z, None)
                                x = source_model.synthesis(w, noise_mode="const")
                            
                            # Apply other transformations after generation
                            if transformation:
                                x = self.apply_transformation(x, transformation)
                    
                    # Extract features (real pixel values)
                    features = self.extract_image_partial(x)
                    true_values = features
                    
                    # Apply ZCA whitening to decoder input if enabled
                    x_decoder = self.apply_zca_whitening(x) if self.use_zca_whitening else x
                    
                    # Predict pixel values
                    pred_values = self.decoder(x_decoder)
                    
                    # Calculate metrics
                    mse = torch.mean(torch.pow(pred_values - true_values, 2), dim=1).cpu().numpy()
                    mae = torch.mean(torch.abs(pred_values - true_values), dim=1).cpu().numpy()
                    
                    mse_per_batch.append(mse)
                    mae_per_batch.append(mae)
                
                # Combine results
                mse_all = np.concatenate(mse_per_batch)
                mae_all = np.concatenate(mae_per_batch)
                
                # Store metrics
                negative_results[key] = {
                    'mse_mean': np.mean(mse_all),
                    'mse_std': np.std(mse_all),
                    'mae_mean': np.mean(mae_all),
                    'mae_std': np.std(mae_all),
                    'mse_values': mse_all,  # Store all values for FPR calculation
                    'mae_values': mae_all   # Store all values for FPR calculation
                }
                
                # Progress reporting
                if self.rank == 0 and (idx+1) % max(1, total_evals//5) == 0:
                    logging.info(f"Completed {idx+1}/{total_evals} negative sample evaluations")
        
        return negative_results
    
    def _evaluate_negative_samples(self, all_z):
        """
        Evaluate negative samples by comparing against pretrained models and transformations.
        
        Args:
            all_z (torch.Tensor): All latent vectors for evaluation
            
        Returns:
            dict: Dictionary mapping negative sample types to their MSE distances
        """
        negative_distances_all = {}
        batch_size = self.config.evaluate.batch_size
        num_samples = self.config.evaluate.num_samples
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        # Get the list of evaluations to run
        evaluations_to_run = []
        
        # Add pretrained model evaluations
        if getattr(self.config.evaluate, 'evaluate_pretrained', True):
            if getattr(self.config.evaluate, 'evaluate_ffhq1k', True) and 'ffhq1k' in self.pretrained_models:
                evaluations_to_run.append(('ffhq1k', None))
            
            if getattr(self.config.evaluate, 'evaluate_ffhq30k', True) and 'ffhq30k' in self.pretrained_models:
                evaluations_to_run.append(('ffhq30k', None))
            
            if getattr(self.config.evaluate, 'evaluate_ffhq70k_bcr', True) and 'ffhq70k-bcr' in self.pretrained_models:
                evaluations_to_run.append(('ffhq70k-bcr', None))
            
            if getattr(self.config.evaluate, 'evaluate_ffhq70k_noaug', True) and 'ffhq70k-noaug' in self.pretrained_models:
                evaluations_to_run.append(('ffhq70k-noaug', None))
        
        # Add transformations
        if getattr(self.config.evaluate, 'evaluate_transforms', True):
            # Truncation
            if getattr(self.config.evaluate, 'evaluate_truncation', True):
                evaluations_to_run.append((None, 'truncation_original'))
            if getattr(self.config.evaluate, 'evaluate_truncation_watermarked', True):
                evaluations_to_run.append((None, 'truncation_watermarked'))
            
            # Int8 Quantization
            if getattr(self.config.evaluate, 'evaluate_quantization', True):
                evaluations_to_run.append((None, 'quantization_original'))
            if getattr(self.config.evaluate, 'evaluate_quantization_watermarked', True):
                evaluations_to_run.append((None, 'quantization_watermarked'))
            
            # Int4 Quantization
            if getattr(self.config.evaluate, 'evaluate_quantization_int4', True):
                evaluations_to_run.append((None, 'quantization_int4_original'))
            if getattr(self.config.evaluate, 'evaluate_quantization_int4_watermarked', True):
                evaluations_to_run.append((None, 'quantization_int4_watermarked'))
            
            # Int2 Quantization
            if getattr(self.config.evaluate, 'evaluate_quantization_int2', True):
                evaluations_to_run.append((None, 'quantization_int2_original'))
            if getattr(self.config.evaluate, 'evaluate_quantization_int2_watermarked', True):
                evaluations_to_run.append((None, 'quantization_int2_watermarked'))
            
            # Downsampling
            if getattr(self.config.evaluate, 'evaluate_downsample', True):
                evaluations_to_run.append((None, 'downsample_original'))
            if getattr(self.config.evaluate, 'evaluate_downsample_watermarked', True):
                evaluations_to_run.append((None, 'downsample_watermarked'))
            
            # JPEG compression
            if getattr(self.config.evaluate, 'evaluate_jpeg', True):
                evaluations_to_run.append((None, 'jpeg_original'))
            if getattr(self.config.evaluate, 'evaluate_jpeg_watermarked', True):
                evaluations_to_run.append((None, 'jpeg_watermarked'))
        
        # Run evaluations
        total_evals = len(evaluations_to_run)
        if self.rank == 0 and total_evals > 0:
            logging.info(f"Running {total_evals} negative sample evaluations...")
        
        with torch.no_grad():
            for idx, (model_name, transformation) in enumerate(evaluations_to_run):
                key = model_name if model_name else transformation
                
                # Skip if we've already evaluated this configuration
                if key in negative_distances_all:
                    continue
                
                distances_per_batch = []
                
                # Process in batches
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_samples)
                    
                    # Extract batch of latent vectors
                    z = all_z[start_idx:end_idx]
                    
                    # Process the negative sample batch
                    batch_results = self._process_negative_sample_batch(z, model_name, transformation)
                    
                    if 'negative_mse_distances' not in batch_results:
                        if self.rank == 0:
                            logging.error(f"Missing MSE distances in batch results for {key}")
                        continue
                    
                    distances_per_batch.append(batch_results['negative_mse_distances'])
                
                # Combine results
                if distances_per_batch:
                    negative_distances_all[key] = np.concatenate(distances_per_batch)
                
                # Progress reporting
                if self.rank == 0 and (idx+1) % max(1, total_evals//5) == 0:
                    logging.info(f"Completed {idx+1}/{total_evals} negative sample evaluations")
        
        return negative_distances_all
    
    def visualize_samples(self):
        """
        Generate and visualize a set of samples with their watermark keys.
        """
        if self.rank != 0:
            return  # Only perform visualization on the master process
        
        # Skip visualization if disabled
        if not getattr(self.config.evaluate, 'enable_visualization', False):
            logging.info("Skipping visualization (disabled by configuration)")
            return
        
        # Create main visualization directory
        vis_dir = os.path.join(self.config.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate fixed latent vectors for visualization
        np.random.seed(self.config.evaluate.visualization_seed)
        num_vis_samples = self.config.evaluate.num_vis_samples
        z_vis = torch.from_numpy(
            np.random.randn(num_vis_samples, self.latent_dim)
        ).float().to(self.device)
        
        # Only visualize what's explicitly enabled
        # Visualize the watermarked model samples
        vis_subdir = os.path.join(vis_dir, "watermarked")
        self.visualize_model_samples(output_subdir=vis_subdir, z_vis=z_vis)
        
        # Visualize the pretrained model samples if enabled
        evaluate_pretrained = getattr(self.config.evaluate, 'evaluate_pretrained', False)
        if evaluate_pretrained and getattr(self.config.evaluate, 'visualize_pretrained', False):
            for model_name in self.pretrained_models:
                # Only visualize explicitly enabled models
                model_enabled = getattr(self.config.evaluate, f'evaluate_{model_name.replace("-", "_")}', False)
                model_viz_enabled = getattr(self.config.evaluate, f'visualize_{model_name.replace("-", "_")}', False)
                
                if model_enabled and model_viz_enabled:
                    vis_subdir = os.path.join(vis_dir, f"pretrained_{model_name}")
                    self.visualize_model_samples(model_name=model_name, output_subdir=vis_subdir, z_vis=z_vis)
        
        # Visualize transformations if enabled
        evaluate_transforms = getattr(self.config.evaluate, 'evaluate_transforms', False)
        if evaluate_transforms and getattr(self.config.evaluate, 'visualize_transforms', False):
            # For each transformation, only visualize if both evaluation and visualization are enabled
            
            # Truncation
            if self._should_visualize_transform('truncation'):
                vis_subdir = os.path.join(vis_dir, "truncation_original")
                self.visualize_model_samples(transformation='truncation_original', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
                
            if self._should_visualize_transform('truncation_watermarked'):
                vis_subdir = os.path.join(vis_dir, "truncation_watermarked")
                self.visualize_model_samples(transformation='truncation_watermarked', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
            
            # Int8 Quantization
            if self._should_visualize_transform('quantization'):
                vis_subdir = os.path.join(vis_dir, "quantization_original")
                self.visualize_model_samples(transformation='quantization_original', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
                
            if self._should_visualize_transform('quantization_watermarked'):
                vis_subdir = os.path.join(vis_dir, "quantization_watermarked")
                self.visualize_model_samples(transformation='quantization_watermarked', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
            
            # Int4 Quantization
            if self._should_visualize_transform('quantization_int4'):
                vis_subdir = os.path.join(vis_dir, "quantization_int4_original")
                self.visualize_model_samples(transformation='quantization_int4_original', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
                
            if self._should_visualize_transform('quantization_int4_watermarked'):
                vis_subdir = os.path.join(vis_dir, "quantization_int4_watermarked")
                self.visualize_model_samples(transformation='quantization_int4_watermarked', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
            
            # Int2 Quantization
            if self._should_visualize_transform('quantization_int2'):
                vis_subdir = os.path.join(vis_dir, "quantization_int2_original")
                self.visualize_model_samples(transformation='quantization_int2_original', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
                
            if self._should_visualize_transform('quantization_int2_watermarked'):
                vis_subdir = os.path.join(vis_dir, "quantization_int2_watermarked")
                self.visualize_model_samples(transformation='quantization_int2_watermarked', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
            
            # Downsampling
            if self._should_visualize_transform('downsample'):
                vis_subdir = os.path.join(vis_dir, "downsample_original")
                self.visualize_model_samples(transformation='downsample_original', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
                
            if self._should_visualize_transform('downsample_watermarked'):
                vis_subdir = os.path.join(vis_dir, "downsample_watermarked")
                self.visualize_model_samples(transformation='downsample_watermarked', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
            
            # JPEG compression
            if self._should_visualize_transform('jpeg'):
                vis_subdir = os.path.join(vis_dir, "jpeg_original")
                self.visualize_model_samples(transformation='jpeg_original', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
                
            if self._should_visualize_transform('jpeg_watermarked'):
                vis_subdir = os.path.join(vis_dir, "jpeg_watermarked")
                self.visualize_model_samples(transformation='jpeg_watermarked', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
        
        logging.info("Visualization complete")

    def _should_visualize_transform(self, transform_name):
        """
        Helper method to check if a transformation should be visualized.
        
        Args:
            transform_name (str): The name of the transformation to check.
            
        Returns:
            bool: True if the transformation should be visualized, False otherwise.
        """
        # Check if evaluation is enabled for this transform
        eval_enabled = getattr(self.config.evaluate, f'evaluate_{transform_name}', False)
        
        # Check if visualization is enabled for this transform
        viz_enabled = getattr(self.config.evaluate, f'visualize_{transform_name}', False)
        
        # Only visualize if both flags are True
        return eval_enabled and viz_enabled
    
    def visualize_model_samples(self, model_name=None, transformation=None, output_subdir="watermarked", z_vis=None):
        """
        Generate and visualize a set of samples for a specific model or transformation.
        
        Args:
            model_name (str, optional): Name of the pretrained model to use.
            transformation (str, optional): Name of the transformation to apply.
            output_subdir (str): Subdirectory to save visualizations in.
            z_vis (torch.Tensor, optional): Fixed latent vectors for visualization.
        """
        # Create visualization directory
        os.makedirs(output_subdir, exist_ok=True)
        
        # Get number of samples for visualization
        num_vis_samples = self.config.evaluate.num_vis_samples
        
        # Generate latent vectors if not provided
        if z_vis is None:
            np.random.seed(self.config.evaluate.visualization_seed)
            z_vis = torch.from_numpy(
                np.random.randn(num_vis_samples, self.latent_dim)
            ).float().to(self.device)
        
        with torch.no_grad():
            # Determine which model to use for comparison
            if model_name is not None or transformation is not None:
                # Negative sample case
                batch_result = self._process_negative_sample_batch(z_vis, model_name, transformation)
                
                # Extract images and features
                x_neg = batch_result['x_neg']
                features_neg = batch_result['features_neg']
                
                # Convert to numpy for visualization
                x_neg_np = x_neg.detach().cpu().numpy()
                
                # Get the watermarked model images for reference comparison
                w_water = self.watermarked_model.mapping(z_vis, None)
                x_water = self.watermarked_model.synthesis(w_water, noise_mode="const")
                
                # Also extract original images for comparison
                w_orig = self.gan_model.mapping(z_vis, None)
                x_orig = self.gan_model.synthesis(w_orig, noise_mode="const")
                
                # Extract features
                if self.use_image_pixels:
                    features_water = self.extract_image_partial(x_water)
                    features_orig = self.extract_image_partial(x_orig)
                else:
                    # Extract latent features
                    if w_water.ndim == 3:
                        w_water_single = w_water[:, 0, :]
                        w_orig_single = w_orig[:, 0, :]
                    else:
                        w_water_single = w_water
                        w_orig_single = w_orig
                    
                    # Convert indices to tensor if needed
                    if isinstance(self.latent_indices, np.ndarray):
                        latent_indices = torch.tensor(self.latent_indices, dtype=torch.long, device=self.device)
                    else:
                        latent_indices = self.latent_indices
                    
                    features_water = w_water_single[:, latent_indices]
                    features_orig = w_orig_single[:, latent_indices]
                
                # Generate true key using watermarked model features
                true_keys = self.key_mapper(features_water)
                
                # Predict keys based on decoder mode
                if self.direct_feature_decoder:
                    # Use features directly as input to the decoder
                    pred_keys_neg_logits = self.decoder(features_neg)
                    pred_keys_water_logits = self.decoder(features_water)
                    pred_keys_orig_logits = self.decoder(features_orig)
                else:
                    # Use the full image as input to the decoder
                    pred_keys_neg_logits = self.decoder(x_neg)
                    pred_keys_water_logits = self.decoder(x_water)
                    pred_keys_orig_logits = self.decoder(x_orig)
                
                # Convert to probabilities and binary keys
                pred_keys_neg_probs = torch.sigmoid(pred_keys_neg_logits)
                pred_keys_neg = pred_keys_neg_probs > 0.5
                
                pred_keys_water_probs = torch.sigmoid(pred_keys_water_logits)
                pred_keys_water = pred_keys_water_probs > 0.5
                
                pred_keys_orig_probs = torch.sigmoid(pred_keys_orig_logits)
                pred_keys_orig = pred_keys_orig_probs > 0.5
                
                # Save images with annotations
                for i in range(num_vis_samples):
                    # Determine match status
                    matches_reference = torch.all(pred_keys_neg[i] == true_keys[i]).item()
                    matches_watermarked = torch.all(pred_keys_water[i] == true_keys[i]).item()
                    matches_original = torch.all(pred_keys_orig[i] == true_keys[i]).item()
                    
                    # Log details
                    if getattr(self.config.evaluate, 'verbose_visualization', False):
                        logging.info(f"{output_subdir.split('/')[-1]} - Sample {i+1}:")
                        logging.info(f"  True key: {true_keys[i].detach().cpu().numpy().tolist()}")
                        logging.info(f"  Predicted key: {pred_keys_neg[i].detach().cpu().numpy().tolist()}")
                        logging.info(f"  Matches reference: {matches_reference}")
                    
                    # Save image with annotations
                    sample_filename = os.path.join(output_subdir, f"sample_{i+1}.png")
                    save_visualization(
                        image=x_neg_np[i],
                        true_key=true_keys[i].detach().cpu().numpy(),
                        pred_key=pred_keys_neg[i].detach().cpu().numpy(),
                        pred_probs=pred_keys_neg_probs[i].detach().cpu().numpy(),
                        output_path=sample_filename,
                        title=f"{output_subdir.split('/')[-1]} - Sample {i+1}",
                        match_status=matches_reference
                    )
            
            else:
                # Standard comparison case - watermarked vs original
                # Generate images from watermarked model
                w_water = self.watermarked_model.mapping(z_vis, None)
                x_water = self.watermarked_model.synthesis(w_water, noise_mode="const")
                
                # Generate images from original model
                w_orig = self.gan_model.mapping(z_vis, None)
                x_orig = self.gan_model.synthesis(w_orig, noise_mode="const")
                
                # Extract features
                if self.use_image_pixels:
                    features_water = self.extract_image_partial(x_water)
                    features_orig = self.extract_image_partial(x_orig)
                else:
                    # Extract latent features
                    if w_water.ndim == 3:
                        w_water_single = w_water[:, 0, :]
                        w_orig_single = w_orig[:, 0, :]
                    else:
                        w_water_single = w_water
                        w_orig_single = w_orig
                    
                    # Convert indices to tensor if needed
                    if isinstance(self.latent_indices, np.ndarray):
                        latent_indices = torch.tensor(self.latent_indices, dtype=torch.long, device=self.device)
                    else:
                        latent_indices = self.latent_indices
                    
                    features_water = w_water_single[:, latent_indices]
                    features_orig = w_orig_single[:, latent_indices]
                
                # Generate true key using watermarked image features
                true_keys = self.key_mapper(features_water)
                
                # Predict keys based on decoder mode
                if self.direct_feature_decoder:
                    # Use features directly as input to the decoder
                    pred_keys_water_logits = self.decoder(features_water)
                    pred_keys_orig_logits = self.decoder(features_orig)
                else:
                    # Use the full image as input to the decoder
                    pred_keys_water_logits = self.decoder(x_water)
                    pred_keys_orig_logits = self.decoder(x_orig)
                
                # Convert to probabilities and binary keys
                pred_keys_water_probs = torch.sigmoid(pred_keys_water_logits)
                pred_keys_water = pred_keys_water_probs > 0.5
                
                pred_keys_orig_probs = torch.sigmoid(pred_keys_orig_logits)
                pred_keys_orig = pred_keys_orig_probs > 0.5
                
                # Convert images to numpy for visualization
                x_water_np = x_water.detach().cpu().numpy()
                x_orig_np = x_orig.detach().cpu().numpy()
                
                # Create subdirectories for watermarked vs original
                watermarked_dir = os.path.join(output_subdir, "watermarked")
                original_dir = os.path.join(output_subdir, "original")
                os.makedirs(watermarked_dir, exist_ok=True)
                os.makedirs(original_dir, exist_ok=True)
                
                # Save images with annotations
                for i in range(num_vis_samples):
                    # Determine match status
                    matches_watermarked = torch.all(pred_keys_water[i] == true_keys[i]).item()
                    matches_original = torch.all(pred_keys_orig[i] == true_keys[i]).item()
                    
                    # Log details
                    if getattr(self.config.evaluate, 'verbose_visualization', False):
                        logging.info(f"Sample {i+1}:")
                        logging.info(f"  True key: {true_keys[i].detach().cpu().numpy().tolist()}")
                        logging.info(f"  Watermarked pred: {pred_keys_water[i].detach().cpu().numpy().tolist()} (Match: {matches_watermarked})")
                        logging.info(f"  Original pred: {pred_keys_orig[i].detach().cpu().numpy().tolist()} (Match: {matches_original})")
                    
                    # Save watermarked image with annotations
                    water_filename = os.path.join(watermarked_dir, f"sample_{i+1}.png")
                    save_visualization(
                        image=x_water_np[i],
                        true_key=true_keys[i].detach().cpu().numpy(),
                        pred_key=pred_keys_water[i].detach().cpu().numpy(),
                        pred_probs=pred_keys_water_probs[i].detach().cpu().numpy(),
                        output_path=water_filename,
                        title=f"Watermarked - Sample {i+1}",
                        match_status=matches_watermarked
                    )
                    
                    # Save original image with annotations
                    orig_filename = os.path.join(original_dir, f"sample_{i+1}.png")
                    save_visualization(
                        image=x_orig_np[i],
                        true_key=true_keys[i].detach().cpu().numpy(),
                        pred_key=pred_keys_orig[i].detach().cpu().numpy(),
                        pred_probs=pred_keys_orig_probs[i].detach().cpu().numpy(),
                        output_path=orig_filename,
                        title=f"Original - Sample {i+1}",
                        match_status=matches_original
                    )

    def calculate_threshold_at_tpr(self, watermarked_distances, target_tpr=0.95):
        """
        Calculate the threshold where target_tpr (e.g. 95%) of watermarked images have MSE distances below it.
        
        Args:
            watermarked_distances (np.ndarray): Array of MSE distances for watermarked images
            target_tpr (float): Target true positive rate (default: 0.95)
            
        Returns:
            float: Threshold value
        """
        # Sort distances in ascending order
        sorted_distances = np.sort(watermarked_distances)
        
        # Find the index corresponding to the target TPR
        idx = int(np.ceil(len(sorted_distances) * target_tpr)) - 1
        
        # Ensure index is valid
        idx = max(0, min(idx, len(sorted_distances) - 1))
        
        # Return the threshold
        return sorted_distances[idx]

    def calculate_fpr_at_threshold(self, negative_distances, threshold):
        """
        Calculate the false positive rate (FPR) for negative samples at the given threshold.
        
        Args:
            negative_distances (np.ndarray): Array of MSE distances for negative images
            threshold (float): Threshold value
            
        Returns:
            float: False positive rate (percentage of negative samples below threshold)
        """
        # Count how many negative samples are below the threshold
        below_threshold = np.sum(negative_distances <= threshold)
        
        # Calculate FPR
        fpr = (below_threshold / len(negative_distances)) * 100
        
        return fpr
    
    def evaluate(self, evaluation_mode='batch'):
        """
        Run evaluation based on the specified mode.
        
        Args:
            evaluation_mode (str): Evaluation mode, one of 'batch', 'visual', or 'both'.
                Default is 'batch' for metrics-only evaluation.
            
        Returns:
            dict: Evaluation metrics (if batch evaluation was performed).
        """
        metrics = None
        
        # Log evaluation configuration
        if self.rank == 0:
            logging.info("Starting evaluation with the following configuration:")
            logging.info(f"  Evaluation mode: {evaluation_mode}")
            logging.info(f"  Approach: {'Image-based' if self.use_image_pixels else 'Latent-based'}")
            logging.info(f"  Direct feature decoder: {self.direct_feature_decoder}")
            logging.info(f"  Direct pixel prediction: {self.direct_pixel_pred}")
            if self.use_image_pixels:
                if self.enable_multi_decoder:
                    for i, pixel_indices in enumerate(self.image_pixel_indices_list):
                        logging.info(f"  Decoder {i+1} pixel count: {len(pixel_indices)}")
                else:
                    if self.image_pixel_indices is not None:
                        logging.info(f"  Pixel count: {len(self.image_pixel_indices)}")
                    else:
                        logging.warning("  Pixel indices not initialized. This may cause issues with image-based evaluation.")
            else:
                if hasattr(self, 'latent_indices') and self.latent_indices is not None:
                    logging.info(f"  Latent indices length: {len(self.latent_indices)}")
                    if hasattr(self, 'w_partial_set_seed'):
                        logging.info(f"  Latent indices seed: {self.w_partial_set_seed}")
            
            logging.info(f"  Key length: {self.config.model.key_length}")
            logging.info(f"  Key mapper seed: {self.config.model.key_mapper_seed}")
            
            # Report visualization settings
            visualize_mode = evaluation_mode in ['visual', 'both']
            logging.info(f"  Visualization enabled: {visualize_mode}")
            if visualize_mode:
                logging.info(f"  Comparisons enabled: {getattr(self.config.evaluate, 'save_comparisons', False)}")
        
        # Initialize visualization batch counter if needed
        if self.rank == 0 and evaluation_mode in ['visual', 'both']:
            self.batch_counter = 0
        
        # Batch evaluation first if enabled
        if evaluation_mode in ['batch', 'both']:
            metrics = self.evaluate_batch()
            
            # Print metrics table if we have results and are on rank 0
            if metrics and self.rank == 0:
                # For direct pixel prediction mode, we have different metrics
                if self.direct_pixel_pred:
                    # We've already logged the main metrics in _evaluate_direct_pixel_pred_batch
                    pass
                else:
                    # Validate required metrics are present
                    required_keys = ['watermarked_mse_distances', 'original_mse_distances', 'lpips_losses']
                    missing_keys = [key for key in required_keys if key not in metrics]
                    
                    if missing_keys:
                        logging.error(f"Missing required metrics: {missing_keys}")
                        logging.error("Metrics computation may have failed. Please check the evaluation logs.")
                        return metrics
                    
                    try:
                        # Calculate threshold at 95% TPR using watermarked distances
                        watermarked_distances = metrics['watermarked_mse_distances']
                        if not isinstance(watermarked_distances, np.ndarray):
                            watermarked_distances = np.concatenate([watermarked_distances])
                            
                        threshold_95tpr = self.calculate_threshold_at_tpr(watermarked_distances, target_tpr=0.95)
                        
                        # Print header
                        logging.info("\nEvaluation Results:")
                        logging.info("-" * 100)
                        logging.info(f"{'Model/Transform':<40}{'FPR@95%TPR':>15}{'Avg MSE':>15}{'ROC-AUC':>15}{'LPIPS':>15}")
                        logging.info("-" * 100)
                        
                        # Print original model results
                        original_distances = metrics['original_mse_distances']
                        if not isinstance(original_distances, np.ndarray):
                            original_distances = np.concatenate([original_distances])
                            
                        original_fpr = self.calculate_fpr_at_threshold(original_distances, threshold_95tpr)
                        original_avg_mse = np.mean(original_distances)
                        roc_auc = metrics.get('roc_auc', 0)
                        
                        lpips_losses = metrics['lpips_losses']
                        if not isinstance(lpips_losses, np.ndarray):
                            lpips_losses = np.concatenate([lpips_losses])
                        avg_lpips = np.mean(lpips_losses)
                        
                        logging.info(f"{'Original Model':<40}{original_fpr:>15.2f}{original_avg_mse:>15.4f}{roc_auc:>15.4f}{avg_lpips:>15.4f}")
                        
                        # Print negative sample results if available
                        if 'negative_distances_all' in metrics:
                            logging.info("\nProcessing negative samples:")
                            for sample_type, distances in metrics['negative_distances_all'].items():
                                if not isinstance(distances, np.ndarray):
                                    distances = np.concatenate([distances])
                                fpr = self.calculate_fpr_at_threshold(distances, threshold_95tpr)
                                avg_mse = np.mean(distances)
                                
                                # Only log the main table row, skip the detailed stats
                                logging.info(f"{sample_type:<40}{fpr:>15.2f}{avg_mse:>15.4f}{'-':>15}{'-':>15}")
                        else:
                            logging.warning("No negative samples were evaluated. Check if negative sample evaluation is enabled.")
                        
                        logging.info("-" * 100)
                        logging.info(f"Threshold at 95% TPR: {threshold_95tpr:.4f}")
                        logging.info("-" * 100)
                        
                    except Exception as e:
                        logging.error(f"Error processing metrics: {str(e)}")
                        logging.error("Raw metrics content:", metrics)
                        logging.error(str(e), exc_info=True)
                        return metrics
        
        # Visual evaluation after batch if enabled
        if evaluation_mode in ['visual', 'both'] and self.rank == 0:
            # Use a reduced sample count for visualizations if not specified
            if not hasattr(self.config.evaluate, 'num_vis_samples'):
                self.config.evaluate.num_vis_samples = min(10, self.config.evaluate.num_samples)
            
            # Only run visualizations if explicitly requested
            self.visualize_samples()
        
        return metrics 

    def _process_negative_sample_batch(self, z, model_name=None, transformation=None):
        """
        Process a batch of latent vectors for negative sample evaluation.
        
        Args:
            z (torch.Tensor): Batch of latent vectors
            model_name (str, optional): Name of the model to use, if using a pretrained model
            transformation (str, optional): Transformation to apply, if any
        
        Returns:
            dict: Dictionary of metrics
        """
        # Special handling for direct pixel prediction mode
        if self.direct_pixel_pred:
            return self._process_negative_sample_batch_direct_pixel(z, model_name, transformation)
            
        batch_size = z.shape[0]
        
        # Generate negative sample images
        if model_name is not None:
            # Use pretrained model
            model = self.pretrained_models[model_name]
            
            if hasattr(model, 'module'):
                w = model.module.mapping(z, None)
                x = model.module.synthesis(w, noise_mode="const")
            else:
                w = model.mapping(z, None)
                x = model.synthesis(w, noise_mode="const")
        else:
            # Use transformation on watermarked or original model
            use_watermarked = "watermarked" in transformation
            
            if use_watermarked:
                if hasattr(self.watermarked_model, 'module'):
                    w = self.watermarked_model.module.mapping(z, None)
                    x = self.watermarked_model.module.synthesis(w, noise_mode="const")
                else:
                    w = self.watermarked_model.mapping(z, None)
                    x = self.watermarked_model.synthesis(w, noise_mode="const")
            else:
                w = self.gan_model.mapping(z, None)
                x = self.gan_model.synthesis(w, noise_mode="const")
            
            # Apply transformation
            x = self.apply_transformation(x, transformation)
        
        # Extract features
        if self.use_image_pixels:
            features = self.extract_image_partial(x)
        else:
            if w.ndim == 3:
                w_single = w[:, 0, :]
            else:
                w_single = w
            features = w_single[:, self.latent_indices]
        
        # Generate true key using the key mapper
        true_key = self.key_mapper(features)
        
        # Apply ZCA whitening to decoder input if enabled
        x_decoder = self.apply_zca_whitening(x) if self.use_zca_whitening else x
        
        # Predict key based on the decoder mode
        if self.direct_feature_decoder:
            # Use features directly as input to the decoder
            pred_key_logits = self.decoder(features)
        else:
            # Use the image as input
            pred_key_logits = self.decoder(x_decoder)
        
        # Get predicted probabilities
        pred_key_probs = torch.sigmoid(pred_key_logits)
        
        # Calculate distance metrics
        negative_mse_distances = torch.mean(torch.pow(pred_key_probs - true_key, 2), dim=1).cpu().numpy()
        
        return {
            'negative_mse_distances': negative_mse_distances
        }
    
    def _process_negative_sample_batch_direct_pixel(self, z, model_name=None, transformation=None):
        """
        Process a batch of latent vectors for negative sample evaluation in direct pixel prediction mode.
        This version focuses on pixel prediction accuracy rather than key detection.
        
        Args:
            z (torch.Tensor): Batch of latent vectors
            model_name (str, optional): Name of the model to use, if using a pretrained model
            transformation (str, optional): Transformation to apply, if any
        
        Returns:
            dict: Dictionary of metrics specific to direct pixel prediction
        """
        batch_size = z.shape[0]
        
        # Generate negative sample images
        if model_name is not None:
            # Use pretrained model
            model = self.pretrained_models[model_name]
            
            if hasattr(model, 'module'):
                w = model.module.mapping(z, None)
                x = model.module.synthesis(w, noise_mode="const")
            else:
                w = model.mapping(z, None)
                x = model.synthesis(w, noise_mode="const")
        else:
            # Use transformation on watermarked or original model
            use_watermarked = "watermarked" in transformation
            
            if use_watermarked:
                if hasattr(self.watermarked_model, 'module'):
                    w = self.watermarked_model.module.mapping(z, None)
                    x = self.watermarked_model.module.synthesis(w, noise_mode="const")
                else:
                    w = self.watermarked_model.mapping(z, None)
                    x = self.watermarked_model.synthesis(w, noise_mode="const")
            else:
                w = self.gan_model.mapping(z, None)
                x = self.gan_model.synthesis(w, noise_mode="const")
            
            # Apply transformation
            x = self.apply_transformation(x, transformation)
        
        # Extract features (real pixel values)
        features = self.extract_image_partial(x)
        true_values = features
        
        # Apply ZCA whitening to decoder input if enabled
        x_decoder = self.apply_zca_whitening(x) if self.use_zca_whitening else x
        
        # Predict pixel values
        pred_values = self.decoder(x_decoder)
        
        # Calculate MSE distances for each sample
        negative_mse_distances = torch.mean(torch.pow(pred_values - true_values, 2), dim=1).cpu().numpy()
        
        return {
            'negative_mse_distances': negative_mse_distances
        } 

    def apply_transformation(self, x, transformation):
        """
        Apply a transformation to the input images.
        
        Args:
            x (torch.Tensor): Input images
            transformation (str): Transformation name
        
        Returns:
            torch.Tensor: Transformed images
        """
        if 'truncation' in transformation:
            # For truncation, we need to regenerate the images with truncation
            use_watermarked = "watermarked" in transformation
            source_model = self.watermarked_model if use_watermarked else self.gan_model
            truncation_psi = getattr(self.config.evaluate, 'truncation_psi', 2.0)
            
            # Generate new latents for the batch
            batch_size = x.size(0)
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            
            # Apply truncation during generation
            if hasattr(source_model, 'module'):
                w = source_model.module.mapping(z, None, truncation_psi=truncation_psi)
                x_trunc = source_model.module.synthesis(w, noise_mode="const")
            else:
                w = source_model.mapping(z, None, truncation_psi=truncation_psi)
                x_trunc = source_model.synthesis(w, noise_mode="const")
            return x_trunc
            
        elif 'quantization' in transformation:
            use_watermarked = "watermarked" in transformation
            precision = 'int8'
            
            if 'int4' in transformation:
                precision = 'int4'
            elif 'int2' in transformation:
                precision = 'int2'
            
            # Use the pre-quantized models
            if use_watermarked:
                model = self.quantized_watermarked_models[precision]
            else:
                model = self.quantized_models[precision]
            
            # Generate new images using the quantized model
            batch_size = x.size(0)
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            
            if hasattr(model, 'module'):
                w = model.module.mapping(z, None)
                x_quant = model.module.synthesis(w, noise_mode="const")
            else:
                w = model.mapping(z, None)
                x_quant = model.synthesis(w, noise_mode="const")
            return x_quant
                
        elif 'downsample' in transformation:
            downsample_size = getattr(self.config.evaluate, 'downsample_size', 128)
            return downsample_and_upsample(x, downsample_size)
                
        elif 'jpeg' in transformation:
            jpeg_quality = getattr(self.config.evaluate, 'jpeg_quality', 55)
            return apply_jpeg_compression(x, quality=jpeg_quality)
                
        else:
            if self.rank == 0:
                logging.warning(f"Unknown transformation: {transformation}")
            return x