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
        
        # Use provided indices or fall back to self.image_pixel_indices
        indices = pixel_indices if pixel_indices is not None else self.image_pixel_indices
        
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
        
        # Load checkpoint
        checkpoint = load_checkpoint(
            checkpoint_path=self.config.checkpoint_path,
            watermarked_model=None,
            decoder=None,
            optimizer=None,
            key_mapper=None,
            device=self.device
        )
        
        # Load watermarked model from checkpoint
        self.watermarked_model = clone_model(self.gan_model)
        self.watermarked_model.load_state_dict(checkpoint['watermarked_model'])
        self.watermarked_model.eval()
        self.watermarked_model.to(self.device)
        
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
        
        # Initialize or load decoder
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
                    decoder = Decoder(
                        image_size=self.config.model.img_size,
                        channels=3,
                        output_dim=key_length
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
                    watermarked_model=self.watermarked_model if i == 0 else None,  # Only load watermarked model from first checkpoint
                    decoder=decoder,
                    key_mapper=None,  # Set to None to skip loading key mapper state
                    device=self.device
                )
                
                self.decoders.append(decoder)
                self.key_mappers.append(key_mapper)
        else:
            # Standard single decoder setup
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
                self.decoder = Decoder(
                    image_size=self.config.model.img_size,
                    channels=3,
                    output_dim=self.config.model.key_length
                ).to(self.device)
            self.decoder.eval()
            
            # Initialize key mapper with specified seed
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
            
            # Load checkpoint (excluding key mapper) - do this before setting up quantized models
            if self.rank == 0:
                logging.info(f"Loading checkpoint from {self.config.checkpoint_path}...")
            load_checkpoint(
                checkpoint_path=self.config.checkpoint_path,
                watermarked_model=self.watermarked_model,
                decoder=self.decoder,
                key_mapper=None,  # Set to None to skip loading key mapper state
                device=self.device
            )
        
        # Initialize LPIPS loss with the same network as in training
        self.lpips_loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.lpips_loss_fn.eval()  # Ensure it's in eval mode
        
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
        Process a batch of latent vectors for evaluation.
        
        Args:
            z (torch.Tensor): Batch of latent vectors.
            model_name (str, optional): Name of the pretrained model to use.
            transformation (str, optional): Name of the transformation to apply to the original model/images.
        
        Returns:
            dict: Dictionary containing evaluation results.
        """
        start_time = time.time()
        
        if self.rank == 0 and self.enable_timing:
            logging.info(f"Starting batch processing: model={model_name}, transform={transformation}")
            
        with torch.no_grad():
            try:
                # Determine if this is a comparison case (original evaluation) or a negative sample case
                is_comparison_case = model_name is None and transformation is None
                
                if self.rank == 0 and self.enable_timing:
                    gen_start = time.time()
                    logging.info("Generating images and computing features...")
                
                # For negative cases, we only need to evaluate the specific model/transformation
                if not is_comparison_case:
                    result = self._process_negative_sample_batch(z, model_name, transformation)
                    if self.rank == 0 and self.enable_timing:
                        logging.info(f"Negative sample processing completed in {time.time() - gen_start:.2f}s")
                    return result
                
                # Original Model
                w_orig = self.gan_model.mapping(z, None)
                x_orig = self.gan_model.synthesis(w_orig, noise_mode="const")
                
                if self.rank == 0 and self.enable_timing:
                    logging.info(f"Original model generation completed in {time.time() - gen_start:.2f}s")
                    water_start = time.time()
                
                # Watermarked Model
                w_water = self.watermarked_model.mapping(z, None)
                x_water = self.watermarked_model.synthesis(w_water, noise_mode="const")
                
                if self.rank == 0 and self.enable_timing:
                    logging.info(f"Watermarked model generation completed in {time.time() - water_start:.2f}s")
                    feat_start = time.time()
                
                # Apply ZCA whitening if enabled and requested
                if transformation == "zca_whitening":
                    x_water = self.apply_zca_whitening(x_water)
                
                # Calculate difference images only if needed
                save_comparisons = getattr(self.config.evaluate, 'save_comparisons', False)
                diff_images = (x_water - x_orig) if save_comparisons else None
                
                if self.enable_multi_decoder:
                    # Process with multiple decoders
                    watermarked_mse_distances_all = []
                    original_mse_distances_all = []
                    
                    for i, (decoder, key_mapper) in enumerate(zip(self.decoders, self.key_mappers)):
                        # Extract features based on approach
                        if self.use_image_pixels:
                            pixel_indices = self.image_pixel_indices_list[i]
                            features_water = self.extract_image_partial(x_water, pixel_indices)
                            features_orig = self.extract_image_partial(x_orig, pixel_indices)
                        else:
                            if w_water.ndim == 3:
                                w_water_single = w_water[:, 0, :]
                                w_orig_single = w_orig[:, 0, :]
                            else:
                                w_water_single = w_water
                                w_orig_single = w_orig
                            
                            features_water = w_water_single.index_select(1, self.latent_indices)
                            features_orig = w_orig_single.index_select(1, self.latent_indices)
                        
                        # Generate true key and predictions
                        true_key = key_mapper(features_water)
                        
                        if self.direct_feature_decoder:
                            pred_key_water_logits = decoder(features_water)
                            pred_key_orig_logits = decoder(features_orig)
                        else:
                            pred_key_water_logits = decoder(x_water)
                            pred_key_orig_logits = decoder(x_orig)
                        
                        # Convert to probabilities
                        pred_key_water_probs = torch.sigmoid(pred_key_water_logits)
                        pred_key_orig_probs = torch.sigmoid(pred_key_orig_logits)
                        
                        # Calculate metrics
                        watermarked_mse_distance = torch.mean(torch.pow(pred_key_water_probs - true_key, 2), dim=1)
                        original_mse_distance = torch.mean(torch.pow(pred_key_orig_probs - true_key, 2), dim=1)
                        
                        watermarked_mse_distances_all.append(watermarked_mse_distance)
                        original_mse_distances_all.append(original_mse_distance)
                    
                    # Stack all distances and take max along decoder dimension
                    watermarked_mse_distances = torch.stack(watermarked_mse_distances_all, dim=1)
                    original_mse_distances = torch.stack(original_mse_distances_all, dim=1)
                    
                    watermarked_mse_distance = torch.max(watermarked_mse_distances, dim=1)[0]
                    original_mse_distance = torch.max(original_mse_distances, dim=1)[0]
                    
                else:
                    # Standard single decoder processing
                    if self.use_image_pixels:
                        features_water = self.extract_image_partial(x_water)
                        features_orig = self.extract_image_partial(x_orig)
                    else:
                        if w_water.ndim == 3:
                            w_water_single = w_water[:, 0, :]
                            w_orig_single = w_orig[:, 0, :]
                        else:
                            w_water_single = w_water
                            w_orig_single = w_orig
                        
                        features_water = w_water_single.index_select(1, self.latent_indices)
                        features_orig = w_orig_single.index_select(1, self.latent_indices)
                    
                    # Generate true key and predictions
                    true_key = self.key_mapper(features_water)
                    
                    if self.direct_feature_decoder:
                        pred_key_water_logits = self.decoder(features_water)
                        pred_key_orig_logits = self.decoder(features_orig)
                    else:
                        pred_key_water_logits = self.decoder(x_water)
                        pred_key_orig_logits = self.decoder(x_orig)
                    
                    # Convert to probabilities
                    pred_key_water_probs = torch.sigmoid(pred_key_water_logits)
                    pred_key_orig_probs = torch.sigmoid(pred_key_orig_logits)
                    
                    # Calculate metrics
                    watermarked_mse_distance = torch.mean(torch.pow(pred_key_water_probs - true_key, 2), dim=1)
                    original_mse_distance = torch.mean(torch.pow(pred_key_orig_probs - true_key, 2), dim=1)
                
                if self.rank == 0 and self.enable_timing:
                    logging.info(f"Feature extraction and key prediction completed in {time.time() - feat_start:.2f}s")
                    metric_start = time.time()
                
                # Calculate LPIPS loss
                lpips_losses = self.lpips_loss_fn(x_orig, x_water).squeeze()
                
                if self.rank == 0 and self.enable_timing:
                    logging.info(f"Metric calculation completed in {time.time() - metric_start:.2f}s")
                    logging.info(f"Total batch processing time: {time.time() - start_time:.2f}s")
                
                return {
                    'watermarked_mse_distances': watermarked_mse_distance.cpu().numpy(),
                    'original_mse_distances': original_mse_distance.cpu().numpy(),
                    'batch_size': z.size(0),
                    'lpips_losses': lpips_losses.cpu().numpy(),
                    'x_orig': x_orig,
                    'x_water': x_water,
                    'features_water': features_water,
                    'features_orig': features_orig,
                    'diff_images': diff_images
                }
                
            except Exception as e:
                if self.rank == 0:
                    logging.error(f"Error in batch processing: {str(e)}")
                    logging.error(str(e), exc_info=True)
                return self._get_empty_batch_result(z.size(0))
    
    def _process_negative_sample_batch(self, z, model_name=None, transformation=None):
        """
        Process a batch specifically for negative sample evaluation.
        """
        start_time = time.time()
        
        if self.rank == 0 and self.enable_timing:
            logging.info(f"Starting negative sample processing: model={model_name}, transform={transformation}")
        
        try:
            # Cache watermarked reference
            if self.rank == 0 and self.enable_timing:
                ref_start = time.time()
            
            w_water_ref = self.watermarked_model.mapping(z, None)
            x_water_ref = self.watermarked_model.synthesis(w_water_ref, noise_mode="const")
            
            if self.rank == 0 and self.enable_timing:
                logging.info(f"Reference generation completed in {time.time() - ref_start:.2f}s")
                neg_start = time.time()
            
            # Process negative samples
            if model_name is not None:
                negative_model = self.pretrained_models.get(model_name, self.gan_model)
                w_neg = negative_model.mapping(z, None)
                x_neg = negative_model.synthesis(w_neg, noise_mode="const")
            else:
                is_watermarked = '_watermarked' in transformation
                base_model = self.watermarked_model if is_watermarked else self.gan_model
                transformation_base = transformation.replace('_watermarked', '').replace('_original', '')
                
                if self.rank == 0 and self.enable_timing:
                    logging.info(f"Applying transformation: {transformation_base}")
                
                # Apply appropriate transformation
                if transformation_base == 'truncation':
                    truncation_psi = getattr(self.config.evaluate, 'truncation_psi', 2.0)
                    x_neg, w_neg = apply_truncation(base_model, z, truncation_psi, return_w=True)
                elif transformation_base.startswith('quantization'):
                    precision = 'int8' if transformation_base == 'quantization' else transformation_base.split('_')[1]
                    model_dict = self.quantized_watermarked_models if is_watermarked else self.quantized_models
                    quantized_model = model_dict.get(precision, base_model)
                    w_neg = quantized_model.mapping(z, None)
                    x_neg = quantized_model.synthesis(w_neg, noise_mode="const")
                elif transformation_base == 'downsample':
                    w_neg = base_model.mapping(z, None)
                    x_orig = base_model.synthesis(w_neg, noise_mode="const")
                    downsample_size = getattr(self.config.evaluate, 'downsample_size', 128)
                    x_neg = downsample_and_upsample(x_orig, downsample_size)
                elif transformation_base == 'jpeg':
                    w_neg = base_model.mapping(z, None)
                    x_orig = base_model.synthesis(w_neg, noise_mode="const")
                    jpeg_quality = getattr(self.config.evaluate, 'jpeg_quality', 75)
                    x_neg = apply_jpeg_compression(x_orig, jpeg_quality)
                else:
                    w_neg = base_model.mapping(z, None)
                    x_neg = base_model.synthesis(w_neg, noise_mode="const")
            
            if self.rank == 0 and self.enable_timing:
                logging.info(f"Negative sample generation completed in {time.time() - neg_start:.2f}s")
                feat_start = time.time()
            
            if self.enable_multi_decoder:
                # Process with multiple decoders
                negative_mse_distances_all = []
                
                for i, (decoder, key_mapper) in enumerate(zip(self.decoders, self.key_mappers)):
                    # Extract features based on approach
                    if self.use_image_pixels:
                        pixel_indices = self.image_pixel_indices_list[i]
                        features_water = self.extract_image_partial(x_water_ref, pixel_indices)
                        features_neg = self.extract_image_partial(x_neg, pixel_indices)
                    else:
                        if w_water_ref.ndim == 3:
                            w_water_single = w_water_ref[:, 0, :]
                            w_neg_single = w_neg[:, 0, :]
                        else:
                            w_water_single = w_water_ref
                            w_neg_single = w_neg
                        
                        features_water = w_water_single.index_select(1, self.latent_indices)
                        features_neg = w_neg_single.index_select(1, self.latent_indices)
                    
                    # Generate true key and predictions
                    true_key = key_mapper(features_water)
                    
                    if self.direct_feature_decoder:
                        pred_key_neg_logits = decoder(features_neg)
                    else:
                        pred_key_neg_logits = decoder(x_neg)
                    
                    pred_key_neg_probs = torch.sigmoid(pred_key_neg_logits)
                    negative_mse_distance = torch.mean(torch.pow(pred_key_neg_probs - true_key, 2), dim=1)
                    negative_mse_distances_all.append(negative_mse_distance)
                
                # Stack all distances and take max along decoder dimension
                negative_mse_distances = torch.stack(negative_mse_distances_all, dim=1)
                negative_mse_distance = torch.max(negative_mse_distances, dim=1)[0]
            else:
                # Standard single decoder processing
                if self.use_image_pixels:
                    features_water = self.extract_image_partial(x_water_ref)
                    features_neg = self.extract_image_partial(x_neg)
                else:
                    if w_water_ref.ndim == 3:
                        w_water_single = w_water_ref[:, 0, :]
                        w_neg_single = w_neg[:, 0, :]
                    else:
                        w_water_single = w_water_ref
                        w_neg_single = w_neg
                    features_water = w_water_single.index_select(1, self.latent_indices)
                    features_neg = w_neg_single.index_select(1, self.latent_indices)
                
                # Generate true key and predictions
                true_key = self.key_mapper(features_water)
                
                if self.direct_feature_decoder:
                    pred_key_neg_logits = self.decoder(features_neg)
                else:
                    pred_key_neg_logits = self.decoder(x_neg)
                
                pred_key_neg_probs = torch.sigmoid(pred_key_neg_logits)
                negative_mse_distance = torch.mean(torch.pow(pred_key_neg_probs - true_key, 2), dim=1)
            
            if self.rank == 0 and self.enable_timing:
                logging.info(f"Feature extraction and prediction completed in {time.time() - feat_start:.2f}s")
                logging.info(f"Total negative sample processing time: {time.time() - start_time:.2f}s")
            
            return {
                'negative_mse_distances': negative_mse_distance.cpu().numpy(),
                'batch_size': z.size(0),
                'x_neg': x_neg,
                'features_neg': features_neg
            }
            
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error in negative sample processing: {str(e)}")
                logging.error(str(e), exc_info=True)
            return self._get_empty_negative_batch_result(z.size(0))

    def _get_empty_batch_result(self, batch_size):
        """Helper method to return empty batch results on error."""
        return {
            'watermarked_mse_distances': np.ones(batch_size) * 0.5,
            'original_mse_distances': np.ones(batch_size) * 0.5,
            'batch_size': batch_size,
            'lpips_losses': np.ones(batch_size) * 0.5,
            'x_orig': torch.zeros((batch_size, 3, self.config.model.img_size, self.config.model.img_size), device=self.device),
            'x_water': torch.zeros((batch_size, 3, self.config.model.img_size, self.config.model.img_size), device=self.device),
            'features_water': torch.zeros((batch_size, self.image_pixel_count if self.use_image_pixels else len(self.latent_indices)), device=self.device),
            'features_orig': torch.zeros((batch_size, self.image_pixel_count if self.use_image_pixels else len(self.latent_indices)), device=self.device),
            'diff_images': None
        }

    def _get_empty_negative_batch_result(self, batch_size):
        """Helper method to return empty negative batch results on error."""
        return {
            'negative_mse_distances': np.ones(batch_size) * 0.5,
            'batch_size': batch_size,
            'x_neg': torch.zeros((batch_size, 3, self.config.model.img_size, self.config.model.img_size), device=self.device),
            'features_neg': torch.zeros((batch_size, self.image_pixel_count if self.use_image_pixels else len(self.latent_indices)), device=self.device)
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
            
            # Process comparison batches (original vs watermarked)
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

    def _evaluate_negative_samples(self, all_z):
        """
        Evaluate all negative sample types (pretrained models and transformations).
        This is extracted to a separate method for better organization.
        
        Args:
            all_z (torch.Tensor): All latent vectors to use for evaluation.
            
        Returns:
            dict: Dictionary mapping negative sample types to their distances.
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
                    current_batch_size = end_idx - start_idx
                    
                    # Extract batch of latent vectors
                    z = all_z[start_idx:end_idx]
                    
                    # Process batch with the specific model/transformation
                    batch_results = self._process_negative_sample_batch(z, model_name, transformation)
                    
                    # Accumulate distances
                    distances_per_batch.append(batch_results['negative_mse_distances'])
                
                # Combine results
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
            if self.use_image_pixels:
                if self.enable_multi_decoder:
                    for i, pixel_indices in enumerate(self.image_pixel_indices_list):
                        logging.info(f"  Decoder {i+1} pixel count: {len(pixel_indices)}")
                else:
                    logging.info(f"  Pixel count: {len(self.image_pixel_indices)}")
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