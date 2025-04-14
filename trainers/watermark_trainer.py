"""
Trainer for StyleGAN watermarking.
"""
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import lpips
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from config.default_config import Config
from models.decoder import Decoder, FeatureDecoder
from models.key_mapper import KeyMapper
from models.model_utils import clone_model, load_stylegan2_model
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.mutual_info import estimate_mutual_information


class WatermarkTrainer:
    """
    Trainer for StyleGAN watermarking.
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
        Initialize the trainer.
        
        Args:
            config (Config): Training configuration.
            local_rank (int): Local process rank.
            rank (int): Global process rank.
            world_size (int): Total number of processes.
            device (torch.device): Device to run on.
        """
        self.config = config
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.device = device
        
        # Initialize models
        self.gan_model = None
        self.watermarked_model = None
        self.decoder = None
        self.key_mapper = None
        
        # Initialize loss functions
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.lpips_loss_fn = None
        
        # Initialize optimizer
        self.optimizer = None
        
        # Track training progress
        self.global_step = 0
        self.start_iteration = 1  # Track starting iteration for resuming
        
        # ZCA whitening parameters
        self.use_zca_whitening = getattr(self.config.model, 'use_zca_whitening', False)
        self.zca_eps = getattr(self.config.model, 'zca_eps', 1e-5)
        self.zca_batch_size = getattr(self.config.model, 'zca_batch_size', 1000)
        self.zca_mean = None
        self.zca_transform = None
        
        # Flag to control watermarked model freezing
        self.freeze_watermarked_model = getattr(self.config.training, 'freeze_watermarked_model', False)
        if self.freeze_watermarked_model and self.rank == 0:
            logging.info("Watermarked model parameters will be frozen during training")
        
        # Handle selected indices based on the approach
        self.use_image_pixels = self.config.model.use_image_pixels
        
        # Flag for direct feature decoder mode
        self.direct_feature_decoder = getattr(self.config.model, 'direct_feature_decoder', False)
        
        # Flag for direct pixel prediction mode
        self.direct_pixel_pred = getattr(self.config.model, 'direct_pixel_pred', False)
        
        if self.use_image_pixels:
            # For image-based approach, we'll generate pixel indices later
            self.image_pixel_indices = None
            self.image_pixel_count = self.config.model.image_pixel_count
            self.image_pixel_set_seed = self.config.model.image_pixel_set_seed
            logging.info(f"Using image-based approach with {self.image_pixel_count} pixels and seed {self.image_pixel_set_seed}")
        else:
            # For latent-based approach
            self.latent_indices = None
            
            # Check if explicit indices are provided
            if hasattr(self.config.model, 'selected_indices') and self.config.model.selected_indices is not None:
                if isinstance(self.config.model.selected_indices, str):
                    self.latent_indices = [int(idx) for idx in self.config.model.selected_indices.split(',')]
                else:
                    self.latent_indices = self.config.model.selected_indices
                logging.info(f"Using manually specified latent indices: {self.latent_indices}")
            else:
                # We'll generate random indices later once we know the latent dimension
                # For backward compatibility, default to 32 if w_partial_length not present
                self.w_partial_length = getattr(self.config.model, 'w_partial_length', 32)
                # Default seed for backward compatibility
                self.w_partial_set_seed = getattr(self.config.model, 'w_partial_set_seed', 42)
                logging.info(f"Will generate {self.w_partial_length} latent indices with seed {self.w_partial_set_seed}")
    
    def compute_zca_parameters(self) -> None:
        """
        Compute ZCA whitening parameters using batches of generated images.
        This is done once at the start of training using memory-efficient computation.
        """
        if not self.use_zca_whitening:
            return
            
        if self.rank == 0:
            logging.info("Computing ZCA whitening parameters...")
        
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
    
    def setup_models(self) -> None:
        """
        Initialize and set up all models.
        """
        # Load pretrained StyleGAN2 model
        self.gan_model = load_stylegan2_model(
            self.config.model.stylegan2_url,
            self.config.model.stylegan2_local_path,
            self.device
        )
        latent_dim = self.gan_model.z_dim
        logging.info(f"Pretrained model loaded with z_dim={latent_dim}")
        
        # Generate latent indices if not provided explicitly
        if not self.use_image_pixels and self.latent_indices is None:
            self._generate_latent_indices(latent_dim)
        
        # Freeze the pretrained model (for LPIPS computation)
        self.gan_model.eval()
        for param in self.gan_model.parameters():
            param.requires_grad = False
        
        # Clone the pretrained model to obtain a trainable watermarked model
        self.watermarked_model = clone_model(self.gan_model)
        self.watermarked_model.to(self.device)
        
        # Set watermarked model to train or eval mode based on freeze setting
        if self.freeze_watermarked_model:
            # Set to eval mode and freeze parameters
            self.watermarked_model.eval()
            for param in self.watermarked_model.parameters():
                param.requires_grad = False
            if self.rank == 0:
                logging.info("Watermarked model parameters frozen")
        else:
            # Normal training mode
            self.watermarked_model.train()
            if self.rank == 0:
                logging.info("Watermarked model will be trained")
        
        # Wrap with DDP if in distributed training
        if self.world_size > 1:
            # Only wrap with DDP if the model has parameters that require gradients
            if not self.freeze_watermarked_model:
                self.watermarked_model = DDP(self.watermarked_model, device_ids=[self.local_rank])
                if self.rank == 0:
                    logging.info("Wrapped watermarked model with DDP")
            else:
                if self.rank == 0:
                    logging.info("Skipping DDP for frozen watermarked model")
        
        # Initialize pixel indices for image-based approach (needed for both decoder types)
        if self.use_image_pixels:
            self._generate_pixel_indices()
        
        # Determine decoder output dimension based on mode
        if self.direct_pixel_pred:
            # For direct pixel prediction, output dimension is the number of pixels we're predicting
            decoder_output_dim = self.image_pixel_count
            if self.rank == 0:
                logging.info(f"Setting decoder output dimension to {decoder_output_dim} for direct pixel prediction")
        else:
            # Normal mode - output dimension is the binary key length
            decoder_output_dim = self.config.model.key_length
        
        # Initialize decoder model based on the mode
        if self.direct_feature_decoder or self.direct_pixel_pred:
            # Use FeatureDecoder that takes pixel features directly
            input_dim = self.image_pixel_count  # Number of selected pixels
            logging.info(f"Initializing enhanced FeatureDecoder with input_dim={input_dim}, output_dim={decoder_output_dim}")
            self.decoder = FeatureDecoder(
                input_dim=input_dim,
                output_dim=decoder_output_dim,
                hidden_dims=self.config.decoder.hidden_dims,
                activation=self.config.decoder.activation,
                dropout_rate=self.config.decoder.dropout_rate,
                num_residual_blocks=self.config.decoder.num_residual_blocks,
                use_spectral_norm=self.config.decoder.use_spectral_norm,
                use_layer_norm=self.config.decoder.use_layer_norm,
                use_attention=self.config.decoder.use_attention
            ).to(self.device)
        else:
            # Use standard Decoder that takes images
            self.decoder = Decoder(
                image_size=self.config.model.img_size,
                channels=3,
                output_dim=decoder_output_dim
            ).to(self.device)
        
        if self.world_size > 1:
            self.decoder = DDP(self.decoder, device_ids=[self.local_rank])
            if self.rank == 0:
                logging.info("Wrapped decoder with DDP")
        
        # Initialize key mapper based on the approach, skip if using direct pixel prediction
        if not self.direct_pixel_pred:
            if self.use_image_pixels:
                input_dim = self.image_pixel_count  # Number of selected pixels
            else:
                input_dim = len(self.latent_indices)  # Number of selected latent dimensions
                
            self.key_mapper = KeyMapper(
                input_dim=input_dim,
                output_dim=self.config.model.key_length,
                seed=self.config.model.key_mapper_seed,
                use_sine=getattr(self.config.model, 'key_mapper_use_sine', False),
                sensitivity=getattr(self.config.model, 'key_mapper_sensitivity', 10.0)
            ).to(self.device)
            
            if self.rank == 0:
                logging.info(f"Initialized KeyMapper with input_dim={input_dim}, output_dim={self.config.model.key_length}")
        else:
            if self.rank == 0:
                logging.info("Skipping KeyMapper initialization for direct pixel prediction mode")
            
        # Set up LPIPS loss function
        self.lpips_loss_fn = lpips.LPIPS(net='alex').to(self.device)
        
        # Initialize optimizer based on freezing setting
        if self.freeze_watermarked_model:
            # Only optimize decoder parameters
            self.optimizer = optim.Adam(
                self.decoder.parameters(),
                lr=self.config.training.lr
            )
            if self.rank == 0:
                logging.info("Optimizer initialized with decoder parameters only")
        else:
            # Optimize both watermarked model and decoder parameters (original behavior)
            self.optimizer = optim.Adam(
                list(self.watermarked_model.parameters()) + list(self.decoder.parameters()),
                lr=self.config.training.lr
            )
            if self.rank == 0:
                logging.info("Optimizer initialized with both watermarked model and decoder parameters")
        
        # After setting up models, compute ZCA parameters if needed
        if self.use_zca_whitening:
            self.compute_zca_parameters()
            
        # Now that models are set up, validate the direct pixel prediction mode
        if self.direct_pixel_pred:
            # Ensure that direct_pixel_pred is only enabled under specific conditions
            if not (self.freeze_watermarked_model and self.use_image_pixels):
                self.direct_pixel_pred = False
                if self.rank == 0:
                    logging.warning("direct_pixel_pred can only be enabled when both freeze_watermarked_model=True and use_image_pixels=True. Disabling it.")
            elif self.rank == 0:
                logging.info("Using direct pixel prediction mode: decoder will be trained to predict selected pixel values directly")
    
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
            logging.warning(f"Requested {self.image_pixel_count} pixels exceeds total pixels {total_pixels}. Using all pixels.")
            self.image_pixel_count = total_pixels
            self.image_pixel_indices = np.arange(total_pixels)
        else:
            self.image_pixel_indices = np.random.choice(
                total_pixels, 
                size=self.image_pixel_count, 
                replace=False
            )
        logging.info(f"Generated {len(self.image_pixel_indices)} pixel indices with seed {self.image_pixel_set_seed}")
        # Add detailed logging of the actual indices
        if self.rank == 0:
            logging.info(f"Selected pixel indices: {self.image_pixel_indices.tolist()}")
    
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
            logging.warning(f"Requested {self.w_partial_length} indices exceeds latent dimension {latent_dim}. Using all dimensions.")
            self.w_partial_length = latent_dim
            self.latent_indices = np.arange(latent_dim)
        else:
            self.latent_indices = np.random.choice(
                latent_dim, 
                size=self.w_partial_length, 
                replace=False
            )
        logging.info(f"Generated {len(self.latent_indices)} latent indices with seed {self.w_partial_set_seed}")
    
    def validate_indices(self) -> None:
        """
        Validate latent indices to ensure they are valid.
        """
        if not self.use_image_pixels and self.latent_indices is not None:
            latent_dim = self.gan_model.z_dim
            
            if max(self.latent_indices) >= latent_dim:
                raise ValueError(f"latent_indices contains indices >= latent_dim ({latent_dim}). "
                                f"Max index: {max(self.latent_indices)}")
            
            if len(self.latent_indices) != len(set(self.latent_indices)):
                logging.warning("latent_indices contains duplicate indices which may not be intended")
    
    def extract_image_partial(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract partial image using selected pixel indices.
        
        Args:
            images (torch.Tensor): Batch of images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Batch of flattened pixel values at selected indices
        """
        batch_size = images.shape[0]
        
        # Flatten the spatial dimensions: [batch_size, channels*height*width]
        flattened = images.reshape(batch_size, -1)
        
        # Get values at selected indices: [batch_size, pixel_count]
        image_partial = flattened[:, self.image_pixel_indices]
        
        return image_partial
    
    def train_iteration(self) -> Dict[str, float]:
        """
        Run a single training iteration.
        
        Returns:
            dict: Dictionary of metrics for this iteration.
        """
        self.optimizer.zero_grad()
        
        # Get latent dimension from the model
        latent_dim = self.gan_model.z_dim
        
        # Sample a batch of latent vectors z: shape (batch_size, latent_dim)
        z = torch.randn(self.config.training.batch_size, latent_dim, device=self.device)
        
        # Generate watermarked image
        with torch.no_grad() if self.freeze_watermarked_model else torch.enable_grad():
            if hasattr(self.watermarked_model, 'module'):
                w = self.watermarked_model.module.mapping(z, None)
                x_water = self.watermarked_model.module.synthesis(w, noise_mode="const")
            else:
                w = self.watermarked_model.mapping(z, None)
                x_water = self.watermarked_model.synthesis(w, noise_mode="const")

        # Compute the original image using the frozen pretrained model for LPIPS loss
        with torch.no_grad():
            w_orig = self.gan_model.mapping(z, None)
            x_orig = self.gan_model.synthesis(w_orig, noise_mode="const")

        # Extract features based on the approach (using original non-whitened image)
        if self.use_image_pixels:
            features = self.extract_image_partial(x_water)
        else:
            if w.ndim == 3:
                w_single = w[:, 0, :]
            else:
                w_single = w
            features = w_single[:, self.latent_indices]
        
        # Handle training differently based on whether we're using direct pixel prediction
        if self.direct_pixel_pred:
            # In direct pixel prediction mode, the features ARE the targets
            true_values = features
            
            # Apply ZCA whitening to decoder input if enabled
            x_water_decoder = self.apply_zca_whitening(x_water) if self.use_zca_whitening else x_water
            
            # Predict pixel values
            pred_values = self.decoder(x_water_decoder)
            
            # Compute MSE loss between predicted values and actual pixel values
            key_loss = torch.mean(torch.pow(pred_values - true_values, 2))
            
            # Calculate distance metrics between true values and predicted values
            mse_distance = torch.mean(torch.pow(pred_values - true_values, 2), dim=1)
            mse_distance_mean = mse_distance.mean().item()
            mse_distance_std = mse_distance.std().item()
            
            # Mean absolute error (MAE) distance
            mae_distance = torch.mean(torch.abs(pred_values - true_values), dim=1)
            mae_distance_mean = mae_distance.mean().item()
            mae_distance_std = mae_distance.std().item()
            
            # We can't calculate match rate with continuous values, so set to 0
            match_rate = 0.0
            
        else:
            # Normal watermarking mode - get raw activations for logging during first iteration
            if self.global_step == 0 and self.rank == 0:
                with torch.no_grad():
                    # Get first 3 samples
                    sample_features = features[:3]
                    
                    # Get projections, activations, and binary keys
                    projections, activations, binary_keys = self.key_mapper.get_raw_and_binary(sample_features)
                    
                    logging.info("Example outputs during first training iteration:")
                    for i in range(min(3, len(activations))):
                        logging.info(f"  Training sample {i+1}:")
                        
                        # Only log a subset of feature values if there are many
                        feature_str = str(sample_features[i].tolist())
                        if len(feature_str) > 100:  # Truncate if too long
                            # Show just the first few and last few elements
                            feature_values = sample_features[i].tolist()
                            feature_str = str(feature_values[:3]) + " ... " + str(feature_values[-3:])
                            feature_str += f" (total length: {len(feature_values)})"
                        
                        logging.info(f"    Input features:  {feature_str}")
                        logging.info(f"    Pre-activation:  {projections[i].tolist()}")
                        logging.info(f"    Raw activations: {activations[i].tolist()}")
                        logging.info(f"    Binary key:      {binary_keys[i].tolist()}")
            
            # Generate true key using the key mapper (using original non-whitened features)
            true_key = self.key_mapper(features)

            # Apply ZCA whitening to decoder input if enabled
            x_water_decoder = self.apply_zca_whitening(x_water) if self.use_zca_whitening else x_water
            
            # Predict key based on the decoder mode
            if self.direct_feature_decoder:
                # Use features directly as input to the decoder
                pred_key_logits = self.decoder(features)
            else:
                # Use the whitened watermarked image as input
                pred_key_logits = self.decoder(x_water_decoder)
            
            # Get predicted probabilities (before thresholding)
            pred_key_probs = torch.sigmoid(pred_key_logits)

            # Convert predicted logits to binary for match rate calculation
            pred_key_binary = (pred_key_probs > 0.5).float()
            # Calculate exact match rate (percentage of samples where all bits match)
            key_matches = (pred_key_binary == true_key).all(dim=1).float().mean().item()
            match_rate = key_matches * 100  # Convert to percentage
            
            # Calculate distance metrics between true key and predicted probabilities
            # Mean squared error (MSE) distance - range [0, 1]
            mse_distance = torch.mean(torch.pow(pred_key_probs - true_key, 2), dim=1)
            mse_distance_mean = mse_distance.mean().item()
            mse_distance_std = mse_distance.std().item()
            
            # Mean absolute error (MAE) distance - range [0, 1]
            mae_distance = torch.mean(torch.abs(pred_key_probs - true_key), dim=1)
            mae_distance_mean = mae_distance.mean().item()
            mae_distance_std = mae_distance.std().item()

            # Compute key loss (BCE with logits)
            key_loss = self.bce_loss_fn(pred_key_logits, true_key)
        
        # Compute LPIPS loss between original and watermarked images
        # Skip LPIPS calculation in direct feature decoder mode as we're not trying to optimize the image
        if self.direct_feature_decoder or self.direct_pixel_pred:
            lpips_loss = torch.tensor(0.0, device=self.device)
            total_loss = key_loss  # Only key loss matters in direct modes
        else:
            lpips_loss = self.lpips_loss_fn(x_orig, x_water).mean()
            total_loss = key_loss + self.config.training.lambda_lpips * lpips_loss

        total_loss.backward()
        self.optimizer.step()
        
        # Return metrics
        return {
            'key_loss': key_loss.item(),
            'lpips_loss': lpips_loss.item(),
            'total_loss': total_loss.item(),
            'match_rate': match_rate,
            'mse_distance_mean': mse_distance_mean,
            'mse_distance_std': mse_distance_std,
            'mae_distance_mean': mae_distance_mean,
            'mae_distance_std': mae_distance_std
        }
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training state from a checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        # Setup models first if they haven't been initialized
        if self.gan_model is None:
            self.setup_models()
            self.validate_indices()
        
        # Load checkpoint using the utility function
        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            watermarked_model=self.watermarked_model,
            decoder=self.decoder,
            optimizer=self.optimizer,
            key_mapper=self.key_mapper,
            device=self.device
        )
        
        # Set training state
        self.global_step = checkpoint['global_step']
        self.start_iteration = checkpoint['iteration'] + 1  # Start from next iteration
        
        # Re-freeze watermarked model if needed (in case checkpoint was from normal training)
        if self.freeze_watermarked_model:
            # First, make sure we're not dealing with a DDP model
            watermarked_model = self.watermarked_model
            if hasattr(watermarked_model, 'module'):
                # If the model is a DDP model, we need to unwrap it first
                if self.rank == 0:
                    logging.info("Unwrapping DDP from watermarked model before freezing")
                self.watermarked_model = watermarked_model.module
                # Set to device explicitly after unwrapping
                self.watermarked_model.to(self.device)
            
            # Now freeze the model
            for param in self.watermarked_model.parameters():
                param.requires_grad = False
            self.watermarked_model.eval()
                
            if self.rank == 0:
                logging.info("Re-applied freezing to watermarked model after loading checkpoint")
                
            # Recreate optimizer to ensure it only includes decoder parameters
            self.optimizer = optim.Adam(
                self.decoder.parameters(),
                lr=self.config.training.lr
            )
            if self.rank == 0:
                logging.info("Optimizer recreated with decoder parameters only")
        
        if self.rank == 0:
            logging.info(f"Successfully loaded checkpoint from iteration {checkpoint['iteration']}")
            logging.info(f"Resuming training from iteration {self.start_iteration}")
    
    def train(self) -> None:
        """
        Run the training loop.
        """
        try:
            # Setup models if not already done (in case of checkpoint loading)
            if self.gan_model is None:
                self.setup_models()
                self.validate_indices()
            
            # Estimate mutual information if enabled
            if self.config.model.estimate_mutual_info and self.use_image_pixels:
                if self.rank == 0:
                    logging.info("Estimating mutual information between selected pixels and full images...")
                
                # Generate samples for MI estimation
                n_samples = self.config.model.mi_n_samples
                batch_size = min(16, n_samples)  # Use smaller batches to avoid memory issues
                features_list = []
                images_list = []
                
                with torch.no_grad():
                    for i in range(0, n_samples, batch_size):
                        current_batch_size = min(batch_size, n_samples - i)
                        # Generate random latents
                        z = torch.randn(current_batch_size, self.gan_model.z_dim, device=self.device)
                        
                        # Generate images
                        if hasattr(self.watermarked_model, 'module'):
                            w = self.watermarked_model.module.mapping(z, None)
                            x = self.watermarked_model.module.synthesis(w, noise_mode="const")
                        else:
                            w = self.watermarked_model.mapping(z, None)
                            x = self.watermarked_model.synthesis(w, noise_mode="const")
                        
                        # Extract features
                        features = self.extract_image_partial(x)
                        
                        features_list.append(features)
                        images_list.append(x)
                        
                        if self.rank == 0 and i % 100 == 0:
                            logging.info(f"Generated {i + current_batch_size}/{n_samples} samples for MI estimation")
                
                # Concatenate all samples
                features = torch.cat(features_list, dim=0)
                images = torch.cat(images_list, dim=0)
                
                # Estimate mutual information
                mutual_info, h_features, h_images, h_joint, h_features_given_images = estimate_mutual_information(
                    features=features,
                    images=images,
                    n_samples=n_samples,
                    k=self.config.model.mi_k_neighbors,
                    device=self.device
                )
                
                if self.rank == 0:
                    logging.info(f"Mutual Information Estimation Results:")
                    logging.info(f"  I(features; images) = {mutual_info:.4f} bits")
                    logging.info(f"  H(features) = {h_features:.4f} bits")
                    logging.info(f"  H(images) = {h_images:.4f} bits")
                    logging.info(f"  H(features, images) = {h_joint:.4f} bits")
                    logging.info(f"  H(features|images) = {h_features_given_images:.4f} bits")
                    logging.info(f"  Normalized MI = {mutual_info / min(h_features, h_images):.4f}")
            
            start_time = time.time()
            
            # Ensure watermarked model is in the correct mode
            if self.freeze_watermarked_model:
                # Ensure the model is in eval mode when frozen
                if hasattr(self.watermarked_model, 'module'):
                    self.watermarked_model.module.eval()
                else:
                    self.watermarked_model.eval()
                if self.rank == 0:
                    logging.info("Confirmed watermarked model is in eval mode")
            else:
                # Ensure the model is in training mode when not frozen
                if hasattr(self.watermarked_model, 'module'):
                    self.watermarked_model.module.train()
                else:
                    self.watermarked_model.train()
                if self.rank == 0:
                    logging.info("Confirmed watermarked model is in train mode")
            
            # Print example KeyMapper outputs at the beginning of training (only on rank 0)
            if self.start_iteration == 1 and self.rank == 0 and not self.direct_pixel_pred:
                self._log_key_mapper_examples()
            
            # If using direct pixel prediction, log example pixel values
            if self.start_iteration == 1 and self.rank == 0 and self.direct_pixel_pred:
                self._log_direct_pixel_examples()
            
            # Main training loop - start from self.start_iteration for resuming
            for iteration in range(self.start_iteration, self.config.training.total_iterations + 1):
                # Run training iteration
                metrics = self.train_iteration()
                self.global_step += 1
                
                # Log progress
                if self.global_step % self.config.training.log_interval == 0 and self.rank == 0:
                    # Determine the approach string based on configuration
                    if self.use_image_pixels:
                        if self.direct_pixel_pred:
                            approach = "direct-pixel-prediction"
                        elif self.direct_feature_decoder:
                            approach = "direct-feature-decoder"
                        else:
                            approach = "image-based"
                    else:
                        approach = "latent-based"
                    
                    frozen_str = " (frozen watermarked model)" if self.freeze_watermarked_model else ""
                    elapsed = time.time() - start_time
                    
                    # For direct pixel prediction, don't show match rate as it's not applicable
                    if self.direct_pixel_pred:
                        logging.info(
                            f"Iteration [{iteration}/{self.config.training.total_iterations}] ({approach}){frozen_str} "
                            f"Pixel MSE Loss: {metrics['key_loss']:.4f}, "
                            f"MSE Dist: {metrics['mse_distance_mean']:.4f}±{metrics['mse_distance_std']:.4f}, "
                            f"MAE Dist: {metrics['mae_distance_mean']:.4f}±{metrics['mae_distance_std']:.4f}, "
                            f"Time: {elapsed:.2f}s"
                        )
                    else:
                        logging.info(
                            f"Iteration [{iteration}/{self.config.training.total_iterations}] ({approach}){frozen_str} "
                            f"Key Loss: {metrics['key_loss']:.4f}, LPIPS Loss: {metrics['lpips_loss']:.4f}, "
                            f"Total Loss: {metrics['total_loss']:.4f}, Match Rate: {metrics['match_rate']:.2f}%, "
                            f"MSE Dist: {metrics['mse_distance_mean']:.4f}±{metrics['mse_distance_std']:.4f}, "
                            f"MAE Dist: {metrics['mae_distance_mean']:.4f}±{metrics['mae_distance_std']:.4f}, "
                            f"Time: {elapsed:.2f}s"
                        )
                
                # Save checkpoint at regular intervals
                if iteration % self.config.training.checkpoint_interval == 0:
                    save_checkpoint(
                        iteration=iteration,
                        watermarked_model=self.watermarked_model,
                        decoder=self.decoder,
                        output_dir=self.config.output_dir,
                        rank=self.rank,
                        key_mapper=self.key_mapper,
                        optimizer=self.optimizer,
                        metrics=metrics,
                        global_step=self.global_step,
                        zca_mean=self.zca_mean if self.use_zca_whitening else None,
                        whitening_factors=self.whitening_factors if self.use_zca_whitening else None
                    )
            
            # Final checkpoint if not already saved
            if self.config.training.total_iterations % self.config.training.checkpoint_interval != 0:
                save_checkpoint(
                    iteration=self.config.training.total_iterations,
                    watermarked_model=self.watermarked_model,
                    decoder=self.decoder,
                    output_dir=self.config.output_dir,
                    rank=self.rank,
                    key_mapper=self.key_mapper,
                    optimizer=self.optimizer,
                    metrics=metrics,
                    global_step=self.global_step,
                    zca_mean=self.zca_mean if self.use_zca_whitening else None,
                    whitening_factors=self.whitening_factors if self.use_zca_whitening else None
                )
                
        except Exception as e:
            logging.error(f"Error in training: {str(e)}", exc_info=True)
            raise
            
    def _log_key_mapper_examples(self) -> None:
        """
        Generate and log examples of KeyMapper output for debugging.
        Prints input features, pre-activation projections, raw activation values, 
        and binary keys for 10 random inputs.
        """
        logging.info("Generating KeyMapper example outputs...")
        
        # Get latent dimension from the model
        latent_dim = self.gan_model.z_dim
        
        # Generate 10 random latent vectors
        num_examples = 10
        torch.manual_seed(42)  # For reproducible examples
        z_examples = torch.randn(num_examples, latent_dim, device=self.device)
        
        # Process through the model to get features
        with torch.no_grad():
            if self.use_image_pixels:
                # Generate images first
                if hasattr(self.watermarked_model, 'module'):
                    w = self.watermarked_model.module.mapping(z_examples, None)
                    x_water = self.watermarked_model.module.synthesis(w, noise_mode="const")
                else:
                    w = self.watermarked_model.mapping(z_examples, None)
                    x_water = self.watermarked_model.synthesis(w, noise_mode="const")
                
                # Extract pixel values
                features = self.extract_image_partial(x_water)
            else:
                # Extract latent features
                if hasattr(self.watermarked_model, 'module'):
                    w = self.watermarked_model.module.mapping(z_examples, None)
                else:
                    w = self.watermarked_model.mapping(z_examples, None)
                
                if w.ndim == 3:
                    w_single = w[:, 0, :]
                else:
                    w_single = w
                features = w_single[:, self.latent_indices]
        
        # Use the updated helper method to get projection, activations, and binary keys
        projections, activations, binary_keys = self.key_mapper.get_raw_and_binary(features)
        
        # Log examples
        for i in range(num_examples):
            logging.info(f"Example {i+1}:")
            
            # Only log a subset of feature values if there are many
            feature_str = str(features[i].tolist())
            if len(feature_str) > 100:  # Truncate if too long
                # Show just the first few and last few elements
                feature_values = features[i].tolist()
                feature_str = str(feature_values[:3]) + " ... " + str(feature_values[-3:])
                feature_str += f" (total length: {len(feature_values)})"
            
            logging.info(f"  Input features:  {feature_str}")
            logging.info(f"  Pre-activation:  {projections[i].tolist()}")
            logging.info(f"  Raw activations: {activations[i].tolist()}")
            logging.info(f"  Binary key:      {binary_keys[i].tolist()}")

    def _log_direct_pixel_examples(self) -> None:
        """
        Generate and log examples of direct pixel prediction for debugging.
        Prints input features, predicted pixel values, and actual pixel values for 10 random inputs.
        """
        logging.info("Generating direct pixel prediction example outputs...")
        
        # Get latent dimension from the model
        latent_dim = self.gan_model.z_dim
        
        # Generate 10 random latent vectors
        num_examples = 10
        torch.manual_seed(42)  # For reproducible examples
        z_examples = torch.randn(num_examples, latent_dim, device=self.device)
        
        # Process through the model to get features
        with torch.no_grad():
            if self.use_image_pixels:
                # Generate images first
                if hasattr(self.watermarked_model, 'module'):
                    w = self.watermarked_model.module.mapping(z_examples, None)
                    x_water = self.watermarked_model.module.synthesis(w, noise_mode="const")
                else:
                    w = self.watermarked_model.mapping(z_examples, None)
                    x_water = self.watermarked_model.synthesis(w, noise_mode="const")
                
                # Extract pixel values
                features = self.extract_image_partial(x_water)
            else:
                # Extract latent features
                if hasattr(self.watermarked_model, 'module'):
                    w = self.watermarked_model.module.mapping(z_examples, None)
                else:
                    w = self.watermarked_model.mapping(z_examples, None)
                
                if w.ndim == 3:
                    w_single = w[:, 0, :]
                else:
                    w_single = w
                features = w_single[:, self.latent_indices]
        
        # Predict pixel values
        with torch.no_grad():
            if self.direct_pixel_pred:
                pred_values = self.decoder(features)
            else:
                x_water_decoder = self.apply_zca_whitening(x_water) if self.use_zca_whitening else x_water
                pred_values = self.decoder(x_water_decoder)
        
        # Log examples
        for i in range(num_examples):
            logging.info(f"Example {i+1}:")
            
            # Only log a subset of feature values if there are many
            feature_str = str(features[i].tolist())
            if len(feature_str) > 100:  # Truncate if too long
                # Show just the first few and last few elements
                feature_values = features[i].tolist()
                feature_str = str(feature_values[:3]) + " ... " + str(feature_values[-3:])
                feature_str += f" (total length: {len(feature_values)})"
            
            logging.info(f"  Input features:  {feature_str}")
            logging.info(f"  Predicted values: {pred_values[i].tolist()}")
            logging.info(f"  Actual values:   {features[i].tolist()}") 