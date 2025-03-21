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
from models.decoder import Decoder
from models.key_mapper import KeyMapper
from models.model_utils import clone_model, load_stylegan2_model
from utils.checkpoint import save_checkpoint, load_checkpoint


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
        
        # Flag to control watermarked model freezing
        self.freeze_watermarked_model = getattr(self.config.training, 'freeze_watermarked_model', False)
        if self.freeze_watermarked_model and self.rank == 0:
            logging.info("Watermarked model parameters will be frozen during training")
        
        # Handle selected indices based on the approach
        self.use_image_pixels = self.config.model.use_image_pixels
        
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
            self.watermarked_model = DDP(self.watermarked_model, device_ids=[self.local_rank], 
                                        find_unused_parameters=self.freeze_watermarked_model)
        
        # Initialize decoder model
        self.decoder = Decoder(
            image_size=self.config.model.img_size,
            channels=3,
            output_dim=self.config.model.key_length
        ).to(self.device)
        
        if self.world_size > 1:
            self.decoder = DDP(self.decoder, device_ids=[self.local_rank])
        
        # Initialize key mapper based on the approach
        if self.use_image_pixels:
            # Generate pixel indices if using image-based approach
            self._generate_pixel_indices()
            input_dim = self.image_pixel_count  # Number of selected pixels
        else:
            input_dim = len(self.latent_indices)  # Number of selected latent dimensions
            
        self.key_mapper = KeyMapper(
            input_dim=input_dim,
            output_dim=self.config.model.key_length,
            seed=self.config.model.key_mapper_seed
        ).to(self.device)
        
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

        # Extract features based on the approach
        if self.use_image_pixels:
            # Extract pixel values from the watermarked image
            features = self.extract_image_partial(x_water)
        else:
            # Extract latent features from w (original approach)
            if w.ndim == 3:
                w_single = w[:, 0, :]
            else:
                w_single = w
            features = w_single[:, self.latent_indices]
        
        # Generate true key using the key mapper
        true_key = self.key_mapper(features)  # shape: (batch_size, key_length) with binary values

        # Predict key from watermarked image using the decoder (raw logits)
        pred_key_logits = self.decoder(x_water)
        
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
            if hasattr(self.watermarked_model, 'module'):
                for param in self.watermarked_model.module.parameters():
                    param.requires_grad = False
                self.watermarked_model.module.eval()
            else:
                for param in self.watermarked_model.parameters():
                    param.requires_grad = False
                self.watermarked_model.eval()
                
            if self.rank == 0:
                logging.info("Re-applied freezing to watermarked model after loading checkpoint")
        
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
            
            start_time = time.time()
            
            # Main training loop - start from self.start_iteration for resuming
            for iteration in range(self.start_iteration, self.config.training.total_iterations + 1):
                # Run training iteration
                metrics = self.train_iteration()
                self.global_step += 1
                
                # Log progress
                if self.global_step % self.config.training.log_interval == 0 and self.rank == 0:
                    approach = "image-based" if self.use_image_pixels else "latent-based"
                    frozen_str = " (frozen watermarked model)" if self.freeze_watermarked_model else ""
                    elapsed = time.time() - start_time
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
                        global_step=self.global_step
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
                    global_step=self.global_step
                )
                
        except Exception as e:
            logging.error(f"Error in training: {str(e)}", exc_info=True)
            raise 