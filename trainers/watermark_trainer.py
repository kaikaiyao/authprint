"""
Trainer for StyleGAN watermarking.
"""
import logging
import time
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
        
        # Parse latent indices for partial vector
        if isinstance(self.config.model.selected_indices, str):
            self.latent_indices = [int(idx) for idx in self.config.model.selected_indices.split(',')]
        else:
            self.latent_indices = self.config.model.selected_indices
        
        logging.info(f"Using latent partial indices: {self.latent_indices}")
    
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
        
        # Freeze the pretrained model (for LPIPS computation)
        self.gan_model.eval()
        for param in self.gan_model.parameters():
            param.requires_grad = False
        
        # Clone the pretrained model to obtain a trainable watermarked model
        self.watermarked_model = clone_model(self.gan_model)
        self.watermarked_model.to(self.device)
        self.watermarked_model.train()
        
        # Wrap with DDP if in distributed training
        if self.world_size > 1:
            self.watermarked_model = DDP(self.watermarked_model, device_ids=[self.local_rank])
        
        # Initialize decoder model
        self.decoder = Decoder(
            image_size=self.config.model.img_size,
            channels=3,
            output_dim=self.config.model.key_length
        ).to(self.device)
        
        if self.world_size > 1:
            self.decoder = DDP(self.decoder, device_ids=[self.local_rank])
        
        # Initialize key mapper
        self.key_mapper = KeyMapper(
            input_dim=len(self.latent_indices),
            output_dim=self.config.model.key_length
        ).to(self.device)
        
        # Set up LPIPS loss function
        self.lpips_loss_fn = lpips.LPIPS(net='alex').to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            list(self.watermarked_model.parameters()) + list(self.decoder.parameters()),
            lr=self.config.training.lr
        )
    
    def validate_indices(self) -> None:
        """
        Validate latent indices to ensure they are valid.
        """
        latent_dim = self.gan_model.z_dim
        
        if max(self.latent_indices) >= latent_dim:
            raise ValueError(f"latent_indices contains indices >= latent_dim ({latent_dim}). "
                             f"Max index: {max(self.latent_indices)}")
        
        if len(self.latent_indices) != len(set(self.latent_indices)):
            logging.warning("latent_indices contains duplicate indices which may not be intended")
    
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
        
        # Obtain w latent vector using the mapping network of the watermarked model
        # Access through .module if it's a DDP-wrapped model
        if hasattr(self.watermarked_model, 'module'):
            w = self.watermarked_model.module.mapping(z, None)
        else:
            w = self.watermarked_model.mapping(z, None)
            
        # If w is replicated for each synthesis layer (shape: [batch_size, num_ws, w_dim]), take the first as representative
        if w.ndim == 3:
            w_single = w[:, 0, :]
        else:
            w_single = w

        # Generate watermarked image using the synthesis network with the full w vector
        if hasattr(self.watermarked_model, 'module'):
            x_water = self.watermarked_model.module.synthesis(w, noise_mode="const")
        else:
            x_water = self.watermarked_model.synthesis(w, noise_mode="const")

        # Compute the original image using the frozen pretrained model for LPIPS loss
        with torch.no_grad():
            w_orig = self.gan_model.mapping(z, None)
            if w_orig.ndim == 3:
                w_orig_single = w_orig[:, 0, :]
            else:
                w_orig_single = w_orig
            x_orig = self.gan_model.synthesis(w_orig, noise_mode="const")

        # Extract a partial w vector using the fixed indices from w_single
        w_partial = w_single[:, self.latent_indices]  # shape: (batch_size, len(latent_indices))
        true_key = self.key_mapper(w_partial)  # shape: (batch_size, key_length) with binary values

        # Predict key from watermarked image using the decoder (raw logits)
        pred_key_logits = self.decoder(x_water)

        # Convert predicted logits to binary for match rate calculation
        pred_key_binary = (torch.sigmoid(pred_key_logits) > 0.5).float()
        # Calculate exact match rate (percentage of samples where all bits match)
        key_matches = (pred_key_binary == true_key).all(dim=1).float().mean().item()
        match_rate = key_matches * 100  # Convert to percentage

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
            'match_rate': match_rate
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
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        if isinstance(self.watermarked_model, DDP):
            self.watermarked_model.module.load_state_dict(checkpoint['watermarked_model_state'])
        else:
            self.watermarked_model.load_state_dict(checkpoint['watermarked_model_state'])
            
        if isinstance(self.decoder, DDP):
            self.decoder.module.load_state_dict(checkpoint['decoder_state'])
        else:
            self.decoder.load_state_dict(checkpoint['decoder_state'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Load training state
        self.global_step = checkpoint['global_step']
        self.start_iteration = checkpoint['iteration'] + 1  # Start from next iteration
        
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
                    elapsed = time.time() - start_time
                    logging.info(
                        f"Iteration [{iteration}/{self.config.training.total_iterations}] "
                        f"Key Loss: {metrics['key_loss']:.4f}, LPIPS Loss: {metrics['lpips_loss']:.4f}, "
                        f"Total Loss: {metrics['total_loss']:.4f}, Match Rate: {metrics['match_rate']:.2f}%, "
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
                        optimizer=self.optimizer,
                        metrics=metrics,
                        global_step=self.global_step  # Add global_step to checkpoint
                    )
            
            # Final checkpoint if not already saved
            if self.config.training.total_iterations % self.config.training.checkpoint_interval != 0:
                save_checkpoint(
                    iteration=self.config.training.total_iterations,
                    watermarked_model=self.watermarked_model,
                    decoder=self.decoder,
                    output_dir=self.config.output_dir,
                    rank=self.rank,
                    optimizer=self.optimizer,
                    metrics=metrics,
                    global_step=self.global_step  # Add global_step to checkpoint
                )
                
        except Exception as e:
            logging.error(f"Error in training: {str(e)}", exc_info=True)
            raise 