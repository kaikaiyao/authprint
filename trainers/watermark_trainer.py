"""
Trainer for StyleGAN watermarking.
"""
import logging
import time
from typing import Dict

import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from config.default_config import Config
from models.decoder import Decoder
from models.model_utils import load_stylegan2_model
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
        self.decoder = None
        
        # Initialize optimizer
        self.optimizer = None
        
        # Track training progress
        self.global_step = 0
        self.start_iteration = 1  # Track starting iteration for resuming
        
        # Image pixel selection parameters
        self.image_pixel_indices = None
        self.image_pixel_count = self.config.model.image_pixel_count
        self.image_pixel_set_seed = self.config.model.image_pixel_set_seed
        logging.info(f"Using image-based approach with {self.image_pixel_count} pixels and seed {self.image_pixel_set_seed}")
    
    def setup_models(self) -> None:
        """
        Set up all models needed for training.
        """
        # Load StyleGAN2 model
        if self.rank == 0:
            logging.info("Loading StyleGAN2 model...")
        self.gan_model = load_stylegan2_model(
            self.config.model.stylegan2_url,
            self.config.model.stylegan2_local_path,
            self.device
        )
        
        # Initialize decoder
        decoder_output_dim = self.image_pixel_count  # For direct pixel prediction
        self.decoder = Decoder(
            image_size=self.config.model.img_size,
            channels=3,
            output_dim=decoder_output_dim
        ).to(self.device)
        
        if self.rank == 0:
            logging.info(f"Initialized Decoder with output_dim={decoder_output_dim}")
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.decoder.parameters(),
            lr=self.config.training.lr
        )
        if self.rank == 0:
            logging.info("Optimizer initialized with decoder parameters")
        
        # Wrap models in DDP if using distributed training
        if self.world_size > 1:
            self.decoder = DDP(
                self.decoder,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
            if self.rank == 0:
                logging.info("Models wrapped in DistributedDataParallel")
    
    def _generate_pixel_indices(self) -> None:
        """
        Generate indices for selecting pixels from the image.
        """
        # Set random seed for reproducibility
        torch.manual_seed(self.image_pixel_set_seed)
        
        # Calculate total number of pixels
        total_pixels = self.config.model.img_size * self.config.model.img_size * 3  # RGB images
        
        # Generate random indices
        self.image_pixel_indices = torch.randperm(total_pixels)[:self.image_pixel_count]
        
        if self.rank == 0:
            logging.info(f"Generated {self.image_pixel_count} pixel indices with seed {self.image_pixel_set_seed}")
            logging.info(f"Selected pixel indices: {self.image_pixel_indices.tolist()}")
    
    def validate_indices(self) -> None:
        """
        Validate that indices have been generated.
        """
        if self.image_pixel_indices is None:
            self._generate_pixel_indices()
    
    def extract_image_partial(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract selected pixels from images.
        
        Args:
            images (torch.Tensor): Input images [batch_size, channels, height, width].
            
        Returns:
            torch.Tensor: Selected pixels [batch_size, num_pixels].
        """
        # Ensure indices are generated
        self.validate_indices()
        
        # Flatten images and extract selected pixels
        batch_size = images.size(0)
        flattened = images.view(batch_size, -1)
        return flattened[:, self.image_pixel_indices]
    
    def train_iteration(self) -> Dict[str, float]:
        """
        Run a single training iteration.
        
        Returns:
            Dict[str, float]: Dictionary containing training metrics.
        """
        # Generate random latent vectors
        z = torch.randn(self.config.training.batch_size, self.gan_model.z_dim, device=self.device)
        
        # Generate images
        if hasattr(self.gan_model, 'module'):
            w = self.gan_model.module.mapping(z, None)
            x = self.gan_model.module.synthesis(w, noise_mode="const")
        else:
            w = self.gan_model.mapping(z, None)
            x = self.gan_model.synthesis(w, noise_mode="const")
        
        # Extract features (real pixel values)
        features = self.extract_image_partial(x)
        true_values = features
        
        # Use the image as input to the decoder
        pred_values = self.decoder(x)
        
        # Compute MSE loss between predicted values and actual pixel values
        key_loss = torch.mean(torch.pow(pred_values - true_values, 2))
        
        # Calculate distance metrics between true values and predicted values
        mse_distance = torch.mean(torch.pow(pred_values - true_values, 2), dim=1)
        mse_distance_mean = mse_distance.mean().item()
        mse_distance_std = mse_distance.std().item()
        
        # Optimize
        self.optimizer.zero_grad()
        key_loss.backward()
        self.optimizer.step()
        
        return {
            'key_loss': key_loss.item(),
            'mse_distance_mean': mse_distance_mean,
            'mse_distance_std': mse_distance_std
        }
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        if self.rank == 0:
            logging.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint
        load_checkpoint(
            checkpoint_path=checkpoint_path,
            decoder=self.decoder,
            optimizer=self.optimizer,
            device=self.device
        )
        
        if self.rank == 0:
            logging.info("Checkpoint loaded successfully")
    
    def train(self) -> None:
        """
        Main training loop.
        """
        # Set up models
        self.setup_models()
        
        # Resume from checkpoint if specified
        if self.config.checkpoint_path:
            self.load_checkpoint(self.config.checkpoint_path)
        
        # Training loop
        if self.rank == 0:
            logging.info("Starting training...")
            start_time = time.time()
        
        for iteration in range(self.start_iteration, self.config.training.total_iterations + 1):
            # Run training iteration
            metrics = self.train_iteration()
            
            # Update global step
            self.global_step = iteration
            
            # Log progress
            if self.rank == 0 and iteration % self.config.training.log_interval == 0:
                elapsed = time.time() - start_time
                logging.info(
                    f"Iteration {iteration}/{self.config.training.total_iterations} "
                    f"[{elapsed:.2f}s] "
                    f"Key Loss: {metrics['key_loss']:.6f} "
                    f"MSE: {metrics['mse_distance_mean']:.6f} ± {metrics['mse_distance_std']:.6f}"
                )
            
            # Save checkpoint
            if self.rank == 0 and iteration % self.config.training.checkpoint_interval == 0:
                save_checkpoint(
                    iteration=iteration,
                    decoder=self.decoder,
                    output_dir=self.config.output_dir,
                    rank=self.rank,
                    optimizer=self.optimizer,
                    metrics=metrics,
                    global_step=self.global_step
                )
        
        if self.rank == 0:
            logging.info("Training completed") 