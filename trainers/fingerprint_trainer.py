"""
Trainer for StyleGAN fingerprinting.
"""
import logging
import time
import random
from typing import Dict, List, Optional

import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset

from config.default_config import Config
from models.decoder import DecoderSD_L, DecoderSD_M, DecoderSD_S, StyleGAN2Decoder
from models.stable_diffusion_model import StableDiffusionModel
from utils.checkpoint import save_checkpoint, load_checkpoint


class FingerprintTrainer:
    """
    Trainer for StyleGAN fingerprinting.
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
        self.generative_model = None
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
        
        # Initialize prompt dataset if multi-prompt mode is enabled
        self.prompts: Optional[List[str]] = None
        if self.config.model.enable_multi_prompt:
            if self.config.model.prompt_source == "local":
                self._load_prompt_dataset_local()
            else:  # diffusiondb
                self._load_prompt_dataset_diffusiondb()
    
    def _load_prompt_dataset_local(self) -> None:
        """
        Load prompts from a local file.
        """
        if self.rank == 0:
            logging.info(f"Loading prompts from local file: {self.config.model.prompt_dataset_path}")
        
        try:
            with open(self.config.model.prompt_dataset_path, 'r', encoding='utf-8') as f:
                all_prompts = [line.strip() for line in f if line.strip()]
            
            # Sample the specified number of prompts
            if len(all_prompts) > self.config.model.prompt_dataset_size:
                self.prompts = random.sample(all_prompts, self.config.model.prompt_dataset_size)
            else:
                self.prompts = all_prompts
                if self.rank == 0:
                    logging.warning(f"Prompt dataset contains fewer prompts ({len(all_prompts)}) "
                                  f"than requested ({self.config.model.prompt_dataset_size})")
            
            if self.rank == 0:
                logging.info(f"Loaded {len(self.prompts)} prompts from local file")
                logging.info(f"Sample prompts: {self.prompts[:3]}")
        
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error loading prompt dataset: {str(e)}")
            raise
    
    def _load_prompt_dataset_diffusiondb(self) -> None:
        """
        Load prompts from DiffusionDB dataset.
        """
        if self.rank == 0:
            logging.info("Loading prompts from DiffusionDB dataset")
        
        try:
            # Load the metadata table from DiffusionDB
            subset = "2m" if self.config.model.diffusiondb_subset == "2m" else "large"
            dataset = load_dataset("poloclub/diffusiondb", subset, split="train")
            
            # Extract all unique prompts
            all_prompts = list(set(dataset["prompt"]))
            
            # Sample the specified number of prompts
            if len(all_prompts) > self.config.model.prompt_dataset_size:
                self.prompts = random.sample(all_prompts, self.config.model.prompt_dataset_size)
            else:
                self.prompts = all_prompts
                if self.rank == 0:
                    logging.warning(f"DiffusionDB contains fewer unique prompts ({len(all_prompts)}) "
                                  f"than requested ({self.config.model.prompt_dataset_size})")
            
            if self.rank == 0:
                logging.info(f"Loaded {len(self.prompts)} prompts from DiffusionDB")
                logging.info(f"Sample prompts: {self.prompts[:3]}")
        
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error loading DiffusionDB dataset: {str(e)}")
            raise
    
    def _sample_prompts(self, batch_size: int) -> List[str]:
        """
        Sample prompts for the current batch.
        
        Args:
            batch_size (int): Number of prompts to sample.
            
        Returns:
            List[str]: List of sampled prompts.
        """
        if not self.config.model.enable_multi_prompt or not self.prompts:
            return [self.config.model.sd_prompt] * batch_size
        
        return random.sample(self.prompts, min(batch_size, len(self.prompts)))
    
    def setup_models(self) -> None:
        """
        Set up all models needed for training.
        """
        # Initialize generative model
        if self.rank == 0:
            logging.info(f"Loading {self.config.model.model_type} model...")
            
        model_class = self.config.model.get_model_class()
        model_kwargs = self.config.model.get_model_kwargs(self.device)
        self.generative_model = model_class(**model_kwargs)
        
        # Initialize decoder based on model type and size
        decoder_output_dim = self.image_pixel_count  # For direct pixel prediction
        
        if self.config.model.model_type == "stylegan2":
            self.decoder = StyleGAN2Decoder(
                image_size=self.config.model.img_size,
                channels=3,
                output_dim=decoder_output_dim
            ).to(self.device)
            if self.rank == 0:
                logging.info(f"Initialized StyleGAN2Decoder with output_dim={decoder_output_dim}")
        else:  # stable-diffusion
            decoder_class = {
                "S": DecoderSD_S,
                "M": DecoderSD_M,
                "L": DecoderSD_L
            }[self.config.model.sd_decoder_size]
            
            self.decoder = decoder_class(
                image_size=self.config.model.img_size,
                channels=3,
                output_dim=decoder_output_dim
            ).to(self.device)
            
            if self.rank == 0:
                logging.info(f"Initialized SD-Decoder-{self.config.model.sd_decoder_size} with output_dim={decoder_output_dim}")
        
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
        # Get generation kwargs based on model type
        gen_kwargs = self.config.model.get_generation_kwargs()
        
        # For Stable Diffusion, update prompts if multi-prompt mode is enabled
        if self.config.model.model_type == "stable-diffusion":
            prompts = self._sample_prompts(self.config.training.batch_size)
            gen_kwargs["prompt"] = prompts
        
        # Generate images
        x = self.generative_model.generate_images(
            batch_size=self.config.training.batch_size,
            device=self.device,
            **gen_kwargs
        )
            
        # Extract features (real pixel values)
        features = self.extract_image_partial(x)
        true_values = features
        
        # Get decoder (handle DDP wrapping)
        decoder = self.decoder.module if hasattr(self.decoder, 'module') else self.decoder
        
        # Get predictions
        pred_values = self.decoder(x)
        train_loss = torch.mean(torch.pow(pred_values - true_values, 2))
        
        # Optimize
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        
        # Now compute metrics in eval mode
        decoder.eval()  # Temporarily set to eval mode
        with torch.no_grad():
            pred_values = self.decoder(x)
            mse_distance = torch.mean(torch.pow(pred_values - true_values, 2), dim=1)
            mse_distance_mean = mse_distance.mean().item()
            mse_distance_std = mse_distance.std().item()
        decoder.train()  # Set back to train mode
        
        return {
            'train_loss': train_loss.item(),
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
                    f"Train Loss: {metrics['train_loss']:.6f} "
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