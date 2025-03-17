#!/usr/bin/env python
"""
Attack script for StyleGAN watermarking.
This implements a forging attack that aims to modify original non-watermarked images
so that the real decoder extracts the correct watermark keys from them.

The script supports two modes of operation:
1. Latent vector (w_partial) based approach: Uses selected dimensions from the StyleGAN latent space
   for key generation. This is the original approach.
2. Image-pixel based approach: Uses selected pixels from the generated images for key generation.
   This is an alternative approach that may be more robust in some cases.

The mode is selected using the --use_image_pixels flag. When this flag is set, the script uses
image pixels instead of latent vectors for key generation.
"""
import argparse
import logging
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
from PIL import Image
import lpips
import sklearn.metrics as skmetrics

# Add the parent directory (project root) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.default_config import Config, get_default_config
from models.decoder import Decoder
from models.key_mapper import KeyMapper
from utils.checkpoint import load_checkpoint
from utils.logging_utils import setup_logging
from models.model_utils import load_stylegan2_model, clone_model
from utils.distributed import setup_distributed, cleanup_distributed
from utils.model_loading import load_pretrained_models
from utils.image_transforms import (
    apply_truncation, 
    quantize_model_weights, 
    downsample_and_upsample, 
    apply_jpeg_compression
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Forging Attack against StyleGAN2 Watermarking - Modifies original images to extract correct watermarks")
    
    # Model configuration
    parser.add_argument("--stylegan2_url", type=str,
                        default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl",
                        help="URL for the pretrained StyleGAN2 model")
    parser.add_argument("--stylegan2_local_path", type=str,
                        default="ffhq70k-paper256-ada.pkl",
                        help="Local path to store/load the StyleGAN2 model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to watermarked model checkpoint to attack")
    parser.add_argument("--img_size", type=int, default=256, help="Image resolution")
    parser.add_argument("--key_length", type=int, default=4, help="Length of the binary key (output dimension)")
    parser.add_argument("--selected_indices", type=str, default=None,
                        help="Optional: Comma-separated list of indices for latent partial. If not provided, indices will be generated using w_partial_set_seed")
    parser.add_argument("--w_partial_set_seed", type=int, default=42,
                        help="Random seed for selecting latent indices from the w vector")
    parser.add_argument("--w_partial_length", type=int, default=32,
                        help="Number of dimensions to select from the w vector (default: 32)")
    parser.add_argument("--use_image_pixels", action="store_true", default=False,
                        help="Use image pixels for watermarking instead of latent vectors")
    parser.add_argument("--image_pixel_set_seed", type=int, default=42,
                        help="Random seed for selecting pixel indices from the generated image")
    parser.add_argument("--image_pixel_count", type=int, default=8192,
                        help="Number of pixels to select from the image (default: 8192)")
    parser.add_argument("--key_mapper_seed", type=int, default=2025, 
                        help="Specific random seed for KeyMapper initialization for reproducibility")
    
    # Attack configuration
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for attack")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to attack")
    
    # PGD attack parameters
    parser.add_argument("--pgd_alpha", type=float, nargs='+', 
                        default=[0.0001, 0.001, 0.01, 0.1, 1.0], 
                        help="PGD step sizes to try (can specify multiple values)")
    parser.add_argument("--pgd_steps", type=int, default=100, help="Number of PGD steps")
    parser.add_argument("--pgd_epsilon", type=float, default=1.0, help="Maximum PGD perturbation")
    
    # Surrogate training parameters
    parser.add_argument("--surrogate_lr", type=float, default=1e-4, help="Learning rate for surrogate decoder")
    parser.add_argument("--surrogate_batch_size", type=int, default=32, help="Batch size for surrogate training")
    parser.add_argument("--surrogate_epochs", type=int, default=1, help="Number of epochs for surrogate training")
    parser.add_argument("--surrogate_num_samples", type=int, default=10000, help="Number of samples for surrogate training")
    parser.add_argument("--num_surrogate_models", type=int, default=3, help="Number of surrogate decoders to train")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="attack_results", help="Directory to save attack results")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval for logging attack progress")
    parser.add_argument("--visualization_samples", type=int, default=5, help="Number of samples to visualize")
    
    # Other configuration
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    return parser.parse_args()


class SurrogateDecoder(nn.Module):
    """
    Simplified surrogate decoder for binary classification between original and watermarked images.
    """
    def __init__(self, image_size=256, channels=3):
        super(SurrogateDecoder, self).__init__()
        self.features = nn.Sequential(
            # Initial layer: 256 -> 128
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 -> 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 -> 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32 -> 16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16 -> 8
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8 -> 4
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Calculate the size of the flattened output
        self.feature_size = 512 * (image_size // 64) * (image_size // 64)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # Binary classification - 0: original, 1: watermarked
        )
    
    def forward(self, x):
        """Forward pass for surrogate decoder."""
        features = self.features(x)
        return self.classifier(features)


def generate_latent_indices(latent_dim, w_partial_length, w_partial_set_seed):
    """
    Generate random latent indices for latent-based approach.
    
    Args:
        latent_dim (int): Dimension of the latent space.
        w_partial_length (int): Number of dimensions to select.
        w_partial_set_seed (int): Random seed for reproducibility.
        
    Returns:
        list: List of selected latent indices.
    """
    # Set seed for reproducibility
    np.random.seed(w_partial_set_seed)
    
    # Generate random indices (without replacement)
    if w_partial_length > latent_dim:
        logging.warning(f"Requested {w_partial_length} indices exceeds latent dimension {latent_dim}. Using all dimensions.")
        w_partial_length = latent_dim
        latent_indices = np.arange(latent_dim)
    else:
        latent_indices = np.random.choice(
            latent_dim, 
            size=w_partial_length, 
            replace=False
        )
    logging.info(f"Generated {len(latent_indices)} latent indices with seed {w_partial_set_seed}")
    return latent_indices


def generate_pixel_indices(img_size, channels, image_pixel_count, image_pixel_set_seed):
    """
    Generate random pixel indices for image-based approach.
    
    Args:
        img_size (int): Image resolution.
        channels (int): Number of channels in the image.
        image_pixel_count (int): Number of pixels to select.
        image_pixel_set_seed (int): Random seed for reproducibility.
        
    Returns:
        np.ndarray: Array of selected pixel indices.
    """
    # Set seed for reproducibility
    np.random.seed(image_pixel_set_seed)
    
    # Calculate total number of pixels
    total_pixels = channels * img_size * img_size
    
    # Generate random indices (without replacement)
    if image_pixel_count > total_pixels:
        logging.warning(f"Requested {image_pixel_count} pixels exceeds total pixels {total_pixels}. Using all pixels.")
        image_pixel_count = total_pixels
        pixel_indices = np.arange(total_pixels)
    else:
        pixel_indices = np.random.choice(
            total_pixels, 
            size=image_pixel_count, 
            replace=False
        )
    logging.info(f"Generated {len(pixel_indices)} pixel indices with seed {image_pixel_set_seed}")
    return pixel_indices


def extract_image_partial(images, pixel_indices):
    """
    Extract partial image using selected pixel indices.
    
    Args:
        images (torch.Tensor): Batch of images [batch_size, channels, height, width]
        pixel_indices (np.ndarray): Array of pixel indices to select
            
    Returns:
        torch.Tensor: Batch of flattened pixel values at selected indices
    """
    batch_size = images.shape[0]
    
    # Flatten the spatial dimensions: [batch_size, channels*height*width]
    flattened = images.reshape(batch_size, -1)
    
    # Get values at selected indices: [batch_size, pixel_count]
    image_partial = flattened[:, pixel_indices]
    
    return image_partial


def train_surrogate_decoders(gan_model, watermarked_model, config, local_rank, rank, world_size, device):
    """
    Train surrogate decoder models for attack.
    
    Args:
        gan_model: The original StyleGAN2 model
        watermarked_model: The watermarked StyleGAN2 model
        config: Configuration object
        local_rank: Local process rank
        rank: Global process rank
        world_size: Total number of processes
        device: Device to run training on
        
    Returns:
        list: List of trained surrogate decoder models
    """
    # Initialize surrogate decoders
    surrogate_decoders = []
    
    # Calculate the number of batches
    batch_size = config.attack.surrogate_batch_size
    num_batches = config.attack.surrogate_num_samples // batch_size
    
    if rank == 0:
        logging.info(f"Training {config.attack.num_surrogate_models} surrogate decoders...")
    
    for model_idx in range(config.attack.num_surrogate_models):
        if rank == 0:
            logging.info(f"Training surrogate decoder {model_idx+1}/{config.attack.num_surrogate_models}")
        
        # Initialize surrogate decoder
        surrogate_decoder = SurrogateDecoder(image_size=config.model.img_size).to(device)
        
        # Use DDP if distributed training
        if world_size > 1:
            surrogate_decoder = DDP(surrogate_decoder, device_ids=[local_rank])
        
        # Initialize optimizer
        optimizer = optim.Adam(surrogate_decoder.parameters(), lr=config.attack.surrogate_lr)
        
        # Loss function - binary cross entropy
        criterion = nn.BCEWithLogitsLoss()
        
        # Data generation function
        def generate_batch():
            """Generate a batch of training data for surrogate decoder."""
            # Sample random latents
            z = torch.randn(batch_size, gan_model.z_dim, device=device)
            
            # Get original images
            with torch.no_grad():
                w_orig = gan_model.mapping(z, None)
                x_orig = gan_model.synthesis(w_orig, noise_mode="const")
                
                # Get watermarked images
                if hasattr(watermarked_model, 'module'):
                    w_water = watermarked_model.module.mapping(z, None)
                    x_water = watermarked_model.module.synthesis(w_water, noise_mode="const")
                else:
                    w_water = watermarked_model.mapping(z, None)
                    x_water = watermarked_model.synthesis(w_water, noise_mode="const")
            
            # Prepare labels: 0 for original, 1 for watermarked
            orig_labels = torch.zeros(batch_size, 1, device=device)
            water_labels = torch.ones(batch_size, 1, device=device)
            
            # Combine into batches
            x_combined = torch.cat([x_orig, x_water], dim=0)
            labels_combined = torch.cat([orig_labels, water_labels], dim=0)
            
            return x_combined, labels_combined
        
        # Train the surrogate decoder
        surrogate_decoder.train()
        for epoch in range(config.attack.surrogate_epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            # Only use tqdm on rank 0
            iterator = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{config.attack.surrogate_epochs}") if rank == 0 else range(num_batches)
            for _ in iterator:
                # Generate a batch
                x, labels = generate_batch()
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = surrogate_decoder(x)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                pred = (torch.sigmoid(outputs) > 0.5).float()
                correct += (pred == labels).sum().item()
                total += labels.size(0)
            
            # Calculate epoch metrics
            avg_loss = total_loss / num_batches
            accuracy = 100.0 * correct / total
            
            # Log metrics - only on rank 0
            if rank == 0:
                logging.info(f"Surrogate {model_idx+1} - Epoch {epoch+1}/{config.attack.surrogate_epochs}, "
                             f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Set to evaluation mode
        surrogate_decoder.eval()
        surrogate_decoders.append(surrogate_decoder)
        
        # Save the trained surrogate decoder (only on rank 0)
        if rank == 0:
            surrogate_dir = os.path.join(config.output_dir, "surrogate_decoders")
            os.makedirs(surrogate_dir, exist_ok=True)
            
            # Get state dict (handle DDP wrapping)
            if hasattr(surrogate_decoder, 'module'):
                state_dict = surrogate_decoder.module.state_dict()
            else:
                state_dict = surrogate_decoder.state_dict()
            
            # Save model
            surrogate_path = os.path.join(surrogate_dir, f"surrogate_decoder_{model_idx}.pth")
            torch.save({
                'model_state_dict': state_dict,
                'accuracy': accuracy
            }, surrogate_path)
            logging.info(f"Saved surrogate decoder {model_idx+1} to {surrogate_path}")
    
    return surrogate_decoders


def pgd_attack(images, w_partials, surrogate_decoders, real_decoder, key_mapper, config, device, pixel_indices=None, alpha=0.01):
    """
    Perform PGD attack on the images using surrogate decoders to guide the attack.
    This is a FORGING attack that aims to make the real decoder extract the correct watermark keys
    from the modified original images.
    
    Args:
        images: Original images to attack
        w_partials: Partial latent vectors corresponding to the images (used in w-partial mode)
        surrogate_decoders: List of surrogate decoder models
        real_decoder: The real decoder model
        key_mapper: The key mapper model
        config: Configuration object
        device: Device to run the attack on
        pixel_indices: Array of pixel indices to select (used in image-pixel mode)
        alpha: PGD step size (learning rate) for this attack
        
    Returns:
        dict: Attack results including statistics and attacked images
    """
    # Create a copy of the images for attack (detach to avoid gradient tracking from generation)
    attacked_images = images.clone().detach().requires_grad_(True)
    
    # Get true keys from key mapper
    if config.model.use_image_pixels:
        # Extract pixel values from images for key generation
        features = extract_image_partial(images, pixel_indices)
        true_keys = key_mapper(features)
    else:
        # Use w_partials for key generation (original approach)
        true_keys = key_mapper(w_partials)
    
    # Extract the shape of images
    batch_size = images.shape[0]
    
    # Compute image stats for normalization
    img_min = images.min()
    img_max = images.max()
    
    # Get predicted keys from real decoder before attack (for comparison)
    with torch.no_grad():
        initial_pred = real_decoder(images)
        initial_bin_pred = (torch.sigmoid(initial_pred) > 0.5).float()
        initial_match = (initial_bin_pred == true_keys).all(dim=1).float().mean().item() * 100.0
    
    # Setup optimizer (single step is handled manually in PGD)
    # We'll use the Adam optimizer for better convergence in PGD
    optimizer = optim.Adam([attacked_images], lr=alpha)  # Use the specific alpha value
    
    # PGD attack loop
    for step in range(config.attack.pgd_steps):
        optimizer.zero_grad()
        
        # Calculate average loss from all surrogate decoders
        surrogate_loss = 0
        for surrogate in surrogate_decoders:
            # We want to maximize the "watermarked" classification (labeled 1) for original images (labeled 0)
            # This means minimizing the loss between "outputs" and "target"
            outputs = surrogate(attacked_images)
            
            # Target is 1 (watermarked) for all images
            target = torch.ones(batch_size, 1, device=device)
            
            # Calculate BCE loss
            surrogate_loss += F.binary_cross_entropy_with_logits(outputs, target, reduction='mean')
        
        # Average the loss across all surrogate decoders
        surrogate_loss /= len(surrogate_decoders)
        
        # Compute gradients
        surrogate_loss.backward()

        # Update the attacked images
        optimizer.step()
        
        # Project back to epsilon ball (L-infinity norm constraint)
        with torch.no_grad():
            delta = attacked_images - images
            delta = torch.clamp(delta, -config.attack.pgd_epsilon, config.attack.pgd_epsilon)
            attacked_images.data = torch.clamp(images + delta, img_min, img_max)
    
    # Evaluate attack success on real decoder
    with torch.no_grad():
        final_pred = real_decoder(attacked_images)
        final_pred_probs = torch.sigmoid(final_pred)
        final_bin_pred = (final_pred_probs > 0.5).float()
        final_match = (final_bin_pred == true_keys).all(dim=1).float().mean().item() * 100.0
        
        # Calculate MSE and MAE distances between predicted probabilities and true keys
        # Mean squared error (MSE) distance - range [0, 1]
        mse_distance = torch.mean(torch.pow(final_pred_probs - true_keys, 2), dim=1)
        mse_distance_mean = mse_distance.mean().item()
        mse_distance_std = mse_distance.std().item()
        
        # Mean absolute error (MAE) distance - range [0, 1]
        mae_distance = torch.mean(torch.abs(final_pred_probs - true_keys), dim=1)
        mae_distance_mean = mae_distance.mean().item()
        mae_distance_std = mae_distance.std().item()
        
        # Binary classification by surrogate decoders
        surrogate_classifications = []
        for surrogate in surrogate_decoders:
            output = surrogate(attacked_images)
            # Calculate percentage of images classified as watermarked (class 1)
            pred = (torch.sigmoid(output) > 0.5).float().mean().item() * 100.0
            surrogate_classifications.append(pred)
        
        # Calculate L2 distance between original and attacked images
        l2_distance = torch.norm(images - attacked_images, p=2, dim=(1, 2, 3)).mean().item()
        
        # Calculate LPIPS perceptual distance
        lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
        perceptual_dist = lpips_loss_fn(images, attacked_images).mean().item()
    
    # Return results
    return {
        'attacked_images': attacked_images.detach(),
        'original_images': images.detach(),
        'initial_match_rate': initial_match,
        'final_match_rate': final_match,
        'surrogate_classifications': surrogate_classifications,
        'l2_distance': l2_distance,
        'perceptual_distance': perceptual_dist,
        'key_match_rate': (final_bin_pred == true_keys).float().mean(dim=0).cpu().numpy(),  # Per-bit accuracy
        'true_keys': true_keys.cpu().numpy(),
        'predicted_keys': final_bin_pred.cpu().numpy(),
        'mse_distance_mean': mse_distance_mean,
        'mse_distance_std': mse_distance_std,
        'mae_distance_mean': mae_distance_mean,
        'mae_distance_std': mae_distance_std
    }


def calculate_asr_at_tpr(watermarked_distances, attack_distances, target_tpr=0.95):
    """
    Calculate attack success rate at a specific true positive rate.
    
    Args:
        watermarked_distances: Distances for watermarked images
        attack_distances: Distances for attacked images
        target_tpr: Target true positive rate (default: 0.95)
        
    Returns:
        tuple: (attack success rate, threshold)
    """
    # Sort watermarked distances to find threshold
    sorted_watermarked = np.sort(watermarked_distances)
    threshold_idx = int((1 - target_tpr) * len(sorted_watermarked))
    threshold = sorted_watermarked[threshold_idx]
    
    # Calculate attack success rate (percentage of attack distances below threshold)
    asr = np.mean(attack_distances <= threshold) * 100.0
    
    return asr, threshold


def run_attack(config, local_rank, rank, world_size, device):
    """
    Run attack against watermarked model.
    
    Args:
        config: Configuration object
        local_rank: Local process rank
        rank: Global process rank
        world_size: Total number of processes
        device: Device to run the attack on
        
    Returns:
        dict: Attack results for all alpha values
    """
    # Determine which mode to use
    use_image_pixels = config.model.use_image_pixels
    
    # For latent-based approach
    latent_indices = None
    if not use_image_pixels:
        # Parse latent indices for partial vector if provided
        if hasattr(config.model, 'selected_indices') and config.model.selected_indices is not None:
            if isinstance(config.model.selected_indices, str):
                latent_indices = [int(idx) for idx in config.model.selected_indices.split(',')]
            else:
                latent_indices = config.model.selected_indices
            if rank == 0:
                logging.info(f"Using manually specified latent indices: {latent_indices}")
    
    # For image-based approach
    pixel_indices = None
    if use_image_pixels:
        if rank == 0:
            logging.info(f"Using image-based approach with {config.model.image_pixel_count} pixels and seed {config.model.image_pixel_set_seed}")
    
    # Load the original StyleGAN2 model
    if rank == 0:
        logging.info("Loading StyleGAN2 model...")
    gan_model = load_stylegan2_model(
        config.model.stylegan2_url,
        config.model.stylegan2_local_path,
        device
    )
    gan_model.eval()
    
    # Generate indices if they weren't specified
    if not use_image_pixels and latent_indices is None:
        # Generate latent indices using seed and length
        latent_dim = gan_model.z_dim
        latent_indices = generate_latent_indices(
            latent_dim,
            config.model.w_partial_length,
            config.model.w_partial_set_seed
        )
    
    if use_image_pixels:
        # Generate pixel indices
        channels = 3  # RGB images
        pixel_indices = generate_pixel_indices(
            config.model.img_size,
            channels,
            config.model.image_pixel_count,
            config.model.image_pixel_set_seed
        )
    
    # Clone it to create watermarked model
    watermarked_model = clone_model(gan_model)
    watermarked_model.to(device)
    watermarked_model.eval()
    
    # Initialize decoder model
    decoder = Decoder(
        image_size=config.model.img_size,
        channels=3,
        output_dim=config.model.key_length
    ).to(device)
    decoder.eval()
    
    # Initialize key mapper with specified seed
    if rank == 0:
        logging.info(f"Initializing key mapper with seed {config.model.key_mapper_seed}")
    
    # Key mapper input dimension depends on the mode
    if use_image_pixels:
        input_dim = len(pixel_indices)
    else:
        input_dim = len(latent_indices)
        
    key_mapper = KeyMapper(
        input_dim=input_dim,
        output_dim=config.model.key_length,
        seed=config.model.key_mapper_seed
    ).to(device)
    key_mapper.eval()
    
    # Load checkpoint (excluding optimizer)
    if rank == 0:
        logging.info(f"Loading checkpoint from {config.checkpoint_path}...")
    
    load_checkpoint(
        checkpoint_path=config.checkpoint_path,
        watermarked_model=watermarked_model,
        decoder=decoder,
        key_mapper=key_mapper,
        device=device
    )
    
    # Load pretrained models if needed
    pretrained_models = {}
    if hasattr(config.evaluate, 'evaluate_pretrained') and config.evaluate.evaluate_pretrained:
        pretrained_models = load_pretrained_models(config, device, rank)
    
    # Setup quantized model if needed
    quantized_model = None
    if getattr(config.evaluate, 'evaluate_quantization', False):
        if rank == 0:
            logging.info("Setting up quantized model...")
        quantized_model = quantize_model_weights(gan_model)
        quantized_model.eval()
    
    # Step 1: Train surrogate decoders
    surrogate_decoders = train_surrogate_decoders(
        gan_model, watermarked_model, config, local_rank, rank, world_size, device
    )
    
    # Dictionary to store metrics for each alpha value and each case
    all_attack_metrics = {}
    
    # Get list of alpha values to try
    alpha_values = config.attack.pgd_alpha
    
    # Make sure we have a list even if a single value was provided
    if not isinstance(alpha_values, list):
        alpha_values = [alpha_values]
    
    if rank == 0:
        logging.info(f"Will test {len(alpha_values)} alpha values: {alpha_values}")
    
    # First collect watermarked image distances for threshold calculation
    # Use fixed 1000 samples regardless of config.attack.num_samples
    watermarked_mse_distances = []
    watermarked_mae_distances = []
    
    # Fixed number of samples for threshold computation
    num_threshold_samples = 1000
    batch_size = config.attack.batch_size
    num_batches = (num_threshold_samples + batch_size - 1) // batch_size
    
    if rank == 0:
        logging.info(f"Computing threshold using {num_threshold_samples} watermarked samples...")
    
    # Generate watermarked images and collect distances
    samples_processed = 0
    for batch_idx in range(num_batches):
        actual_batch_size = min(batch_size, num_threshold_samples - samples_processed)
        if actual_batch_size <= 0:
            break
            
        # Generate random latent vectors
        z = torch.randn(actual_batch_size, gan_model.z_dim, device=device)
        
        # Generate watermarked images
        with torch.no_grad():
            w_water = watermarked_model.mapping(z, None)
            x_water = watermarked_model.synthesis(w_water, noise_mode="const")
            
            # Extract features based on approach
            if use_image_pixels:
                features = extract_image_partial(x_water, pixel_indices)
            else:
                if w_water.ndim == 3:
                    w_water_single = w_water[:, 0, :]
                else:
                    w_water_single = w_water
                features = w_water_single[:, latent_indices]
            
            # Get true keys
            true_keys = key_mapper(features)
            
            # Get predicted keys
            pred_logits = decoder(x_water)
            pred_probs = torch.sigmoid(pred_logits)
            
            # Calculate distances
            mse_dist = torch.mean(torch.pow(pred_probs - true_keys, 2), dim=1).cpu().numpy()
            mae_dist = torch.mean(torch.abs(pred_probs - true_keys), dim=1).cpu().numpy()
            
            watermarked_mse_distances.extend(mse_dist)
            watermarked_mae_distances.extend(mae_dist)
            samples_processed += actual_batch_size
    
    # Convert to numpy arrays
    watermarked_mse_distances = np.array(watermarked_mse_distances)
    watermarked_mae_distances = np.array(watermarked_mae_distances)
    
    if rank == 0:
        logging.info(f"Computed threshold using {len(watermarked_mse_distances)} watermarked samples")
    
    # Step 2: Perform PGD attack for each alpha value and each case
    for alpha in alpha_values:
        if rank == 0:
            logging.info(f"\n===== Running attack with alpha={alpha} =====")
        
        # Dictionary to store metrics for each case with this alpha
        case_metrics = {}
        
        # First attack original images
        if rank == 0:
            logging.info("Attacking original images...")
        metrics = attack_case(
            None, None,  # No model_name or transformation
            gan_model, watermarked_model, decoder, key_mapper,
            surrogate_decoders, config, device,
            pixel_indices if use_image_pixels else None,
            latent_indices if not use_image_pixels else None,
            alpha, watermarked_mse_distances, watermarked_mae_distances
        )
        case_metrics['original'] = metrics
        
        # Attack pretrained models if enabled
        if hasattr(config.evaluate, 'evaluate_pretrained') and config.evaluate.evaluate_pretrained:
            for model_name in pretrained_models:
                if rank == 0:
                    logging.info(f"Attacking pretrained model: {model_name}...")
                metrics = attack_case(
                    model_name, None,  # model_name but no transformation
                    pretrained_models[model_name], watermarked_model, decoder, key_mapper,
                    surrogate_decoders, config, device,
                    pixel_indices if use_image_pixels else None,
                    latent_indices if not use_image_pixels else None,
                    alpha, watermarked_mse_distances, watermarked_mae_distances
                )
                case_metrics[f'pretrained_{model_name}'] = metrics
        
        # Attack transformations if enabled
        if hasattr(config.evaluate, 'evaluate_transforms') and config.evaluate.evaluate_transforms:
            # Attack truncation
            if hasattr(config.evaluate, 'evaluate_truncation') and config.evaluate.evaluate_truncation:
                if rank == 0:
                    logging.info("Attacking truncated images...")
                metrics = attack_case(
                    None, 'truncation',
                    gan_model, watermarked_model, decoder, key_mapper,
                    surrogate_decoders, config, device,
                    pixel_indices if use_image_pixels else None,
                    latent_indices if not use_image_pixels else None,
                    alpha, watermarked_mse_distances, watermarked_mae_distances
                )
                case_metrics['truncation'] = metrics
            
            # Attack quantization
            if hasattr(config.evaluate, 'evaluate_quantization') and config.evaluate.evaluate_quantization:
                if rank == 0:
                    logging.info("Attacking quantized model...")
                if quantized_model is not None:
                    metrics = attack_case(
                        None, 'quantization',
                        quantized_model, watermarked_model, decoder, key_mapper,
                        surrogate_decoders, config, device,
                        pixel_indices if use_image_pixels else None,
                        latent_indices if not use_image_pixels else None,
                        alpha, watermarked_mse_distances, watermarked_mae_distances
                    )
                    case_metrics['quantization'] = metrics
            
            # Attack downsample
            if hasattr(config.evaluate, 'evaluate_downsample') and config.evaluate.evaluate_downsample:
                if rank == 0:
                    logging.info("Attacking downsampled images...")
                metrics = attack_case(
                    None, 'downsample',
                    gan_model, watermarked_model, decoder, key_mapper,
                    surrogate_decoders, config, device,
                    pixel_indices if use_image_pixels else None,
                    latent_indices if not use_image_pixels else None,
                    alpha, watermarked_mse_distances, watermarked_mae_distances
                )
                case_metrics['downsample'] = metrics
            
            # Attack JPEG compression
            if hasattr(config.evaluate, 'evaluate_jpeg') and config.evaluate.evaluate_jpeg:
                if rank == 0:
                    logging.info("Attacking JPEG compressed images...")
                metrics = attack_case(
                    None, 'jpeg',
                    gan_model, watermarked_model, decoder, key_mapper,
                    surrogate_decoders, config, device,
                    pixel_indices if use_image_pixels else None,
                    latent_indices if not use_image_pixels else None,
                    alpha, watermarked_mse_distances, watermarked_mae_distances
                )
                case_metrics['jpeg'] = metrics
        
        # Store all case metrics for this alpha
        all_attack_metrics[alpha] = case_metrics
        
        # Print and log metrics for this alpha (only on rank 0)
        if rank == 0:
            logging.info(f"\n===== Attack Results for alpha={alpha} =====")
            for case_name, metrics in case_metrics.items():
                logging.info(f"\n--- Results for {case_name} ---")
                logging.info(f"Number of samples: {metrics['num_samples']}")
                logging.info(f"Initial match rate: {metrics['initial_match_rate']:.2f}%")
                logging.info(f"Final match rate after attack: {metrics['final_match_rate']:.2f}%")
                logging.info(f"MSE Distance: {metrics['mse_distance_mean']:.4f}±{metrics['mse_distance_std']:.4f}")
                logging.info(f"MAE Distance: {metrics['mae_distance_mean']:.4f}±{metrics['mae_distance_std']:.4f}")
                logging.info(f"ASR@95%TPR (MSE): {metrics['asr_95tpr_mse']:.2f}%")
                logging.info(f"ASR@95%TPR (MAE): {metrics['asr_95tpr_mae']:.2f}%")
                
                for i, rate in enumerate(metrics['surrogate_classifications']):
                    logging.info(f"Surrogate decoder {i+1} watermark fooling rate: {rate:.2f}%")
                
                logging.info(f"L2 distance between original and attacked images: {metrics['l2_distance']:.4f}")
                logging.info(f"LPIPS perceptual distance: {metrics['perceptual_distance']:.4f}")
    
    # Save all metrics to file (only on rank 0)
    if rank == 0:
        metrics_path = os.path.join(config.output_dir, "attack_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write("===== Attack Results Summary =====\n\n")
            
            # Create a summary table
            f.write("Alpha | Case | Match Rate | ASR@95%TPR (MSE) | ASR@95%TPR (MAE) | L2 Dist | LPIPS Dist\n")
            f.write("----- | ---- | ---------- | ---------------- | ---------------- | ------- | ----------\n")
            
            for alpha, case_metrics in all_attack_metrics.items():
                for case_name, metrics in case_metrics.items():
                    f.write(f"{alpha:.4f} | {case_name} | {metrics['final_match_rate']:.2f}% | "
                           f"{metrics['asr_95tpr_mse']:.2f}% | {metrics['asr_95tpr_mae']:.2f}% | "
                           f"{metrics['l2_distance']:.4f} | {metrics['perceptual_distance']:.4f}\n")
            
            f.write("\n\n")
            
            # Detailed results for each alpha and case
            for alpha, case_metrics in all_attack_metrics.items():
                f.write(f"\n===== Detailed Results for alpha={alpha} =====\n")
                for case_name, metrics in case_metrics.items():
                    f.write(f"\n--- Results for {case_name} ---\n")
                    f.write(f"Number of samples: {metrics['num_samples']}\n")
                    f.write(f"Initial match rate: {metrics['initial_match_rate']:.2f}%\n")
                    f.write(f"Final match rate after attack: {metrics['final_match_rate']:.2f}%\n")
                    f.write(f"MSE Distance: {metrics['mse_distance_mean']:.4f}±{metrics['mse_distance_std']:.4f}\n")
                    f.write(f"MAE Distance: {metrics['mae_distance_mean']:.4f}±{metrics['mae_distance_std']:.4f}\n")
                    f.write(f"ASR@95%TPR (MSE): {metrics['asr_95tpr_mse']:.2f}%\n")
                    f.write(f"ASR@95%TPR (MAE): {metrics['asr_95tpr_mae']:.2f}%\n")
                    
                    for i, rate in enumerate(metrics['surrogate_classifications']):
                        f.write(f"Surrogate decoder {i+1} watermark fooling rate: {rate:.2f}%\n")
                    
                    f.write(f"L2 distance between original and attacked images: {metrics['l2_distance']:.4f}\n")
                    f.write(f"LPIPS perceptual distance: {metrics['perceptual_distance']:.4f}\n")
    
    return all_attack_metrics


def attack_case(
    model_name, transformation,
    source_model, watermarked_model, decoder, key_mapper,
    surrogate_decoders, config, device,
    pixel_indices=None, latent_indices=None,
    alpha=0.01, watermarked_mse_distances=None, watermarked_mae_distances=None
):
    """
    Attack a specific case (original, pretrained model, or transformation).
    
    Args:
        model_name (str): Name of pretrained model to use (or None)
        transformation (str): Name of transformation to apply (or None)
        source_model: Model to generate images from
        watermarked_model: Watermarked model for comparison
        decoder: Decoder model
        key_mapper: Key mapper model
        surrogate_decoders: List of surrogate decoders
        config: Configuration object
        device: Device to run on
        pixel_indices: Indices for image-pixel approach
        latent_indices: Indices for latent-based approach
        alpha: PGD step size
        watermarked_mse_distances: MSE distances for watermarked images (for threshold)
        watermarked_mae_distances: MAE distances for watermarked images (for threshold)
        
    Returns:
        dict: Attack metrics for this case
    """
    # Initialize metrics
    attack_metrics = {
        'initial_match_rate': 0.0,
        'final_match_rate': 0.0,
        'surrogate_classifications': [],
        'l2_distance': 0.0,
        'perceptual_distance': 0.0,
        'mse_distance_mean': 0.0,
        'mse_distance_std': 0.0,
        'mae_distance_mean': 0.0,
        'mae_distance_std': 0.0,
        'num_samples': 0,
        'all_mse_distances': [],
        'all_mae_distances': []
    }
    
    # Batch attack
    batch_size = config.attack.batch_size
    num_batches = (config.attack.num_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        actual_batch_size = min(batch_size, config.attack.num_samples - batch_idx * batch_size)
        if actual_batch_size <= 0:
            break
        
        # Generate random latent vectors
        z = torch.randn(actual_batch_size, source_model.z_dim, device=device)
        
        # Generate images based on case
        with torch.no_grad():
            if transformation == 'truncation':
                # Apply truncation
                truncation_psi = getattr(config.evaluate, 'truncation_psi', 2.0)
                original_images, w = apply_truncation(source_model, z, truncation_psi, return_w=True)
            else:
                # Normal generation
                w = source_model.mapping(z, None)
                original_images = source_model.synthesis(w, noise_mode="const")
                
                # Apply post-processing transformations
                if transformation == 'downsample':
                    downsample_size = getattr(config.evaluate, 'downsample_size', 128)
                    original_images = downsample_and_upsample(original_images, downsample_size)
                elif transformation == 'jpeg':
                    jpeg_quality = getattr(config.evaluate, 'jpeg_quality', 55)
                    original_images = apply_jpeg_compression(original_images, jpeg_quality)
            
            # Extract w_partial if using latent-based approach
            w_partials = None
            if latent_indices is not None:
                if w.ndim == 3:
                    w_single = w[:, 0, :]
                else:
                    w_single = w
                w_partials = w_single[:, latent_indices]
        
        # Perform PGD attack
        batch_results = pgd_attack(
            original_images, w_partials, surrogate_decoders, decoder, key_mapper,
            config, device, pixel_indices=pixel_indices, alpha=alpha
        )
        
        # Update metrics
        attack_metrics['initial_match_rate'] += batch_results['initial_match_rate'] * actual_batch_size
        attack_metrics['final_match_rate'] += batch_results['final_match_rate'] * actual_batch_size
        if not attack_metrics['surrogate_classifications']:
            attack_metrics['surrogate_classifications'] = [0] * len(batch_results['surrogate_classifications'])
        for i in range(len(batch_results['surrogate_classifications'])):
            attack_metrics['surrogate_classifications'][i] += batch_results['surrogate_classifications'][i] * actual_batch_size
        attack_metrics['l2_distance'] += batch_results['l2_distance'] * actual_batch_size
        attack_metrics['perceptual_distance'] += batch_results['perceptual_distance'] * actual_batch_size
        attack_metrics['mse_distance_mean'] += batch_results['mse_distance_mean'] * actual_batch_size
        attack_metrics['mse_distance_std'] += batch_results['mse_distance_std'] * actual_batch_size
        attack_metrics['mae_distance_mean'] += batch_results['mae_distance_mean'] * actual_batch_size
        attack_metrics['mae_distance_std'] += batch_results['mae_distance_std'] * actual_batch_size
        attack_metrics['num_samples'] += actual_batch_size
        
        # Store all distances for ASR calculation
        pred_keys_tensor = torch.tensor(batch_results['predicted_keys'], device='cpu')
        true_keys_tensor = torch.tensor(batch_results['true_keys'], device='cpu')
        mse_distances = torch.mean(torch.pow(pred_keys_tensor - true_keys_tensor, 2), dim=1).numpy()
        mae_distances = torch.mean(torch.abs(pred_keys_tensor - true_keys_tensor), dim=1).numpy()
        attack_metrics['all_mse_distances'].extend(mse_distances)
        attack_metrics['all_mae_distances'].extend(mae_distances)
    
    # Normalize metrics
    attack_metrics['initial_match_rate'] /= attack_metrics['num_samples']
    attack_metrics['final_match_rate'] /= attack_metrics['num_samples']
    attack_metrics['surrogate_classifications'] = [s / attack_metrics['num_samples'] for s in attack_metrics['surrogate_classifications']]
    attack_metrics['l2_distance'] /= attack_metrics['num_samples']
    attack_metrics['perceptual_distance'] /= attack_metrics['num_samples']
    attack_metrics['mse_distance_mean'] /= attack_metrics['num_samples']
    attack_metrics['mse_distance_std'] /= attack_metrics['num_samples']
    attack_metrics['mae_distance_mean'] /= attack_metrics['num_samples']
    attack_metrics['mae_distance_std'] /= attack_metrics['num_samples']
    
    # Calculate ASR@95%TPR
    if watermarked_mse_distances is not None and watermarked_mae_distances is not None:
        asr_mse, _ = calculate_asr_at_tpr(watermarked_mse_distances, attack_metrics['all_mse_distances'])
        asr_mae, _ = calculate_asr_at_tpr(watermarked_mae_distances, attack_metrics['all_mae_distances'])
        attack_metrics['asr_95tpr_mse'] = asr_mse
        attack_metrics['asr_95tpr_mae'] = asr_mae
    else:
        attack_metrics['asr_95tpr_mse'] = 0.0
        attack_metrics['asr_95tpr_mae'] = 0.0
    
    return attack_metrics


def main():
    """Main entry point for attack."""
    args = parse_args()
    
    # Setup distributed training first
    local_rank, rank, world_size, device = setup_distributed()
    
    # Load default configuration and update with args
    config = get_default_config()
    # Explicitly specify we're in attack mode and include the new image-pixel and w-partial parameters
    config.update_from_args(args, mode='attack')
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Setup logging - only on rank 0
    if rank == 0:
        setup_logging(config.output_dir, rank)
        
        # Log configuration after it's been updated
        logging.info(f"Configuration:\n{config}")
        
        # Log specific approach
        if config.model.use_image_pixels:
            logging.info(f"Using image-pixel based approach with {config.model.image_pixel_count} pixels and seed {config.model.image_pixel_set_seed}")
        else:
            logging.info(f"Using latent vector (w_partial) based approach")
        
        # Log alpha values to be tested
        alpha_values = config.attack.pgd_alpha
        if not isinstance(alpha_values, list):
            alpha_values = [alpha_values]
        logging.info(f"Testing PGD alpha values: {alpha_values}")
            
        logging.info(f"Distributed setup: local_rank={local_rank}, rank={rank}, world_size={world_size}, device={device}")
    
    try:
        # Run attack
        all_attack_metrics = run_attack(config, local_rank, rank, world_size, device)
        
        if rank == 0:
            logging.info("\n===== ATTACK SUMMARY =====")
            logging.info("Alpha | Case | Match Rate | ASR@95%TPR (MSE) | ASR@95%TPR (MAE) | L2 Dist | LPIPS Dist")
            logging.info("----- | ---- | ---------- | ---------------- | ---------------- | ------- | ----------")
            
            # Print a summary of results for each alpha value
            for alpha, case_metrics in all_attack_metrics.items():
                for case_name, metrics in case_metrics.items():
                    logging.info(f"{alpha:.4f} | {case_name} | {metrics['final_match_rate']:.2f}% | {metrics['asr_95tpr_mse']:.2f}% | {metrics['asr_95tpr_mae']:.2f}% | {metrics['l2_distance']:.4f} | {metrics['perceptual_distance']:.4f}")
            
            logging.info(f"Attack completed. Results saved to {config.output_dir}")
        
    except Exception as e:
        if rank == 0:
            logging.error(f"Error in attack: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main() 