#!/usr/bin/env python
"""
Attack script for StyleGAN watermarking.
This implements a forging attack that aims to modify original non-watermarked images
so that the real decoder extracts the correct watermark keys from them.
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

# Add the parent directory (project root) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.default_config import Config, get_default_config
from models.decoder import Decoder
from models.key_mapper import KeyMapper
from utils.checkpoint import load_checkpoint
from utils.logging_utils import setup_logging
from models.model_utils import load_stylegan2_model, clone_model
from utils.distributed import setup_distributed, cleanup_distributed


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
    parser.add_argument("--selected_indices", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31",
                        help="Comma-separated list of indices to select for latent partial")
    parser.add_argument("--key_mapper_seed", type=int, default=2025, 
                        help="Specific random seed for KeyMapper initialization for reproducibility")
    
    # Attack configuration
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for attack")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to attack")
    
    # PGD attack parameters
    parser.add_argument("--pgd_alpha", type=float, default=0.01, help="PGD step size")
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
            iterator = tqdm(range(num_batches)) if rank == 0 else range(num_batches)
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
            if rank == 0 and (epoch + 1) % config.attack.log_interval == 0:
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


def pgd_attack(images, w_partials, surrogate_decoders, real_decoder, key_mapper, config, device):
    """
    Perform PGD attack on the images using surrogate decoders to guide the attack.
    This is a FORGING attack that aims to make the real decoder extract the correct watermark keys
    from the modified original images.
    
    Args:
        images: Original images to attack
        w_partials: Partial latent vectors corresponding to the images
        surrogate_decoders: List of surrogate decoder models
        real_decoder: The real decoder model
        key_mapper: The key mapper model
        config: Configuration object
        device: Device to run the attack on
        
    Returns:
        dict: Attack results including statistics and attacked images
    """
    # Create a copy of the images for attack (detach to avoid gradient tracking from generation)
    attacked_images = images.clone().detach().requires_grad_(True)
    
    # Get true keys from key mapper
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
    optimizer = optim.Adam([attacked_images], lr=config.attack.pgd_alpha)
    
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
        final_bin_pred = (torch.sigmoid(final_pred) > 0.5).float()
        final_match = (final_bin_pred == true_keys).all(dim=1).float().mean().item() * 100.0
        
        # Binary classification by surrogate decoders
        surrogate_classifications = []
        for surrogate in surrogate_decoders:
            output = surrogate(attacked_images)
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
        'predicted_keys': final_bin_pred.cpu().numpy()
    }


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
        dict: Attack results
    """
    # Add debug logging at start of attack
    print(f"[DEBUG] Starting attack on rank {rank} with num_samples = {config.attack.num_samples}")
    
    # Parse latent indices for partial vector
    if isinstance(config.model.selected_indices, str):
        latent_indices = [int(idx) for idx in config.model.selected_indices.split(',')]
    else:
        latent_indices = config.model.selected_indices
    
    # Load the original StyleGAN2 model
    if rank == 0:
        logging.info("Loading StyleGAN2 model...")
    gan_model = load_stylegan2_model(
        config.model.stylegan2_url,
        config.model.stylegan2_local_path,
        device
    )
    gan_model.eval()
    
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
    key_mapper = KeyMapper(
        input_dim=len(latent_indices),
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
    
    # Initialize attack metrics
    attack_metrics = {
        'initial_match_rate': 0.0,
        'final_match_rate': 0.0,
        'surrogate_classifications': [],
        'l2_distance': 0.0,
        'perceptual_distance': 0.0,
        'num_samples': 0
    }
    
    # Batch attack
    batch_size = config.attack.batch_size
    num_batches = (config.attack.num_samples + batch_size - 1) // batch_size
    
    # Step 1: Train surrogate decoders
    surrogate_decoders = train_surrogate_decoders(
        gan_model, watermarked_model, config, local_rank, rank, world_size, device
    )
    
    # Only use tqdm progress bar on rank 0
    batch_iterator = tqdm(range(num_batches), desc="Attacking batches") if rank == 0 else range(num_batches)
    
    # Prepare to save visualizations (only on rank 0)
    if rank == 0:
        vis_dir = os.path.join(config.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        vis_indices = np.random.choice(
            config.attack.num_samples, 
            min(config.attack.visualization_samples, config.attack.num_samples), 
            replace=False
        )
        vis_counter = 0
        vis_images = []
    
    # Step 2: Perform PGD attack on batches
    for batch_idx in batch_iterator:
        actual_batch_size = min(batch_size, config.attack.num_samples - batch_idx * batch_size)
        if actual_batch_size <= 0:
            break
        
        # Generate random latent vectors
        z = torch.randn(actual_batch_size, gan_model.z_dim, device=device)
        
        # Generate original images
        with torch.no_grad():
            w = gan_model.mapping(z, None)
            original_images = gan_model.synthesis(w, noise_mode='const')
            
            # If w is replicated for each synthesis layer, take the first as representative
            if w.ndim == 3:
                w_single = w[:, 0, :]
            else:
                w_single = w
            
            # Extract partial w vector using indices
            w_partial = w_single[:, latent_indices]
        
        # Perform PGD attack
        batch_results = pgd_attack(
            original_images, w_partial, surrogate_decoders, decoder, key_mapper, config, device
        )
        
        # Update metrics (needs to be aggregated across ranks later)
        attack_metrics['initial_match_rate'] += batch_results['initial_match_rate'] * actual_batch_size
        attack_metrics['final_match_rate'] += batch_results['final_match_rate'] * actual_batch_size
        if not attack_metrics['surrogate_classifications']:
            attack_metrics['surrogate_classifications'] = [0] * len(batch_results['surrogate_classifications'])
        for i in range(len(batch_results['surrogate_classifications'])):
            attack_metrics['surrogate_classifications'][i] += batch_results['surrogate_classifications'][i] * actual_batch_size
        attack_metrics['l2_distance'] += batch_results['l2_distance'] * actual_batch_size
        attack_metrics['perceptual_distance'] += batch_results['perceptual_distance'] * actual_batch_size
        attack_metrics['num_samples'] += actual_batch_size
        
        # Save visualizations (only on rank 0)
        if rank == 0:
            global_indices = np.arange(batch_idx * batch_size, (batch_idx * batch_size) + actual_batch_size)
            for i, global_idx in enumerate(global_indices):
                if global_idx in vis_indices:
                    # Save original and attacked image pair
                    orig_img = (batch_results['original_images'][i].permute(1, 2, 0).cpu().numpy() + 1) / 2
                    orig_img = np.clip(orig_img, 0, 1)
                    
                    attacked_img = (batch_results['attacked_images'][i].permute(1, 2, 0).cpu().numpy() + 1) / 2
                    attacked_img = np.clip(attacked_img, 0, 1)
                    
                    # Create a comparison figure
                    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                    axs[0].imshow(orig_img)
                    axs[0].set_title("Original Image")
                    axs[0].axis('off')
                    
                    axs[1].imshow(attacked_img)
                    axs[1].set_title("Attacked Image")
                    axs[1].axis('off')
                    
                    # Add key information
                    true_key = batch_results['true_keys'][i]
                    pred_key = batch_results['predicted_keys'][i]
                    key_match = true_key == pred_key
                    
                    plt.suptitle(f"Image {global_idx}\nTrue Key: {true_key}\nPredicted Key: {pred_key}\nForging Success: {key_match}")
                    plt.tight_layout()
                    
                    # Save figure
                    plt.savefig(os.path.join(vis_dir, f"attack_vis_{global_idx}.png"))
                    plt.close(fig)
                    
                    vis_counter += 1
    
    # Aggregate metrics across ranks in distributed setting
    if world_size > 1:
        for key in ['initial_match_rate', 'final_match_rate', 'l2_distance', 'perceptual_distance', 'num_samples']:
            tensor = torch.tensor([attack_metrics[key]], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            attack_metrics[key] = tensor.item()
        
        for i in range(len(attack_metrics['surrogate_classifications'])):
            tensor = torch.tensor([attack_metrics['surrogate_classifications'][i]], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            attack_metrics['surrogate_classifications'][i] = tensor.item()
    
    # Normalize metrics
    attack_metrics['initial_match_rate'] /= attack_metrics['num_samples']
    attack_metrics['final_match_rate'] /= attack_metrics['num_samples']
    attack_metrics['surrogate_classifications'] = [s / attack_metrics['num_samples'] for s in attack_metrics['surrogate_classifications']]
    attack_metrics['l2_distance'] /= attack_metrics['num_samples']
    attack_metrics['perceptual_distance'] /= attack_metrics['num_samples']
    
    # Print and log metrics (only on rank 0)
    if rank == 0:
        logging.info("\n===== Attack Results =====")
        logging.info(f"Number of samples: {attack_metrics['num_samples']}")
        logging.info(f"Initial match rate: {attack_metrics['initial_match_rate']:.2f}%")
        logging.info(f"Final match rate after attack: {attack_metrics['final_match_rate']:.2f}%")
        logging.info(f"Attack success rate: {attack_metrics['final_match_rate']:.2f}%")
        
        for i, rate in enumerate(attack_metrics['surrogate_classifications']):
            logging.info(f"Surrogate decoder {i+1} classification rate: {rate:.2f}%")
        
        logging.info(f"L2 distance between original and attacked images: {attack_metrics['l2_distance']:.4f}")
        logging.info(f"LPIPS perceptual distance: {attack_metrics['perceptual_distance']:.4f}")
        
        # Save metrics to file
        metrics_path = os.path.join(config.output_dir, "attack_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write("===== Attack Results =====\n")
            f.write(f"Number of samples: {attack_metrics['num_samples']}\n")
            f.write(f"Initial match rate: {attack_metrics['initial_match_rate']:.2f}%\n")
            f.write(f"Final match rate after attack: {attack_metrics['final_match_rate']:.2f}%\n")
            f.write(f"Attack success rate: {attack_metrics['final_match_rate']:.2f}%\n")
            
            for i, rate in enumerate(attack_metrics['surrogate_classifications']):
                f.write(f"Surrogate decoder {i+1} classification rate: {rate:.2f}%\n")
            
            f.write(f"L2 distance between original and attacked images: {attack_metrics['l2_distance']:.4f}\n")
            f.write(f"LPIPS perceptual distance: {attack_metrics['perceptual_distance']:.4f}\n")
    
    return attack_metrics


def main():
    """Main entry point for attack."""
    args = parse_args()
    
    # Setup distributed training first
    local_rank, rank, world_size, device = setup_distributed()
    
    # Load default configuration and update with args
    config = get_default_config()
    print(f"[DEBUG] Before update - num_samples = {config.attack.num_samples}")
    config.update_from_args(args, mode='attack')  # Explicitly specify we're in attack mode
    print(f"[DEBUG] After update - num_samples = {config.attack.num_samples}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Setup logging - only on rank 0
    if rank == 0:
        setup_logging(config.output_dir, rank)
        
        # Log configuration after it's been updated
        logging.info(f"Configuration:\n{config}")
        logging.info(f"Distributed setup: local_rank={local_rank}, rank={rank}, world_size={world_size}, device={device}")
    
    try:
        # Run attack
        attack_metrics = run_attack(config, local_rank, rank, world_size, device)
        
        if rank == 0:
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