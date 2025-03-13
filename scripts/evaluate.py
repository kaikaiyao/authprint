#!/usr/bin/env python
"""
Evaluation script for StyleGAN watermarking.
"""
import argparse
import logging
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from PIL import Image

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
    parser = argparse.ArgumentParser(description="Evaluation for StyleGAN2 Watermarking")
    
    # Model configuration
    parser.add_argument("--stylegan2_url", type=str,
                        default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl",
                        help="URL for the pretrained StyleGAN2 model")
    parser.add_argument("--stylegan2_local_path", type=str,
                        default="ffhq70k-paper256-ada.pkl",
                        help="Local path to store/load the StyleGAN2 model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to checkpoint to evaluate")
    parser.add_argument("--img_size", type=int, default=256, help="Image resolution")
    parser.add_argument("--key_length", type=int, default=4, help="Length of the binary key (output dimension)")
    parser.add_argument("--selected_indices", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31",
                        help="Comma-separated list of indices to select for latent partial")
    parser.add_argument("--key_mapper_seed", type=int, default=2025, 
                        help="Specific random seed for KeyMapper initialization for reproducibility")

    # Evaluation configuration
    parser.add_argument("--evaluation_mode", type=str, choices=['batch', 'visual', 'both'], default='both',
                        help="Evaluation mode: batch (large-scale metrics), visual (sample visualization), or both")
    parser.add_argument("--num_samples", type=int, default=1000, 
                        help="Number of samples to evaluate in batch mode")
    parser.add_argument("--num_vis_samples", type=int, default=10,
                        help="Number of samples to visualize in visual mode")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for evaluation")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                        help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on")
    
    return parser.parse_args()


def save_image_grid(images, filename, nrow=None, scale=True):
    """
    Save batch of images as a grid in a single file.
    
    Args:
        images (torch.Tensor): Batch of images [B, C, H, W]
        filename (str): Output filename
        nrow (int): Number of images per row
        scale (bool): Whether to scale pixel values to [0, 255]
    """
    if nrow is None:
        nrow = int(np.sqrt(images.shape[0]))
    
    # Convert to numpy and transpose from [B, C, H, W] to [B, H, W, C]
    images_np = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    # Scale to [0, 255] if needed
    if scale:
        images_np = ((images_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    # Create grid
    h, w = images_np.shape[1:3]
    grid_h = images_np.shape[0] // nrow if images_np.shape[0] % nrow == 0 else images_np.shape[0] // nrow + 1
    grid_w = nrow
    
    grid = np.zeros((grid_h * h, grid_w * w, 3), dtype=np.uint8)
    
    for idx, img in enumerate(images_np):
        i = idx // nrow
        j = idx % nrow
        grid[i * h:(i + 1) * h, j * w:(j + 1) * w] = img
    
    Image.fromarray(grid).save(filename)


def save_visualization(orig_images, watermarked_images, diff_images, output_dir, prefix="sample"):
    """
    Save original, watermarked, and difference images.
    
    Args:
        orig_images (torch.Tensor): Original images
        watermarked_images (torch.Tensor): Watermarked images
        diff_images (torch.Tensor): Difference images
        output_dir (str): Output directory
        prefix (str): Prefix for filenames
    """
    num_samples = orig_images.shape[0]
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save individual images
    for i in range(num_samples):
        # Save original image
        orig_path = os.path.join(vis_dir, f"{prefix}_{i}_original.png")
        orig_img = ((orig_images[i].detach().cpu() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        orig_img = orig_img.permute(1, 2, 0).numpy()
        Image.fromarray(orig_img).save(orig_path)
        
        # Save watermarked image
        watermarked_path = os.path.join(vis_dir, f"{prefix}_{i}_watermarked.png")
        watermarked_img = ((watermarked_images[i].detach().cpu() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        watermarked_img = watermarked_img.permute(1, 2, 0).numpy()
        Image.fromarray(watermarked_img).save(watermarked_path)
        
        # Save difference image with proper scaling for visibility
        diff_path = os.path.join(vis_dir, f"{prefix}_{i}_difference.png")
        # Scale difference for better visibility (amplify the subtle differences)
        diff_img = diff_images[i].detach().cpu()
        # Normalize and scale to full range for better visualization
        diff_min = diff_img.min()
        diff_max = diff_img.max()
        if diff_max > diff_min:  # Avoid division by zero
            diff_img = (diff_img - diff_min) / (diff_max - diff_min)
        diff_img = (diff_img * 255).clamp(0, 255).to(torch.uint8)
        diff_img = diff_img.permute(1, 2, 0).numpy()
        Image.fromarray(diff_img).save(diff_path)
    
    # Save grid images
    save_image_grid(orig_images, os.path.join(vis_dir, f"{prefix}_all_original.png"))
    save_image_grid(watermarked_images, os.path.join(vis_dir, f"{prefix}_all_watermarked.png"))
    
    # For difference grid, rescale each difference image individually for better visibility
    norm_diff_images = []
    for i in range(diff_images.shape[0]):
        diff_img = diff_images[i]
        diff_min = diff_img.min()
        diff_max = diff_img.max()
        if diff_max > diff_min:  # Avoid division by zero
            diff_img = (diff_img - diff_min) / (diff_max - diff_min)
        norm_diff_images.append(diff_img)
    norm_diff_images = torch.stack(norm_diff_images)
    save_image_grid(norm_diff_images, os.path.join(vis_dir, f"{prefix}_all_differences.png"), scale=False)


def evaluate_watermark(config, local_rank, rank, world_size, device, evaluation_mode='both'):
    """
    Evaluate the watermarking model.
    
    Args:
        config (Config): Configuration object.
        local_rank (int): Local process rank.
        rank (int): Global process rank.
        world_size (int): Total number of processes.
        device (torch.device): Device to run evaluation on.
        evaluation_mode (str): Evaluation mode 'batch', 'visual', or 'both'.
        
    Returns:
        dict: Evaluation metrics.
    """
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
    latent_dim = gan_model.z_dim
    
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
    
    # Load checkpoint (excluding key mapper)
    if rank == 0:
        logging.info(f"Loading checkpoint from {config.checkpoint_path}...")
    load_checkpoint(
        checkpoint_path=config.checkpoint_path,
        watermarked_model=watermarked_model,
        decoder=decoder,
        key_mapper=None,  # Set to None to skip loading key mapper state
        device=device
    )
    
    # Initialize metrics
    metrics = {
        'watermarked_match_rate': 0.0,
        'original_match_rate': 0.0,
        'watermarked_lpips_loss_avg': 0.0,
        'watermarked_lpips_loss_std': 0.0,
        'num_samples_processed': 0
    }
    
    # Initialize LPIPS loss
    lpips_loss_fn = torch.nn.functional.mse_loss  # Use MSE as a simpler alternative to LPIPS
    all_lpips_losses = []
    
    # Batch evaluation
    if evaluation_mode in ['batch', 'both']:
        if rank == 0:
            logging.info(f"Running batch evaluation with {config.num_samples} samples...")
        
        num_batches = (config.num_samples + config.batch_size - 1) // config.batch_size
        watermarked_correct = 0
        original_correct = 0
        total_samples = 0
        
        # Only use tqdm progress bar on rank 0
        batch_iterator = tqdm(range(num_batches), desc="Evaluating batches") if rank == 0 else range(num_batches)
        
        for _ in batch_iterator:
            current_batch_size = min(config.batch_size, config.num_samples - total_samples)
            if current_batch_size <= 0:
                break
                
            with torch.no_grad():
                # Sample latent vectors
                z = torch.randn(current_batch_size, latent_dim, device=device)
                
                # Generate original images
                w_orig = gan_model.mapping(z, None)
                x_orig = gan_model.synthesis(w_orig, noise_mode="const")
                
                # Generate watermarked images
                w_water = watermarked_model.mapping(z, None)
                x_water = watermarked_model.synthesis(w_water, noise_mode="const")
                
                # Extract latent partial and compute true key
                if w_orig.ndim == 3:
                    w_orig_single = w_orig[:, 0, :]
                    w_water_single = w_water[:, 0, :]
                else:
                    w_orig_single = w_orig
                    w_water_single = w_water
                
                w_partial = w_water_single[:, latent_indices]
                true_key = key_mapper(w_partial)
                
                # Compute key from watermarked image
                pred_key_water_logits = decoder(x_water)
                pred_key_water = (torch.sigmoid(pred_key_water_logits) > 0.5).float()
                
                # Compute key from original image
                pred_key_orig_logits = decoder(x_orig)
                pred_key_orig = (torch.sigmoid(pred_key_orig_logits) > 0.5).float()
                
                # Calculate match rates (all bits must match)
                water_matches = (pred_key_water == true_key).all(dim=1).sum().item()
                orig_matches = (pred_key_orig == true_key).all(dim=1).sum().item()
                
                watermarked_correct += water_matches
                original_correct += orig_matches
                total_samples += current_batch_size
                
                # Calculate LPIPS (using MSE as proxy)
                lpips_losses = lpips_loss_fn(x_water, x_orig, reduction='none').mean(dim=[1, 2, 3])
                all_lpips_losses.extend(lpips_losses.cpu().numpy())
        
        # Calculate final metrics
        watermarked_match_rate = (watermarked_correct / total_samples) * 100
        original_match_rate = (original_correct / total_samples) * 100
        lpips_loss_avg = np.mean(all_lpips_losses)
        lpips_loss_std = np.std(all_lpips_losses)
        
        metrics['watermarked_match_rate'] = watermarked_match_rate
        metrics['original_match_rate'] = original_match_rate
        metrics['watermarked_lpips_loss_avg'] = lpips_loss_avg
        metrics['watermarked_lpips_loss_std'] = lpips_loss_std
        metrics['num_samples_processed'] = total_samples
        
        if rank == 0:
            logging.info(f"Watermarked match rate: {watermarked_match_rate:.2f}%")
            logging.info(f"Original match rate: {original_match_rate:.2f}%")
            logging.info(f"LPIPS loss avg: {lpips_loss_avg:.6f}, std: {lpips_loss_std:.6f}")
    
    # Visual evaluation - only perform on rank 0
    if evaluation_mode in ['visual', 'both'] and rank == 0:
        logging.info(f"Running visual evaluation with {config.num_vis_samples} samples...")
        
        with torch.no_grad():
            # Sample latent vectors
            z_vis = torch.randn(config.num_vis_samples, latent_dim, device=device)
            
            # Generate original images
            w_orig_vis = gan_model.mapping(z_vis, None)
            x_orig_vis = gan_model.synthesis(w_orig_vis, noise_mode="const")
            
            # Generate watermarked images
            w_water_vis = watermarked_model.mapping(z_vis, None)
            x_water_vis = watermarked_model.synthesis(w_water_vis, noise_mode="const")
            
            # Compute difference
            diff_vis = x_water_vis - x_orig_vis
            
            # Save visualizations
            save_visualization(x_orig_vis, x_water_vis, diff_vis, config.output_dir)
            
            # Extract latent partial and compute true key for each sample
            if w_water_vis.ndim == 3:
                w_water_single_vis = w_water_vis[:, 0, :]
            else:
                w_water_single_vis = w_water_vis
            
            w_partial_vis = w_water_single_vis[:, latent_indices]
            true_keys_vis = key_mapper(w_partial_vis)
            
            # Predict keys for both original and watermarked images
            pred_keys_water_vis = torch.sigmoid(decoder(x_water_vis)) > 0.5
            pred_keys_orig_vis = torch.sigmoid(decoder(x_orig_vis)) > 0.5
            
            # Save key comparison to log
            for i in range(config.num_vis_samples):
                logging.info(f"Sample {i}:")
                logging.info(f"  True key: {true_keys_vis[i].cpu().numpy().astype(int)}")
                logging.info(f"  Watermarked pred: {pred_keys_water_vis[i].cpu().numpy().astype(int)}")
                logging.info(f"  Original pred: {pred_keys_orig_vis[i].cpu().numpy().astype(int)}")
                logging.info(f"  Watermarked match: {(pred_keys_water_vis[i] == true_keys_vis[i]).all().item()}")
                logging.info(f"  Original match: {(pred_keys_orig_vis[i] == true_keys_vis[i]).all().item()}")
    
    return metrics


def main():
    """Main entry point for evaluation."""
    args = parse_args()
    
    # Load default configuration and update with args
    config = get_default_config()
    config.update_from_args(args)
    config.checkpoint_path = args.checkpoint_path
    config.num_samples = args.num_samples
    config.num_vis_samples = args.num_vis_samples
    config.evaluation_mode = args.evaluation_mode
    
    # Setup distributed training
    local_rank, rank, world_size, device = setup_distributed()
    
    # Setup logging - only on rank 0
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
        setup_logging(config.output_dir, rank)
        
        # Log configuration
        logging.info(f"Configuration:\n{config}")
        logging.info(f"Distributed setup: local_rank={local_rank}, rank={rank}, world_size={world_size}, device={device}")
    
    try:
        # Run evaluation
        metrics = evaluate_watermark(config, local_rank, rank, world_size, device, config.evaluation_mode)
        
        # Save metrics - only on rank 0
        if rank == 0:
            metrics_path = os.path.join(config.output_dir, "evaluation_metrics.txt")
            with open(metrics_path, 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
            
            logging.info(f"Evaluation completed. Results saved to {config.output_dir}")
        
    except Exception as e:
        if rank == 0:
            logging.error(f"Error in evaluation: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main() 