#!/usr/bin/env python
"""
Experiment script to analyze how decoder predictions shift when pixels are manipulated.
"""
import argparse
import logging
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# Add the parent directory (project root) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.default_config import get_default_config
from utils.distributed import setup_distributed, cleanup_distributed
from utils.logging_utils import setup_logging
from models.stylegan2_model import StyleGAN2Model
from models.decoder import DecoderSD_L, DecoderSD_M, DecoderSD_S, StyleGAN2Decoder
from utils.checkpoint import load_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pixel Manipulation Experiment")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, default="stylegan2",
                       choices=["stylegan2"],
                       help="Type of generative model to use")
    
    # StyleGAN2 configuration
    parser.add_argument("--stylegan2_url", type=str,
                        default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl",
                        help="URL for the pretrained StyleGAN2 model")
    parser.add_argument("--stylegan2_local_path", type=str,
                        default="ffhq70k-paper256-ada.pkl",
                        help="Local path to store/load the StyleGAN2 model")
    
    # Decoder configuration
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to decoder checkpoint")
    parser.add_argument("--decoder_size", type=str, default="M",
                        choices=["S", "M", "L"],
                        help="Size of decoder to use (S=Small, M=Medium, L=Large)")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image resolution")
    parser.add_argument("--image_pixel_count", type=int, default=32,
                        help="Number of pixels to predict")
    parser.add_argument("--image_pixel_set_seed", type=int, default=42,
                        help="Random seed for selecting pixel indices")
    
    # Experiment configuration
    parser.add_argument("--num_images", type=int, default=10,
                        help="Number of different images to test")
    parser.add_argument("--num_repeats", type=int, default=5,
                        help="Number of repeats for each pixel count")
    parser.add_argument("--max_pixels", type=int, default=65536,
                        help="Maximum number of pixels to manipulate")
    parser.add_argument("--pixel_steps", type=str, default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536",
                        help="Comma-separated list of number of pixels to manipulate")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="pixel_manipulation_results",
                        help="Directory to save experiment results")
    
    return parser.parse_args()


def compute_prediction_distance(pred1, pred2):
    """
    Compute distance between two decoder predictions.
    You can modify this function to use different distance metrics.
    """
    return torch.nn.functional.mse_loss(pred1, pred2).item()


def manipulate_pixels(image, num_pixels, manipulation_value=-1.0):
    """
    Randomly manipulate a specified number of pixels in the image.
    
    Args:
        image (torch.Tensor): Input image tensor (C x H x W)
        num_pixels (int): Number of pixels to manipulate
        manipulation_value (float): Value to set for manipulated pixels
    
    Returns:
        torch.Tensor: Manipulated image
        list: Indices of manipulated pixels (for logging/visualization)
    """
    manipulated_image = image.clone()
    
    # Get image dimensions
    C, H, W = image.shape
    total_pixels = H * W
    
    # Randomly select pixels to manipulate
    flat_indices = np.random.choice(total_pixels, size=num_pixels, replace=False)
    y_indices = flat_indices // W
    x_indices = flat_indices % W
    
    # Manipulate selected pixels across all channels
    for c in range(C):
        manipulated_image[c, y_indices, x_indices] = manipulation_value
    
    return manipulated_image, list(zip(y_indices, x_indices))


def run_experiment(args, decoder, generator, device):
    """Run the pixel manipulation experiment."""
    # Parse pixel steps
    pixel_steps = [int(x) for x in args.pixel_steps.split(',')]
    
    # Initialize results dictionary
    results = {
        'pixel_counts': pixel_steps,
        'distances': [],
        'std_devs': []
    }
    
    # For each test image
    for img_idx in tqdm(range(args.num_images), desc="Processing images"):
        # Generate a random image
        with torch.no_grad():
            z = torch.randn(1, generator.z_dim, device=device)
            original_image = generator(z)[0]  # Take first image
        
        # Get original decoder prediction
        original_pred = decoder(original_image.unsqueeze(0))
        
        # For each pixel count
        img_distances = []
        for num_pixels in pixel_steps:
            step_distances = []
            
            # Repeat multiple times for each pixel count
            for _ in range(args.num_repeats):
                # Manipulate pixels
                manipulated_image, _ = manipulate_pixels(
                    original_image.clone(),
                    num_pixels
                )
                
                # Get new prediction
                with torch.no_grad():
                    new_pred = decoder(manipulated_image.unsqueeze(0))
                
                # Compute and store distance
                distance = compute_prediction_distance(original_pred, new_pred)
                step_distances.append(distance)
            
            img_distances.append(step_distances)
        
        # Convert to numpy for easier computation
        img_distances = np.array(img_distances)
        
        # Store results for this image
        if len(results['distances']) == 0:
            results['distances'] = img_distances
        else:
            results['distances'] += img_distances
    
    # Average results across all images
    results['distances'] = results['distances'] / args.num_images
    
    # Compute standard deviations
    results['std_devs'] = np.std(results['distances'], axis=1)
    results['distances'] = np.mean(results['distances'], axis=1)
    
    return results


def save_results(results, output_dir):
    """Save experiment results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    np.savez(
        os.path.join(output_dir, 'pixel_manipulation_results.npz'),
        pixel_counts=results['pixel_counts'],
        distances=results['distances'],
        std_devs=results['std_devs']
    )
    
    # Create a simple text summary
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("Pixel Manipulation Experiment Results\n")
        f.write("===================================\n\n")
        f.write("Pixels Manipulated | Mean Distance | Std Dev\n")
        f.write("-----------------------------------------\n")
        for px, dist, std in zip(results['pixel_counts'], 
                               results['distances'],
                               results['std_devs']):
            f.write(f"{px:16d} | {dist:12.6f} | {std:7.6f}\n")


def main():
    """Main entry point for the experiment."""
    args = parse_args()
    
    # Setup distributed training first
    local_rank, rank, world_size, device = setup_distributed()
    
    # Load default configuration and update with args
    config = get_default_config()
    config.update_from_args(args, mode='evaluate')
    
    # Create output directory and setup logging
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
        setup_logging(config.output_dir, rank)
        logging.info(f"Configuration:\n{config}")
    
    try:
        # Initialize models
        generator = StyleGAN2Model(
            url=args.stylegan2_url,
            local_path=args.stylegan2_local_path,
            device=device,
            img_size=args.img_size
        ).to(device)
        
        # Initialize decoder based on model type and size
        decoder_output_dim = args.image_pixel_count  # For direct pixel prediction
        
        if args.model_type == "stylegan2":
            decoder = StyleGAN2Decoder(
                image_size=args.img_size,
                channels=3,
                output_dim=decoder_output_dim
            ).to(device)
            if rank == 0:
                logging.info(f"Initialized StyleGAN2Decoder with output_dim={decoder_output_dim}")
        else:  # stable-diffusion
            decoder_class = {
                "S": DecoderSD_S,
                "M": DecoderSD_M,
                "L": DecoderSD_L
            }[args.decoder_size]
            
            decoder = decoder_class(
                image_size=args.img_size,
                channels=3,
                output_dim=decoder_output_dim
            ).to(device)
            
            if rank == 0:
                logging.info(f"Initialized SD-Decoder-{args.decoder_size} with output_dim={decoder_output_dim}")
        
        decoder.eval()
        
        # Load checkpoint using the utility function
        if rank == 0:
            logging.info(f"Loading checkpoint from {args.checkpoint_path}...")
        
        load_checkpoint(
            checkpoint_path=args.checkpoint_path,
            decoder=decoder,
            device=device
        )
        
        # Run experiment
        results = run_experiment(args, decoder, generator, device)
        
        # Save results
        if rank == 0:
            save_results(results, args.output_dir)
            logging.info(f"Experiment completed. Results saved to {args.output_dir}")
        
    except Exception as e:
        if rank == 0:
            logging.error(f"Error in experiment: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main() 