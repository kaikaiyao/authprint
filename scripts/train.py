#!/usr/bin/env python
"""
Main training script for StyleGAN watermarking.
"""
import argparse
import logging
import sys
import os

# Add the parent directory (project root) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.default_config import Config, get_default_config
from trainers.watermark_trainer import WatermarkTrainer
from utils.distributed import setup_distributed, cleanup_distributed
from utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Watermarking Training Pipeline for StyleGAN2-ADA")
    
    # Model configuration
    parser.add_argument("--stylegan2_url", type=str,
                        default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl",
                        help="URL for the pretrained StyleGAN2 model")
    parser.add_argument("--stylegan2_local_path", type=str,
                        default="ffhq70k-paper256-ada.pkl",
                        help="Local path to store/load the StyleGAN2 model")
    parser.add_argument("--img_size", type=int, default=256, help="Image resolution")
    parser.add_argument("--key_length", type=int, default=4, help="Length of the binary key (output dimension)")
    parser.add_argument("--selected_indices", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31",
                        help="Comma-separated list of indices to select for latent partial")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--total_iterations", type=int, default=100000, help="Total number of training iterations")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda_lpips", type=float, default=1.0, help="Weight for LPIPS loss")
    parser.add_argument("--log_interval", type=int, default=1, help="Interval for logging training progress")
    parser.add_argument("--checkpoint_interval", type=int, default=10000, help="Interval for saving checkpoints")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save logs and checkpoints")
    
    # Other configuration
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    return parser.parse_args()


def main():
    """Main entry point for training."""
    args = parse_args()
    
    # Load default configuration and update with args
    config = get_default_config()
    config.update_from_args(args)
    
    # Setup distributed training
    local_rank, rank, world_size, device = setup_distributed()
    
    # Setup logging
    setup_logging(config.output_dir, rank)
    
    # Log configuration
    if rank == 0:
        logging.info(f"Configuration:\n{config}")
        logging.info(f"Distributed setup: local_rank={local_rank}, rank={rank}, world_size={world_size}, device={device}")
    
    try:
        # Initialize trainer
        trainer = WatermarkTrainer(config, local_rank, rank, world_size, device)
        
        # Run training
        trainer.train()
        
    except Exception as e:
        logging.error(f"Error in training: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main() 