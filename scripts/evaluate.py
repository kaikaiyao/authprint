#!/usr/bin/env python
"""
Evaluation script for StyleGAN watermarking.
"""
import argparse
import logging
import os
import sys

# Add the parent directory (project root) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.default_config import get_default_config
from evaluators.watermark_evaluator import WatermarkEvaluator
from utils.distributed import setup_distributed, cleanup_distributed
from utils.logging_utils import setup_logging


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
    
    return parser.parse_args()


def main():
    """Main entry point for evaluation."""
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
        
        # Log configuration after it's been updated
        logging.info(f"Configuration:\n{config}")
        logging.info(f"Distributed setup: local_rank={local_rank}, rank={rank}, world_size={world_size}, device={device}")
    
    try:
        # Initialize evaluator
        evaluator = WatermarkEvaluator(config, local_rank, rank, world_size, device)
        
        # Run evaluation
        metrics = evaluator.evaluate(config.evaluate.evaluation_mode)
        
        if rank == 0:
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