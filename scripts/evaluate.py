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
    
    # Additional evaluation options for negative samples - enabled by default
    parser.add_argument("--evaluate_neg_samples", action="store_true", default=True,
                        help="Whether to evaluate on negative samples")
    
    # Pre-trained model options - enabled by default
    parser.add_argument("--evaluate_pretrained", action="store_true", default=True,
                        help="Whether to evaluate on additional pretrained models")
    parser.add_argument("--evaluate_ffhq1k", action="store_true", default=True,
                        help="Evaluate on ffhq1k model")
    parser.add_argument("--evaluate_ffhq30k", action="store_true", default=True,
                        help="Evaluate on ffhq30k model")
    parser.add_argument("--evaluate_ffhq70k_bcr", action="store_true", default=True,
                        help="Evaluate on ffhq70k-bcr model")
    parser.add_argument("--evaluate_ffhq70k_noaug", action="store_true", default=True,
                        help="Evaluate on ffhq70k-noaug model")
    
    # Image transformation options - enabled by default
    parser.add_argument("--evaluate_transforms", action="store_true", default=True,
                        help="Whether to evaluate on transformed images")
    parser.add_argument("--evaluate_truncation", action="store_true", default=True,
                        help="Evaluate on truncated images")
    parser.add_argument("--truncation_psi", type=float, default=2.0,
                        help="Truncation psi parameter")
    parser.add_argument("--evaluate_quantization", action="store_true", default=True,
                        help="Evaluate on images from quantized model")
    parser.add_argument("--evaluate_downsample", action="store_true", default=True,
                        help="Evaluate on downsampled and upsampled images")
    parser.add_argument("--downsample_size", type=int, default=128,
                        help="Size to downsample to before upsampling back")
    parser.add_argument("--evaluate_jpeg", action="store_true", default=False,
                        help="Evaluate on JPEG compressed images")
    parser.add_argument("--jpeg_quality", type=int, default=55,
                        help="JPEG compression quality (0-100)")
    
    # Disable flags for convenience
    parser.add_argument("--disable_all_neg_samples", action="store_true", default=False,
                        help="Disable all negative sample evaluations (overrides individual settings)")
    
    # Parse args and handle boolean flags properly
    args = parser.parse_args()
    
    # Handle disable flag
    if args.disable_all_neg_samples:
        args.evaluate_neg_samples = False
    
    return args


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
        
        # Log explicit evaluation options
        logging.info("Evaluation Options:")
        logging.info(f"  Evaluate negative samples: {config.evaluate.evaluate_neg_samples}")
        if config.evaluate.evaluate_neg_samples:
            logging.info(f"  Evaluate pretrained models: {config.evaluate.evaluate_pretrained}")
            if config.evaluate.evaluate_pretrained:
                logging.info(f"    - FFHQ1K: {config.evaluate.evaluate_ffhq1k}")
                logging.info(f"    - FFHQ30K: {config.evaluate.evaluate_ffhq30k}")
                logging.info(f"    - FFHQ70K-BCR: {config.evaluate.evaluate_ffhq70k_bcr}")
                logging.info(f"    - FFHQ70K-NOAUG: {config.evaluate.evaluate_ffhq70k_noaug}")
            
            logging.info(f"  Evaluate transforms: {config.evaluate.evaluate_transforms}")
            if config.evaluate.evaluate_transforms:
                logging.info(f"    - Truncation (psi={config.evaluate.truncation_psi}): {config.evaluate.evaluate_truncation}")
                logging.info(f"    - Quantization: {config.evaluate.evaluate_quantization}")
                logging.info(f"    - Downsample (size={config.evaluate.downsample_size}): {config.evaluate.evaluate_downsample}")
                logging.info(f"    - JPEG (quality={config.evaluate.jpeg_quality}): {config.evaluate.evaluate_jpeg}")
                
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