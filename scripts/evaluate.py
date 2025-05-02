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
    parser.add_argument("--checkpoint_path", type=str,
                        help="Path to checkpoint to evaluate")
    parser.add_argument("--img_size", type=int, default=256, help="Image resolution")
    parser.add_argument("--image_pixel_set_seed", type=int, default=42,
                        help="Random seed for selecting pixel indices from the generated image")
    parser.add_argument("--image_pixel_count", type=int, default=32,
                        help="Number of pixels to select from the image (default: 32)")

    # Pretrained model configuration
    parser.add_argument("--pretrained_models", type=str, nargs='+', default=[],
                        help="List of pretrained model names to use for evaluation. Available options: "
                             "ffhq70k-ada, ffhq1k, ffhq30k, ffhq70k-bcr, ffhq70k-noaug. "
                             "If empty, all available models will be used.")
    parser.add_argument("--custom_pretrained_models", type=str, nargs='+', default=[],
                        help="List of custom pretrained model specifications in format: "
                             "name:url:local_path. Example: 'custom1:http://example.com/model.pkl:local.pkl'")

    # Evaluation configuration
    parser.add_argument("--num_samples", type=int, default=1000, 
                        help="Number of samples to evaluate")
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
    
    # Process custom pretrained models
    custom_models = {}
    for model_spec in args.custom_pretrained_models:
        try:
            name, url, local_path = model_spec.split(':')
            custom_models[name] = (url, local_path)
        except ValueError:
            if rank == 0:
                logging.error(f"Invalid custom model specification: {model_spec}. "
                            f"Format should be 'name:url:local_path'")
            continue
    
    # Create output directory and setup logging
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
        setup_logging(config.output_dir, rank)
        
        # Log configuration after it's been updated
        logging.info(f"Configuration:\n{config}")
        logging.info(f"Distributed setup: local_rank={local_rank}, rank={rank}, world_size={world_size}, device={device}")
        
        if args.pretrained_models:
            logging.info(f"Using specified pretrained models: {args.pretrained_models}")
        if custom_models:
            logging.info(f"Using custom pretrained models: {list(custom_models.keys())}")
    
    try:
        # Initialize evaluator
        evaluator = WatermarkEvaluator(
            config=config,
            local_rank=local_rank,
            rank=rank,
            world_size=world_size,
            device=device,
            selected_pretrained_models=args.pretrained_models,
            custom_pretrained_models=custom_models
        )
        
        # Run evaluation
        evaluator.evaluate()
        
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