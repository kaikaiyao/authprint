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
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint to resume training from")
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
    parser.add_argument("--key_mapper_use_sine", action="store_true", default=False,
                        help="Use sine-based mapping in the KeyMapper (more sensitive to input changes)")
    parser.add_argument("--key_mapper_sensitivity", type=float, default=20.0,
                        help="Sensitivity parameter for sine-based mapping (higher values: more sensitive to changes)")
    parser.add_argument("--freeze_watermarked_model", action="store_true", default=False,
                        help="Freeze the watermarked model parameters, only train the decoder")
    parser.add_argument("--direct_feature_decoder", action="store_true", default=False,
                        help="When true and using image pixels with a frozen watermarked model, train decoder directly on pixel features instead of full images")
    
    # New: Mutual information estimation parameters
    parser.add_argument("--estimate_mutual_info", action="store_true", default=False,
                        help="Estimate mutual information between selected pixels and full images at the start of training")
    parser.add_argument("--mi_n_samples", type=int, default=1000,
                        help="Number of samples to use for mutual information estimation")
    parser.add_argument("--mi_k_neighbors", type=int, default=3,
                        help="Number of nearest neighbors for k-NN entropy estimation")
    
    # Enhanced FeatureDecoder configuration
    parser.add_argument("--decoder_hidden_dims", type=str, default="1024,2048,1024,512,256",
                        help="Comma-separated list of hidden layer dimensions for FeatureDecoder")
    parser.add_argument("--decoder_activation", type=str, default="gelu",
                        choices=["leaky_relu", "relu", "gelu", "swish", "mish"],
                        help="Activation function for FeatureDecoder")
    parser.add_argument("--decoder_dropout_rate", type=float, default=0.3,
                        help="Dropout rate for FeatureDecoder")
    parser.add_argument("--decoder_num_residual_blocks", type=int, default=3,
                        help="Number of residual blocks in FeatureDecoder")
    parser.add_argument("--decoder_no_spectral_norm", action="store_true", default=False,
                        help="Disable spectral normalization in FeatureDecoder")
    parser.add_argument("--decoder_no_layer_norm", action="store_true", default=False,
                        help="Disable layer normalization in FeatureDecoder")
    parser.add_argument("--decoder_no_attention", action="store_true", default=False,
                        help="Disable self-attention mechanism in FeatureDecoder")
    
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
    
    # ZCA whitening configuration
    parser.add_argument("--use_zca_whitening", action="store_true", default=False,
                        help="Apply ZCA whitening to decoder input images")
    parser.add_argument("--zca_eps", type=float, default=1e-5,
                        help="Epsilon for numerical stability in ZCA whitening")
    parser.add_argument("--zca_batch_size", type=int, default=1000,
                        help="Batch size for computing ZCA statistics")
    
    return parser.parse_args()


def main():
    """Main entry point for training."""
    args = parse_args()
    
    # Setup distributed training first
    local_rank, rank, world_size, device = setup_distributed()
    
    # Load default configuration and update with args
    config = get_default_config()
    config.update_from_args(args, mode='train')
    
    # Setup logging
    setup_logging(config.output_dir, rank)
    
    # Log configuration
    if rank == 0:
        logging.info(f"Configuration:\n{config}")
        logging.info(f"Distributed setup: local_rank={local_rank}, rank={rank}, world_size={world_size}, device={device}")
    
    try:
        # Initialize trainer
        trainer = WatermarkTrainer(config, local_rank, rank, world_size, device)
        
        # Load checkpoint if specified
        if config.checkpoint_path:
            if rank == 0:
                logging.info(f"Resuming training from checkpoint: {config.checkpoint_path}")
            trainer.load_checkpoint(config.checkpoint_path)
        
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