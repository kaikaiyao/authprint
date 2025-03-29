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
    parser.add_argument("--key_mapper_use_sine", action="store_true", default=False,
                        help="Use sine-based mapping in the KeyMapper (more sensitive to input changes)")
    parser.add_argument("--key_mapper_sensitivity", type=float, default=20.0,
                        help="Sensitivity parameter for sine-based mapping (higher values: more sensitive to changes)")
    parser.add_argument("--direct_feature_decoder", action="store_true", default=False,
                        help="Whether the decoder was trained directly on pixel features instead of full images")
    
    # ZCA whitening configuration
    parser.add_argument("--use_zca_whitening", action="store_true", default=False,
                        help="Apply ZCA whitening to decoder input images")
    parser.add_argument("--zca_eps", type=float, default=1e-5,
                        help="Epsilon for numerical stability in ZCA whitening")
    parser.add_argument("--zca_batch_size", type=int, default=1000,
                        help="Batch size for computing ZCA statistics")
    parser.add_argument("--evaluate_zca_whitening", action="store_true", default=True,
                        help="Evaluate on ZCA whitened images")
    parser.add_argument("--evaluate_zca_whitening_watermarked", action="store_true", default=True,
                        help="Evaluate on ZCA whitened images from watermarked model")

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
    
    # Visualization configuration
    parser.add_argument("--visualization_seed", type=int, default=42,
                        help="Random seed for generating visualization samples (ensures consistency across models)")
    parser.add_argument("--verbose_visualization", action="store_true", default=False,
                        help="Enable verbose logging for individual sample visualization details")
    
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
    
    # Truncation options
    parser.add_argument("--evaluate_truncation", action="store_true", default=True,
                        help="Evaluate on truncated images (original model)")
    parser.add_argument("--evaluate_truncation_watermarked", action="store_true", default=True,
                        help="Evaluate on truncated images (watermarked model)")
    parser.add_argument("--truncation_psi", type=float, default=2.0,
                        help="Truncation psi parameter")
    
    # Int8 quantization options
    parser.add_argument("--evaluate_quantization", action="store_true", default=True,
                        help="Evaluate on images from int8 quantized model (original model)")
    parser.add_argument("--evaluate_quantization_watermarked", action="store_true", default=True,
                        help="Evaluate on images from int8 quantized model (watermarked model)")
    
    # Int4 quantization options
    parser.add_argument("--evaluate_quantization_int4", action="store_true", default=True,
                        help="Evaluate on images from int4 quantized model (original model)")
    parser.add_argument("--evaluate_quantization_int4_watermarked", action="store_true", default=True,
                        help="Evaluate on images from int4 quantized model (watermarked model)")
    
    # Int2 quantization options
    parser.add_argument("--evaluate_quantization_int2", action="store_true", default=True,
                        help="Evaluate on images from int2 quantized model (original model)")
    parser.add_argument("--evaluate_quantization_int2_watermarked", action="store_true", default=True,
                        help="Evaluate on images from int2 quantized model (watermarked model)")
    
    # Downsampling options
    parser.add_argument("--evaluate_downsample", action="store_true", default=True,
                        help="Evaluate on downsampled and upsampled images (original model)")
    parser.add_argument("--evaluate_downsample_watermarked", action="store_true", default=True,
                        help="Evaluate on downsampled and upsampled images (watermarked model)")
    parser.add_argument("--downsample_size", type=int, default=128,
                        help="Size to downsample to before upsampling back")
    
    # JPEG compression options
    parser.add_argument("--evaluate_jpeg", action="store_true", default=False,
                        help="Evaluate on JPEG compressed images (original model)")
    parser.add_argument("--evaluate_jpeg_watermarked", action="store_true", default=False,
                        help="Evaluate on JPEG compressed images (watermarked model)")
    parser.add_argument("--jpeg_quality", type=int, default=55,
                        help="JPEG compression quality (0-100)")
    
    # Disable flags for convenience
    parser.add_argument("--disable_all_neg_samples", action="store_true", default=False,
                        help="Disable all negative sample evaluations (overrides individual settings)")
    
    # Multi-decoder mode options
    parser.add_argument("--enable_multi_decoder", action="store_true", default=False,
                        help="Enable multi-decoder mode for evaluation")
    parser.add_argument("--multi_decoder_checkpoints", type=str, default=None,
                        help="Comma-separated list of checkpoint paths for multi-decoder mode")
    parser.add_argument("--multi_decoder_key_lengths", type=str, default=None,
                        help="Comma-separated list of key lengths for each decoder")
    parser.add_argument("--multi_decoder_key_mapper_seeds", type=str, default=None,
                        help="Comma-separated list of key mapper seeds for each decoder")
    parser.add_argument("--multi_decoder_pixel_counts", type=str, default=None,
                        help="Comma-separated list of pixel counts for each decoder")
    parser.add_argument("--multi_decoder_pixel_seeds", type=str, default=None,
                        help="Comma-separated list of pixel seeds for each decoder")
    
    # Parse args and handle boolean flags properly
    args = parser.parse_args()
    
    # Handle disable flag
    if args.disable_all_neg_samples:
        args.evaluate_neg_samples = False
    
    # Handle multi-decoder mode arguments
    if args.enable_multi_decoder:
        if not args.multi_decoder_checkpoints:
            raise ValueError("Multi-decoder mode requires --multi_decoder_checkpoints")
        args.multi_decoder_checkpoints = args.multi_decoder_checkpoints.split(',')
        
        if args.multi_decoder_key_lengths:
            args.multi_decoder_key_lengths = [int(x) for x in args.multi_decoder_key_lengths.split(',')]
        if args.multi_decoder_key_mapper_seeds:
            args.multi_decoder_key_mapper_seeds = [int(x) for x in args.multi_decoder_key_mapper_seeds.split(',')]
        if args.multi_decoder_pixel_counts:
            args.multi_decoder_pixel_counts = [int(x) for x in args.multi_decoder_pixel_counts.split(',')]
        if args.multi_decoder_pixel_seeds:
            args.multi_decoder_pixel_seeds = [int(x) for x in args.multi_decoder_pixel_seeds.split(',')]
    
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
                if config.evaluate.evaluate_truncation:
                    logging.info(f"      - Original model: {config.evaluate.evaluate_truncation}")
                    logging.info(f"      - Watermarked model: {config.evaluate.evaluate_truncation_watermarked}")
                
                logging.info(f"    - Int8 Quantization: {config.evaluate.evaluate_quantization}")
                if config.evaluate.evaluate_quantization:
                    logging.info(f"      - Original model: {config.evaluate.evaluate_quantization}")
                    logging.info(f"      - Watermarked model: {config.evaluate.evaluate_quantization_watermarked}")
                
                logging.info(f"    - Int4 Quantization: {config.evaluate.evaluate_quantization_int4}")
                if config.evaluate.evaluate_quantization_int4:
                    logging.info(f"      - Original model: {config.evaluate.evaluate_quantization_int4}")
                    logging.info(f"      - Watermarked model: {config.evaluate.evaluate_quantization_int4_watermarked}")
                
                logging.info(f"    - Int2 Quantization: {config.evaluate.evaluate_quantization_int2}")
                if config.evaluate.evaluate_quantization_int2:
                    logging.info(f"      - Original model: {config.evaluate.evaluate_quantization_int2}")
                    logging.info(f"      - Watermarked model: {config.evaluate.evaluate_quantization_int2_watermarked}")
                
                logging.info(f"    - Downsample (size={config.evaluate.downsample_size}): {config.evaluate.evaluate_downsample}")
                if config.evaluate.evaluate_downsample:
                    logging.info(f"      - Original model: {config.evaluate.evaluate_downsample}")
                    logging.info(f"      - Watermarked model: {config.evaluate.evaluate_downsample_watermarked}")
                
                logging.info(f"    - JPEG (quality={config.evaluate.jpeg_quality}): {config.evaluate.evaluate_jpeg}")
                if config.evaluate.evaluate_jpeg:
                    logging.info(f"      - Original model: {config.evaluate.evaluate_jpeg}")
                    logging.info(f"      - Watermarked model: {config.evaluate.evaluate_jpeg_watermarked}")
                
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