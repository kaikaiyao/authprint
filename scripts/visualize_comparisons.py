#!/usr/bin/env python
"""
Script to visualize and compare images generated from different models and transformations.
Takes a fixed latent vector and generates images using original model and all negative cases.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torchvision.utils as vutils
import numpy as np
from PIL import Image

# Add the parent directory (project root) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.default_config import get_default_config
from models.model_utils import load_stylegan2_model
from utils.model_loading import load_pretrained_models
from utils.image_transforms import (
    quantize_model_weights,
    downsample_and_upsample
)
from utils.logging_utils import setup_logging


def tensor_to_pil(tensor):
    """Convert a tensor to a PIL Image."""
    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    # Convert to PIL Image
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(ndarr)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize comparisons between original and negative cases")
    
    # Model configuration
    parser.add_argument("--stylegan2_url", type=str,
                        default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl",
                        help="URL for the pretrained StyleGAN2 model")
    parser.add_argument("--stylegan2_local_path", type=str,
                        default="ffhq70k-paper256-ada.pkl",
                        help="Local path to store/load the StyleGAN2 model")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image resolution")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="comparison_images",
                        help="Directory to save comparison images")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of different z vectors to generate comparisons for")
    
    return parser.parse_args()


def generate_image(model, z, noise_mode="const"):
    """Generate an image using the given model and latent vector."""
    with torch.no_grad():
        if hasattr(model, 'module'):
            w = model.module.mapping(z, None)
            img = model.module.synthesis(w, noise_mode=noise_mode)
        else:
            w = model.mapping(z, None)
            img = model.synthesis(w, noise_mode=noise_mode)
    return img


def main():
    """Main entry point for visualization."""
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(args.output_dir, 0)
    logging.info(f"Configuration:\n{args}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load original model
    original_model = load_stylegan2_model(
        args.stylegan2_url,
        args.stylegan2_local_path,
        device
    ).eval()
    
    # Load pretrained models
    pretrained_models = load_pretrained_models(device, 0)
    
    # Create quantized models
    quantized_models = {}
    try:
        quantized_models['int8'] = quantize_model_weights(original_model, 'int8').eval()
        logging.info("Successfully created int8 quantized model")
    except Exception as e:
        logging.error(f"Failed to create int8 model: {str(e)}")
    
    try:
        quantized_models['int4'] = quantize_model_weights(original_model, 'int4').eval()
        logging.info("Successfully created int4 quantized model")
    except Exception as e:
        logging.error(f"Failed to create int4 model: {str(e)}")
    
    # Generate multiple samples
    for sample_idx in range(args.num_samples):
        sample_dir = output_dir / f"sample_{sample_idx}"
        sample_dir.mkdir(exist_ok=True)
        
        # Generate fixed latent vector
        z = torch.randn(1, original_model.z_dim, device=device)
        
        # Generate and save original image
        original_img = generate_image(original_model, z)[0]
        original_pil = tensor_to_pil(original_img)
        original_pil.save(sample_dir / "original.png")
        logging.info(f"Saved original image for sample {sample_idx}")
        
        # Generate and save images from pretrained models
        for model_name, model in pretrained_models.items():
            img = generate_image(model, z)[0]
            pil_img = tensor_to_pil(img)
            pil_img.save(sample_dir / f"pretrained_{model_name}.png")
            logging.info(f"Saved image for pretrained model {model_name}")
        
        # Generate and save images from quantized models
        for quant_name, model in quantized_models.items():
            img = generate_image(model, z)[0]
            pil_img = tensor_to_pil(img)
            pil_img.save(sample_dir / f"quantized_{quant_name}.png")
            logging.info(f"Saved image for quantized model {quant_name}")
        
        # Generate and save downsampled images
        for size in [128, 224]:
            img = generate_image(original_model, z)[0]
            downsampled_img = downsample_and_upsample(img.unsqueeze(0), downsample_size=size)[0]
            pil_img = tensor_to_pil(downsampled_img)
            pil_img.save(sample_dir / f"downsampled_{size}.png")
            logging.info(f"Saved image for downsample size {size}")
        
        # Create a grid of all images
        all_images = [original_img]
        all_labels = ['original']
        
        # Add pretrained model images
        for model_name in pretrained_models.keys():
            img = generate_image(pretrained_models[model_name], z)[0]
            all_images.append(img)
            all_labels.append(f'pretrained_{model_name}')
        
        # Add quantized model images
        for quant_name in quantized_models.keys():
            img = generate_image(quantized_models[quant_name], z)[0]
            all_images.append(img)
            all_labels.append(f'quantized_{quant_name}')
        
        # Add downsampled images
        for size in [128, 224]:
            img = generate_image(original_model, z)[0]
            downsampled_img = downsample_and_upsample(img.unsqueeze(0), downsample_size=size)[0]
            all_images.append(downsampled_img)
            all_labels.append(f'downsampled_{size}')
        
        # Create and save grid
        grid = vutils.make_grid(torch.stack(all_images), nrow=4, padding=2, normalize=True)
        grid_pil = tensor_to_pil(grid)
        grid_pil.save(sample_dir / "comparison_grid.png")
        logging.info(f"Saved comparison grid for sample {sample_idx}")
        
        # Save labels
        with open(sample_dir / "image_labels.txt", "w") as f:
            for label in all_labels:
                f.write(f"{label}\n")
        
        logging.info(f"Completed processing sample {sample_idx}")
    
    logging.info("Visualization completed successfully")


if __name__ == "__main__":
    main() 