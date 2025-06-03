#!/usr/bin/env python
"""
Script to compute FID scores between Stable Diffusion 2.1 and other SD models (1.5 to 1.1).
Supports multi-GPU processing for faster image generation and incremental FID computation.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Generator

import torch
import torch.distributed as dist
import numpy as np
import torchvision

# Add the parent directory (project root) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.stable_diffusion_model import StableDiffusionModel
from utils.metrics import calculate_fid
from utils.distributed import setup_distributed, cleanup_distributed
from utils.logging_utils import setup_logging
from utils.model_loading import STABLE_DIFFUSION_MODELS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute FID scores between SD 2.1 and other SD models")
    
    # Model configuration
    parser.add_argument("--reference_model", type=str, default="sd-2.1",
                        help="Reference model to compare against (default: sd-2.1)")
    parser.add_argument("--comparison_models", type=str, nargs='+',
                        default=["sd-1.5", "sd-1.4", "sd-1.3", "sd-1.2", "sd-1.1"],
                        help="List of models to compare with the reference model")
    
    # Generation configuration
    parser.add_argument("--prompt", type=str,
                        default="A photorealistic advertisement poster for a Japanese cafe named 'NOVA CAFE', with the name written clearly in both English and Japanese on a street sign, a storefront banner, and a coffee cup. The scene is set at night with neon lighting, rain-slick streets reflecting the glow, and people walking by in motion blur. Cinematic tone, Leica photo quality, ultra-detailed textures.",
                        help="Prompt for image generation")
    parser.add_argument("--img_size", type=int, default=1024,
                        help="Size of generated images")
    parser.add_argument("--num_images", type=int, default=10000,
                        help="Number of images to generate for FID calculation")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per GPU")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Number of images to process at once for FID computation")
    parser.add_argument("--num_inference_steps", type=int, default=25,
                        help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    
    # Hardware configuration
    parser.add_argument("--enable_cpu_offload", action="store_true",
                        help="Enable CPU offloading for SD models")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32"],
                        help="Model dtype")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="fid_results",
                        help="Directory to save results")
    parser.add_argument("--save_images", action="store_true",
                        help="Save generated images for inspection")
    
    return parser.parse_args()


def generate_images_distributed(
    model: StableDiffusionModel,
    num_images: int,
    batch_size: int,
    rank: int,
    world_size: int,
    chunk_size: int,
    **kwargs
) -> Generator[torch.Tensor, None, None]:
    """Generate images in a distributed manner across GPUs.
    
    Args:
        model: SD model to generate images with
        num_images: Total number of images to generate
        batch_size: Batch size per GPU
        rank: Current process rank
        world_size: Total number of processes
        chunk_size: Number of images to generate per chunk
        **kwargs: Additional arguments for image generation
        
    Yields:
        torch.Tensor: Generated images [N, C, H, W] for each chunk
    """
    # Calculate number of images per GPU
    images_per_gpu = num_images // world_size
    if rank < num_images % world_size:
        images_per_gpu += 1
    
    # Process in chunks
    for chunk_start in range(0, images_per_gpu, chunk_size):
        chunk_end = min(chunk_start + chunk_size, images_per_gpu)
        chunk_images = []
        
        # Generate images for this chunk
        for i in range(chunk_start, chunk_end, batch_size):
            current_batch_size = min(batch_size, chunk_end - i)
            batch = model.generate_images(
                batch_size=current_batch_size,
                **kwargs
            )
            chunk_images.append(batch)
            
            if rank == 0:
                total_progress = (i + current_batch_size) * world_size
                logging.info(f"Generated {total_progress}/{num_images} images")
        
        # Yield this chunk's images
        yield torch.cat(chunk_images, dim=0)
        
        # Clear memory
        del chunk_images
        torch.cuda.empty_cache()


def compute_fid_scores(args, rank: int, world_size: int, device: torch.device) -> Dict[str, float]:
    """Compute FID scores between reference model and comparison models.
    
    Args:
        args: Command line arguments
        rank: Process rank
        world_size: Total number of processes
        device: Torch device
        
    Returns:
        Dict[str, float]: FID scores for each comparison
    """
    # Initialize models
    if rank == 0:
        logging.info("Initializing models...")
    
    # Initialize reference model
    ref_model = StableDiffusionModel(
        model_name=STABLE_DIFFUSION_MODELS[args.reference_model],
        device=device,
        img_size=args.img_size,
        dtype=getattr(torch, args.dtype),
        enable_cpu_offload=args.enable_cpu_offload
    )
    
    # Compute FID scores for each comparison model
    fid_scores = {}
    for model_name in args.comparison_models:
        if rank == 0:
            logging.info(f"\nComputing FID between {args.reference_model} and {model_name}...")
        
        # Initialize comparison model
        comp_model = StableDiffusionModel(
            model_name=STABLE_DIFFUSION_MODELS[model_name],
            device=device,
            img_size=args.img_size,
            dtype=getattr(torch, args.dtype),
            enable_cpu_offload=args.enable_cpu_offload
        )
        
        # Initialize FID statistics
        ref_activations = []
        comp_activations = []
        
        # Generate and process images in chunks
        ref_generator = generate_images_distributed(
            model=ref_model,
            num_images=args.num_images,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            chunk_size=args.chunk_size,
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale
        )
        
        comp_generator = generate_images_distributed(
            model=comp_model,
            num_images=args.num_images,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            chunk_size=args.chunk_size,
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale
        )
        
        # Process chunks
        for chunk_idx, (ref_chunk, comp_chunk) in enumerate(zip(ref_generator, comp_generator)):
            # Save images if requested
            if args.save_images and rank == 0:
                save_dir = Path(args.output_dir) / "images"
                for model, chunk, prefix in [(args.reference_model, ref_chunk, "ref"), (model_name, comp_chunk, "comp")]:
                    model_dir = save_dir / model
                    model_dir.mkdir(parents=True, exist_ok=True)
                    for i, img in enumerate(chunk):
                        img_path = model_dir / f"{prefix}_chunk{chunk_idx}_img{i:05d}.png"
                        torchvision.utils.save_image(img, img_path)
            
            # Update FID statistics
            chunk_fid = calculate_fid(
                ref_chunk,
                comp_chunk,
                batch_size=args.batch_size,
                device=device
            )
            
            if rank == 0:
                logging.info(f"Chunk {chunk_idx + 1} FID: {chunk_fid:.4f}")
            
            # Accumulate activations for final FID computation
            ref_activations.append(ref_chunk)
            comp_activations.append(comp_chunk)
            
            # Clear memory after processing each chunk
            del ref_chunk, comp_chunk
            torch.cuda.empty_cache()
        
        # Compute final FID score using all accumulated activations
        final_fid = calculate_fid(
            torch.cat(ref_activations, dim=0),
            torch.cat(comp_activations, dim=0),
            batch_size=args.batch_size,
            device=device
        )
        
        if rank == 0:
            logging.info(f"Final FID score between {args.reference_model} and {model_name}: {final_fid:.4f}")
        
        fid_scores[model_name] = final_fid
        
        # Clean up
        del ref_activations, comp_activations, comp_model
        torch.cuda.empty_cache()
    
    # Clean up reference model
    del ref_model
    torch.cuda.empty_cache()
    
    return fid_scores


def main():
    """Main entry point for FID computation."""
    args = parse_args()
    
    # Setup distributed processing
    local_rank, rank, world_size, device = setup_distributed()
    
    # Create output directory and setup logging
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        setup_logging(args.output_dir, rank)
        logging.info(f"Configuration:\n{args}")
        logging.info(f"Distributed setup: {world_size} GPUs")
    
    try:
        # Compute FID scores
        fid_scores = compute_fid_scores(args, rank, world_size, device)
        
        # Save results
        if rank == 0:
            results_file = Path(args.output_dir) / "fid_scores.txt"
            with open(results_file, "w") as f:
                f.write(f"FID Scores (Reference: {args.reference_model})\n")
                f.write("-" * 50 + "\n")
                for model_name, score in fid_scores.items():
                    f.write(f"{model_name}: {score:.4f}\n")
            
            logging.info(f"\nResults saved to {results_file}")
    
    except Exception as e:
        if rank == 0:
            logging.error(f"Error during FID computation: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main() 