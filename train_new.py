#!/usr/bin/env python
import argparse
import copy
import logging
import os
import pickle
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import lpips

# Append the stylegan2-ada-pytorch folder (assumes you cloned it)
sys.path.append("./stylegan2-ada-pytorch")
import dnnlib
import legacy

# --------------
# Utility Functions (from your provided code)
# --------------
def save_finetuned_model(model, path, filename):
    model_cpu = copy.deepcopy(model).cpu()
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    with open(save_path, 'wb') as f:
        pickle.dump(model_cpu, f)

def load_finetuned_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def clone_model(model):
    """
    Clones a model and ensures all parameters in the cloned model require gradients.
    """
    cloned_model = copy.deepcopy(model)
    cloned_model.train()
    for param in cloned_model.parameters():
        param.requires_grad = True
    return cloned_model

def load_stylegan2_model(url: str, local_path: str, device: torch.device) -> torch.nn.Module:
    """Load a pre-trained StyleGAN2 model from a URL or local path."""
    if not os.path.exists(local_path):
        logging.info(f"Downloading StyleGAN2 model to {local_path}...")
        torch.hub.download_url_to_file(url, local_path)
        logging.info("Download complete.")
    with dnnlib.util.open_url(local_path) as f:
        # Load the pickle and extract the generator 'G_ema'
        model = legacy.load_network_pkl(f)['G_ema'].to(device)
    return model

# --------------
# Model Definitions
# --------------

class KeyMapper(nn.Module):
    """
    Fixed secret mapping: maps a 32-element z_partial to a 4-bit binary key.
    """
    def __init__(self, input_dim=32, output_dim=4):
        super(KeyMapper, self).__init__()
        # Secret parameters (fixed, non-trainable)
        self.register_buffer('W', torch.randn(input_dim, output_dim))
        self.register_buffer('b', torch.randn(output_dim))
    
    def forward(self, z_partial):
        # Linear projection + tanh activation
        projection = torch.matmul(z_partial, self.W) + self.b
        activated = torch.tanh(projection)
        target = (activated > 0).float()  # binary output
        return target

class Decoder(nn.Module):
    """
    Decoder network that predicts 4-bit key logits from an input image.
    With increased capacity for better convergence.
    """
    def __init__(self, image_size=256, channels=3, output_dim=4):
        super(Decoder, self).__init__()
        # Increase number of filters and add more layers
        self.features = nn.Sequential(
            # Initial layer: 256 -> 128
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 -> 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 -> 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32 -> 16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16 -> 8
            nn.Conv2d(512, 768, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(768),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8 -> 4
            nn.Conv2d(768, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Global pooling and deeper classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)

# --------------
# Logging Setup
# --------------
def setup_logging(output_dir, rank):
    try:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f"train_rank{rank}.log")
        logging.basicConfig(
            level=logging.INFO if rank == 0 else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler() if rank == 0 else logging.NullHandler()
            ]
        )
    except Exception as e:
        # Fallback to console logging if file logging fails
        print(f"Warning: Failed to setup file logging: {str(e)}")
        logging.basicConfig(
            level=logging.INFO if rank == 0 else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler() if rank == 0 else logging.NullHandler()]
        )

def save_checkpoint(iteration, watermarked_model, decoder, output_dir, rank):
    if rank == 0:
        ckpt_path = os.path.join(output_dir, f"checkpoint_iter{iteration}.pth")
        # Handle DDP-wrapped models by accessing .module if needed
        w_model = watermarked_model.module if hasattr(watermarked_model, 'module') else watermarked_model
        dec = decoder.module if hasattr(decoder, 'module') else decoder
        
        torch.save({
            'watermarked_model': w_model.state_dict(),
            'decoder': dec.state_dict()
        }, ckpt_path)
        logging.info(f"Saved checkpoint at iteration {iteration} to {ckpt_path}")

# --------------
# Main Training Function
# --------------
def main(args):
    try:
        # Distributed setup
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        setup_logging(args.output_dir, rank)
        logging.info(f"Rank: {rank}, World Size: {world_size}, Device: {device}")

        # ------------------------------
        # Load Pretrained StyleGAN2 Model
        # ------------------------------
        gan_model = load_stylegan2_model(args.stylegan2_url, args.stylegan2_local_path, device)
        latent_dim = gan_model.z_dim
        logging.info(f"Pretrained model loaded with z_dim={latent_dim}")

        # Freeze the pretrained model (for LPIPS computation)
        gan_model.eval()
        for param in gan_model.parameters():
            param.requires_grad = False

        # ------------------------------
        # Initialize Watermarked Model
        # ------------------------------
        # Clone the pretrained model to obtain a trainable watermarked model
        watermarked_model = clone_model(gan_model)
        watermarked_model.to(device)
        watermarked_model.train()

        if world_size > 1:
            watermarked_model = DDP(watermarked_model, device_ids=[rank])

        # ------------------------------
        # Initialize Decoder and KeyMapper
        # ------------------------------
        decoder = Decoder(image_size=args.img_size, channels=3, output_dim=4).to(device)
        if world_size > 1:
            decoder = DDP(decoder, device_ids=[rank])

        # Fixed indices for z_partial (passed as comma-separated string)
        z_indices = [int(idx) for idx in args.z_indices.split(',')]
        logging.info(f"Using z_partial indices: {z_indices}")
        
        # Validate z_indices
        if max(z_indices) >= latent_dim:
            raise ValueError(f"z_indices contains indices >= latent_dim ({latent_dim}). Max index: {max(z_indices)}")
        if len(z_indices) != len(set(z_indices)):
            logging.warning("z_indices contains duplicate indices which may not be intended")
        
        key_mapper = KeyMapper(input_dim=len(z_indices), output_dim=4).to(device)

        # Loss functions
        bce_loss_fn = nn.BCEWithLogitsLoss()
        lpips_loss_fn = lpips.LPIPS(net='alex').to(device)

        # Optimizer (jointly train watermarked model and decoder)
        optimizer = optim.Adam(list(watermarked_model.parameters()) + list(decoder.parameters()), lr=args.lr)

        # Set models to training mode
        watermarked_model.train()
        decoder.train()
        
        global_step = 0
        for iteration in range(1, args.total_iterations + 1):
            optimizer.zero_grad()
            # Sample a batch of latent vectors z: shape (batch_size, latent_dim)
            z = torch.randn(args.batch_size, latent_dim, device=device)

            # Generate watermarked image from watermarked model
            # Using the StyleGAN2-ADA API: pass z and other arguments as needed.
            # Here, noise_mode is set to "const" for consistency.
            x_water = watermarked_model(z, None, truncation_psi=1.0, noise_mode="const")
            # Compute the original image using the frozen pretrained model for LPIPS loss
            with torch.no_grad():
                x_orig = gan_model(z, None, truncation_psi=1.0, noise_mode="const")

            # Extract z_partial using fixed indices
            z_partial = z[:, z_indices]  # shape: (batch_size, len(z_indices))
            true_key = key_mapper(z_partial)  # shape: (batch_size, 4) with binary values

            # Predict key from watermarked image using the decoder (raw logits)
            pred_key_logits = decoder(x_water)

            # Compute key loss (BCE with logits)
            key_loss = bce_loss_fn(pred_key_logits, true_key)
            # Compute LPIPS loss between original and watermarked images
            lpips_loss = lpips_loss_fn(x_orig, x_water).mean()
            total_loss = key_loss + args.lambda_lpips * lpips_loss

            total_loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % args.log_interval == 0 and rank == 0:
                logging.info(f"Iteration [{iteration}/{args.total_iterations}] "
                         f"Key Loss: {key_loss.item():.4f}, LPIPS Loss: {lpips_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")
            
            # Save checkpoint at regular intervals
            if iteration % args.checkpoint_interval == 0:
                save_checkpoint(iteration, watermarked_model, decoder, args.output_dir, rank)

        # Final checkpoint is already saved by the interval logic if total_iterations is divisible by checkpoint_interval
        # Otherwise, save the final state explicitly
        if args.total_iterations % args.checkpoint_interval != 0:
            save_checkpoint(args.total_iterations, watermarked_model, decoder, args.output_dir, rank)

        if world_size > 1:
            dist.destroy_process_group()
            
    except Exception as e:
        logging.error(f"Error in training: {str(e)}", exc_info=True)
        if 'world_size' in locals() and world_size > 1:
            dist.destroy_process_group()
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watermarking Training Pipeline for StyleGAN2-ADA")
    parser.add_argument("--stylegan2_url", type=str,
                        default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl",
                        help="URL for the pretrained StyleGAN2 model")
    parser.add_argument("--stylegan2_local_path", type=str,
                        default="ffhq70k-paper256-ada.pkl",
                        help="Local path to store/load the StyleGAN2 model")
    parser.add_argument("--img_size", type=int, default=256, help="Image resolution")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--total_iterations", type=int, default=100000, help="Total number of training iterations")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda_lpips", type=float, default=0.5, help="Weight for LPIPS loss")
    parser.add_argument("--z_indices", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31",
                        help="Comma-separated list of indices to select for z_partial (should total 32 indices)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save logs and checkpoints")
    parser.add_argument("--log_interval", type=int, default=1, help="Interval for logging training progress")
    parser.add_argument("--checkpoint_interval", type=int, default=10000, help="Interval for saving checkpoints")
    args = parser.parse_args()
    
    main(args)
