"""
Checkpoint utilities for saving and loading model checkpoints.
"""
import logging
import os
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def save_checkpoint(
    iteration: int,
    decoder: Union[nn.Module, DDP],
    output_dir: str,
    rank: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metrics: Optional[Dict] = None,
    global_step: Optional[int] = None
) -> None:
    """
    Save a checkpoint of the decoder model and training state.
    
    Args:
        iteration (int): Current iteration number.
        decoder (nn.Module or DDP): The decoder model.
        output_dir (str): Directory to save the checkpoint to.
        rank (int): Process rank in distributed training.
        optimizer (torch.optim.Optimizer, optional): Optimizer to save.
        metrics (Dict, optional): Current training metrics.
        global_step (int, optional): Global step counter for training progress.
    """
    if rank != 0:
        return  # Only save from the master process
    
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"checkpoint_iter{iteration}.pth")
    
    # Handle DDP-wrapped model by accessing .module if needed
    dec = decoder.module if hasattr(decoder, 'module') else decoder
    
    checkpoint = {
        'iteration': iteration,
        'decoder_state': dec.state_dict(),
        'global_step': global_step,
        'metrics': metrics
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state'] = optimizer.state_dict()
    
    torch.save(checkpoint, ckpt_path)
    logging.info(f"Saved checkpoint at iteration {iteration} to {ckpt_path}")


def load_checkpoint(
    checkpoint_path: str,
    decoder: Union[nn.Module, DDP],
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Load a checkpoint into decoder model and optimizer.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        decoder (nn.Module or DDP): The decoder model.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        device (torch.device): Device to load the checkpoint onto.
        
    Returns:
        dict: The loaded checkpoint with training metadata.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint to CPU first to avoid potential CUDA memory issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load decoder state
    dec = decoder.module if hasattr(decoder, 'module') else decoder
    dec.load_state_dict(checkpoint['decoder_state'])
    
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    logging.info(f"Loaded checkpoint from {checkpoint_path} (iteration {checkpoint.get('iteration', 'unknown')})")
    
    return checkpoint 