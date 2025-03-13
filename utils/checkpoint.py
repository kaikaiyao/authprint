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
    watermarked_model: Union[nn.Module, DDP],
    decoder: Union[nn.Module, DDP],
    output_dir: str,
    rank: int,
    key_mapper: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    global_step: Optional[int] = None,
    **kwargs
) -> None:
    """
    Save a checkpoint of the model and optimizer.
    
    Args:
        iteration (int): Current iteration number.
        watermarked_model (nn.Module or DDP): The watermarked generator model.
        decoder (nn.Module or DDP): The decoder model.
        output_dir (str): Directory to save the checkpoint to.
        rank (int): Process rank in distributed training.
        key_mapper (nn.Module, optional): The key mapper model to save.
        optimizer (torch.optim.Optimizer, optional): Optimizer to save.
        global_step (int, optional): Global step counter for training progress.
        **kwargs: Additional items to save in the checkpoint.
    """
    if rank != 0:
        return  # Only save from the master process
    
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"checkpoint_iter{iteration}.pth")
    
    # Handle DDP-wrapped models by accessing .module if needed
    w_model = watermarked_model.module if hasattr(watermarked_model, 'module') else watermarked_model
    dec = decoder.module if hasattr(decoder, 'module') else decoder
    
    checkpoint = {
        'iteration': iteration,
        'watermarked_model_state': w_model.state_dict(),
        'decoder_state': dec.state_dict(),
        'global_step': global_step
    }
    
    # Save key_mapper state if provided
    if key_mapper is not None:
        checkpoint['key_mapper_state'] = key_mapper.state_dict()
    
    if optimizer is not None:
        checkpoint['optimizer_state'] = optimizer.state_dict()
    
    # Add any additional items
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, ckpt_path)
    logging.info(f"Saved checkpoint at iteration {iteration} to {ckpt_path}")


def load_checkpoint(
    checkpoint_path: str,
    watermarked_model: Union[nn.Module, DDP],
    decoder: Union[nn.Module, DDP],
    optimizer: Optional[torch.optim.Optimizer] = None,
    key_mapper: Optional[nn.Module] = None,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Load a checkpoint into model and optimizer.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        watermarked_model (nn.Module or DDP): The watermarked generator model.
        decoder (nn.Module or DDP): The decoder model.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        key_mapper (nn.Module, optional): The key mapper model to load.
        device (torch.device): Device to load the checkpoint onto.
        
    Returns:
        dict: The loaded checkpoint with additional metadata.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DDP-wrapped models
    w_model = watermarked_model.module if hasattr(watermarked_model, 'module') else watermarked_model
    dec = decoder.module if hasattr(decoder, 'module') else decoder
    
    w_model.load_state_dict(checkpoint['watermarked_model_state'])
    dec.load_state_dict(checkpoint['decoder_state'])
    
    # Load key_mapper state if provided and exists in checkpoint
    if key_mapper is not None and 'key_mapper_state' in checkpoint:
        key_mapper.load_state_dict(checkpoint['key_mapper_state'])
    
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    logging.info(f"Loaded checkpoint from {checkpoint_path} (iteration {checkpoint.get('iteration', 'unknown')})")
    
    return checkpoint 