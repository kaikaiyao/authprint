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


def check_key_mapper_attributes(checkpoint_path: str, device: torch.device = torch.device('cpu')):
    """
    Check if a checkpoint contains a KeyMapper with the sine-based mapping attributes.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Device to load the checkpoint on.
        
    Returns:
        dict: A dictionary with 'has_use_sine' and 'has_sensitivity' keys.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        result = {
            'has_use_sine': False,
            'has_sensitivity': False
        }
        
        # Check if key_mapper_state exists in checkpoint
        if 'key_mapper_state' in checkpoint:
            key_mapper_state = checkpoint['key_mapper_state']
            
            # Check for _metadata first
            if '_metadata' in key_mapper_state:
                metadata = key_mapper_state['_metadata']
                if 'use_sine' in metadata:
                    result['has_use_sine'] = True
                if 'sensitivity' in metadata:
                    result['has_sensitivity'] = True
            
            # If not found in metadata, check for buffers directly
            if 'use_sine' in key_mapper_state:
                result['has_use_sine'] = True
            if 'sensitivity' in key_mapper_state:
                result['has_sensitivity'] = True
        
        return result
    except Exception as e:
        logging.warning(f"Error checking KeyMapper attributes in checkpoint: {str(e)}")
        return {'has_use_sine': False, 'has_sensitivity': False}


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
    dec = decoder.module if hasattr(decoder, 'module') else decoder
    
    # Only load watermarked model state if model is provided
    if watermarked_model is not None:
        w_model = watermarked_model.module if hasattr(watermarked_model, 'module') else watermarked_model
        w_model.load_state_dict(checkpoint['watermarked_model_state'])
    
    dec.load_state_dict(checkpoint['decoder_state'])
    
    # Check for key_mapper compatibility if provided
    if key_mapper is not None and 'key_mapper_state' in checkpoint:
        try:
            # First, check if it has the new attributes
            key_mapper_attrs = check_key_mapper_attributes(checkpoint_path, device)
            
            # If loading an old checkpoint into a new KeyMapper, ensure backward compatibility
            if not key_mapper_attrs['has_use_sine'] and hasattr(key_mapper, 'use_sine'):
                # Old checkpoint, new KeyMapper model
                logging.info("Loading checkpoint from a model without sine-based mapping capability")
                
                # If current KeyMapper is configured to use sine, warn about incompatibility
                if key_mapper.use_sine:
                    logging.warning("Current KeyMapper is configured to use sine-based mapping, "
                                   "but loading a checkpoint trained without it. "
                                   "Setting use_sine=False for compatibility.")
                    key_mapper.use_sine = False
            
            # Load state dict - this will ignore missing keys in the checkpoint
            key_mapper.load_state_dict(checkpoint['key_mapper_state'], strict=False)
            
        except Exception as e:
            logging.warning(f"Error loading KeyMapper state: {str(e)}. "
                           "Continuing with possibly incomplete state.")
    
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    logging.info(f"Loaded checkpoint from {checkpoint_path} (iteration {checkpoint.get('iteration', 'unknown')})")
    
    return checkpoint 