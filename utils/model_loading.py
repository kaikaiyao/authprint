"""
Utilities for loading StyleGAN2 models.
"""
import logging
import os
import traceback
from typing import Dict, Tuple

from models.model_utils import load_stylegan2_model


# Dictionary of pretrained model URLs and local paths
PRETRAINED_MODELS = {
    'ffhq1k': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq1k-paper256-ada.pkl",
               "ffhq1k-paper256-ada.pkl"),
    'ffhq30k': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq30k-paper256-ada.pkl",
                "ffhq30k-paper256-ada.pkl"),
    'ffhq70k-bcr': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada-bcr.pkl",
                    "ffhq70k-paper256-ada-bcr.pkl"),
    'ffhq70k-noaug': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-noaug.pkl",
                     "ffhq70k-paper256-noaug.pkl")
}


def load_pretrained_models(device, rank=0):
    """
    Load all pretrained StyleGAN2 models.
    
    Args:
        device: Device to load models on
        rank: Distributed training rank
        
    Returns:
        Dictionary mapping model names to loaded models
    """
    models = {}
    
    # Load all models from PRETRAINED_MODELS
    for model_name, (url, local_path) in PRETRAINED_MODELS.items():
        if rank == 0:
            logging.info(f"Loading pretrained model: {model_name}")
        
        try:
            models[model_name] = load_stylegan2_model(url, local_path, device)
            models[model_name].eval()
            if rank == 0:
                logging.info(f"Successfully loaded pretrained model: {model_name}")
        except Exception as e:
            if rank == 0:
                logging.error(f"Failed to load pretrained model {model_name}: {str(e)}")
                logging.error(traceback.format_exc())
            # Skip this model and continue with others
            continue
    
    if rank == 0:
        logging.info(f"Loaded {len(models)} pretrained models: {list(models.keys())}")
    
    return models 