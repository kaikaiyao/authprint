"""
Utilities for loading StyleGAN2 models.
"""
import logging
import os
import traceback
from typing import Dict, Tuple, Optional

from models.model_utils import load_stylegan2_model


# Dictionary of pretrained model URLs and local paths
PRETRAINED_MODELS = {
    # FFHQ Models
    'ffhq70k-ada': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl",
               "ffhq70k-paper256-ada.pkl"), # this serves as a control test
    'ffhq1k': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq1k-paper256-ada.pkl",
               "ffhq1k-paper256-ada.pkl"),
    'ffhq30k': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq30k-paper256-ada.pkl",
                "ffhq30k-paper256-ada.pkl"),
    'ffhq70k-bcr': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada-bcr.pkl",
                    "ffhq70k-paper256-ada-bcr.pkl"),
    'ffhq70k-noaug': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-noaug.pkl",
                     "ffhq70k-paper256-noaug.pkl"),
    
    # LSUN Cat Models
    'lsuncat100k-ada': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/lsuncat100k-paper256-ada.pkl",
                        "lsuncat100k-paper256-ada.pkl"),
    'lsuncat1k': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/lsuncat1k-paper256-ada.pkl",
                  "lsuncat1k-paper256-ada.pkl"),
    'lsuncat30k': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/lsuncat30k-paper256-ada.pkl",
                   "lsuncat30k-paper256-ada.pkl"),
    'lsuncat100k-bcr': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/lsuncat100k-paper256-ada-bcr.pkl",
                        "lsuncat100k-paper256-ada-bcr.pkl"),
    'lsuncat100k-noaug': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/lsuncat100k-paper256-noaug.pkl",
                          "lsuncat100k-paper256-noaug.pkl")
}


def load_pretrained_models(device, rank=0, model_dict: Optional[Dict[str, Tuple[str, str]]] = None):
    """
    Load pretrained StyleGAN2 models.
    
    Args:
        device: Device to load models on
        rank: Distributed training rank
        model_dict: Optional dictionary mapping model names to (url, local_path) tuples.
                   If None, uses PRETRAINED_MODELS.
        
    Returns:
        Dictionary mapping model names to loaded models
    """
    models = {}
    
    # Use provided model dictionary or default to PRETRAINED_MODELS
    model_dict = model_dict if model_dict is not None else PRETRAINED_MODELS
    
    # Load all models from model dictionary
    for model_name, (url, local_path) in model_dict.items():
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