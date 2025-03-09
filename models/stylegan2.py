import os
import sys
import torch
from torch.distributed import is_initialized, get_rank
from utils.file_utils import download_file
import logging

sys.path.append("./stylegan2-ada-pytorch")
import dnnlib
import legacy

def load_stylegan2_model(url: str, local_path: str, device: torch.device) -> torch.nn.Module:
    """Load a pre-trained StyleGAN2 model from a URL or local path."""
    if not os.path.exists(local_path):
        logging.info(f"Downloading StyleGAN2 model to {local_path}...")
        torch.hub.download_url_to_file(url, local_path)
        logging.info("Download complete.")

    with dnnlib.util.open_url(local_path) as f:
        model = legacy.load_network_pkl(f)['G_ema'].to(device)
    return model

def is_stylegan2(model: torch.nn.Module) -> bool:
    """Check if a model is a StyleGAN2 model."""
    return hasattr(model, 'synthesis') and hasattr(model, 'mapping')