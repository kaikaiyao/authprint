import torch
from typing import Optional
from .base_model import BaseGenerativeModel
from models.model_utils import load_stylegan2_model

class StyleGAN2Model(BaseGenerativeModel):
    """StyleGAN2 model implementation."""
    
    def __init__(
        self,
        model_url: str,
        model_path: str,
        device: torch.device,
        img_size: int = 256
    ):
        """Initialize StyleGAN2 model.
        
        Args:
            model_url (str): URL to download model from
            model_path (str): Local path to save/load model
            device (torch.device): Device to load model on
            img_size (int): Output image size
        """
        self._device = device
        self._img_size = img_size
        self.model = load_stylegan2_model(model_url, model_path, device)
        self.model.eval()
        
    def generate_images(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate images using StyleGAN2.
        
        Args:
            batch_size (int): Number of images to generate
            device (torch.device, optional): Device override
            **kwargs: Additional arguments (unused)
            
        Returns:
            torch.Tensor: Generated images [B, C, H, W]
        """
        device = device or self._device
        z = torch.randn(batch_size, self.z_dim, device=device)
        
        with torch.no_grad():
            if hasattr(self.model, 'module'):
                w = self.model.module.mapping(z, None)
                images = self.model.module.synthesis(w, noise_mode="const")
            else:
                w = self.model.mapping(z, None)
                images = self.model.synthesis(w, noise_mode="const")
                
        return images
    
    def get_model_type(self) -> str:
        return "stylegan2"
    
    def get_model_name(self) -> str:
        return "stylegan2-ada"
        
    @property
    def image_size(self) -> int:
        return self._img_size

    @property
    def z_dim(self) -> int:
        """Get the latent dimension size."""
        if hasattr(self.model, 'module'):
            return self.model.module.z_dim
        return self.model.z_dim 