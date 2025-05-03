import torch
from typing import Optional, Dict, Any
from diffusers import DiffusionPipeline, EulerDiscreteScheduler
from .base_model import BaseGenerativeModel
import numpy as np

class StableDiffusionModel(BaseGenerativeModel):
    """Stable Diffusion model implementation."""
    
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        img_size: int = 1024,  # SDXL default
        dtype: torch.dtype = torch.float16,
        enable_cpu_offload: bool = False
    ):
        """Initialize Stable Diffusion model.
        
        Args:
            model_name (str): HuggingFace model name
            device (torch.device): Device to load model on
            img_size (int): Output image size
            dtype (torch.dtype): Model dtype
            enable_cpu_offload (bool): Whether to enable CPU offloading
        """
        self._device = device
        self._img_size = img_size
        self._model_name = model_name
        
        # Initialize pipeline with better scheduler
        self.pipe = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16"
        )
        
        # Use better scheduler
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        if enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(device)
            
        # Enable memory efficient attention if using older torch
        if torch.__version__ < "2.0":
            self.pipe.enable_xformers_memory_efficient_attention()
        else:
            # Use torch.compile for better performance
            self.pipe.unet = torch.compile(
                self.pipe.unet,
                mode="reduce-overhead",
                fullgraph=True
            )
    
    def generate_images(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate images using Stable Diffusion.
        
        Args:
            batch_size (int): Number of images to generate
            device (torch.device, optional): Device override
            **kwargs: Additional arguments passed to pipeline
                prompt (str): Text prompt
                num_inference_steps (int): Number of denoising steps
                guidance_scale (float): Classifier-free guidance scale
                
        Returns:
            torch.Tensor: Generated images [B, C, H, W]
        """
        # Extract generation parameters
        prompt = kwargs.get("prompt", "A high quality photo")
        num_inference_steps = kwargs.get("num_inference_steps", 30)
        guidance_scale = kwargs.get("guidance_scale", 7.5)
        
        # Generate images
        with torch.no_grad():
            output = self.pipe(
                prompt=[prompt] * batch_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
        # Convert images to tensor
        images = torch.stack([
            torch.from_numpy(np.array(img)).permute(2, 0, 1)
            for img in output.images
        ]).to(device or self._device)
        
        return images
    
    def get_model_type(self) -> str:
        return "stable-diffusion"
    
    def get_model_name(self) -> str:
        return self._model_name.split("/")[-1]
        
    @property
    def image_size(self) -> int:
        return self._img_size 