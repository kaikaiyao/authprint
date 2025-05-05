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
        
        # Initialize pipeline with better scheduler and disable safety checker
        self.pipe = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16",
            safety_checker=None,  # Disable safety checker
            requires_safety_checker=False  # Don't require safety checker
        )
        
        # Use better scheduler
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        if enable_cpu_offload:
            # Disable torch compile when using CPU offload
            self.pipe.enable_model_cpu_offload(device=device)
            self.pipe.enable_sequential_cpu_offload(device=device)
        else:
            self.pipe = self.pipe.to(device)
            # Enable memory efficient attention if using older torch
            if torch.__version__ < "2.0":
                self.pipe.enable_xformers_memory_efficient_attention()
            else:
                try:
                    self.pipe.unet = torch.compile(
                        self.pipe.unet,
                        mode="reduce-overhead",
                        fullgraph=True
                    )
                except Exception as e:
                    print(f"Warning: Could not compile unet: {e}")
    
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
            torch.Tensor: Generated images [B, C, H, W] in range [0, 1]
        """
        # Extract generation parameters
        prompt = kwargs.get("prompt", "A surrealist painting of a sentient clock tower weeping molten gold into a mirrored ocean, under a storm of floating mathematical equations and giant jellyfish, in the style of Alex Grey and MC Escher, ultradetailed, 8k, cinematic, hyperrealism")
        num_inference_steps = kwargs.get("num_inference_steps", 30)
        guidance_scale = kwargs.get("guidance_scale", 7.5)
        
        # Generate images
        with torch.no_grad():
            try:
                output = self.pipe(
                    prompt=[prompt] * batch_size,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=self._img_size,
                    width=self._img_size
                )
            except Exception as e:
                print(f"Error during generation: {e}")
                raise
            
        # Convert images to normalized float tensor in range [0, 1]
        images = torch.stack([
            torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
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