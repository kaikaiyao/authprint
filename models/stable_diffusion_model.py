import torch
from typing import Optional, Dict, Any
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models import AutoencoderKL
from .base_model import BaseGenerativeModel
import numpy as np

class StableDiffusionModel(BaseGenerativeModel):
    """Stable Diffusion model implementation."""
    
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        img_size: int = 768,
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
        # Print actual pipeline model configurations
        print(f"[Debug] Pipeline Configuration:")
        print(f"- UNet: {self.pipe.unet.config._name_or_path}")
        print(f"- VAE: {self.pipe.vae.config._name_or_path}")
        print(f"- Text Encoder: {self.pipe.text_encoder.config._name_or_path}")
        print(f"- Scheduler: {self.pipe.scheduler.__class__.__name__}")
        
        # Extract generation parameters
        prompt = kwargs.get("prompt", "A photorealistic advertisement poster for a Japanese cafe named 'NOVA CAFE', with the name written clearly in both English and Japanese on a street sign, a storefront banner, and a coffee cup. The scene is set at night with neon lighting, rain-slick streets reflecting the glow, and people walking by in motion blur. Cinematic tone, Leica photo quality, ultra-detailed textures.")
        num_inference_steps = kwargs.get("num_inference_steps", 50)
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

    def eval(self):
        """Set the model to evaluation mode."""
        self.pipe.unet.eval()
        self.pipe.vae.eval()
        self.pipe.text_encoder.eval()
        return self

    def train(self):
        """Set the model to training mode."""
        self.pipe.unet.train()
        self.pipe.vae.train()
        self.pipe.text_encoder.train()
        return self

    def quantize(self, precision='int8'):
        """Quantize model weights to specified precision.
        
        Args:
            precision (str): Quantization precision ('int8' or 'int4')
            
        Returns:
            StableDiffusionModel: New model instance with quantized weights
        """
        def quantize_tensor(tensor, bit_precision):
            # Handle empty or NaN tensors
            if tensor.numel() == 0 or torch.isnan(tensor).any():
                return tensor.clone()
                
            # Get max abs value, avoiding division by zero
            max_abs_val = torch.max(torch.abs(tensor))
            if max_abs_val == 0:
                return tensor.clone()
            
            # Set quantization parameters based on precision
            if bit_precision == 'int8':
                max_val = 127
            elif bit_precision == 'int4':
                max_val = 7  # 2^3 - 1
            else:
                # Default to int8
                max_val = 127
                
            # Scale to appropriate range
            scale = float(max_val) / max_abs_val
            quantized = torch.round(tensor * scale)
            quantized = torch.clamp(quantized, -max_val, max_val)
            
            # Scale back to original range
            dequantized = quantized / scale
            return dequantized

        # Create a new model instance
        quantized_model = StableDiffusionModel(
            model_name=self._model_name,
            device=self._device,
            img_size=self._img_size,
            dtype=self.pipe.dtype,
            enable_cpu_offload=False  # Disable CPU offload for quantized model
        )

        # Quantize UNet parameters
        with torch.no_grad():
            for name, param in quantized_model.pipe.unet.named_parameters():
                param.copy_(quantize_tensor(param, precision))

        # Quantize VAE parameters
        with torch.no_grad():
            for name, param in quantized_model.pipe.vae.named_parameters():
                param.copy_(quantize_tensor(param, precision))

        return quantized_model 