"""
Evaluator for StyleGAN watermarking.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import lpips
import numpy as np
from tqdm import tqdm
import sklearn.metrics as skmetrics

from config.default_config import Config
from models.decoder import Decoder
from models.key_mapper import KeyMapper
from models.model_utils import clone_model, load_stylegan2_model
from utils.checkpoint import load_checkpoint
from utils.metrics import calculate_metrics, save_metrics_plots, save_metrics_text
from utils.visualization import save_visualization
from utils.image_transforms import (
    apply_truncation, 
    quantize_model_weights, 
    downsample_and_upsample, 
    apply_jpeg_compression
)
from utils.model_loading import load_pretrained_models


class WatermarkEvaluator:
    """
    Evaluator for StyleGAN watermarking.
    """
    def __init__(
        self,
        config: Config,
        local_rank: int,
        rank: int,
        world_size: int,
        device: torch.device
    ):
        """
        Initialize the evaluator.
        
        Args:
            config (Config): Configuration object.
            local_rank (int): Local process rank.
            rank (int): Global process rank.
            world_size (int): Total number of processes.
            device (torch.device): Device to run evaluation on.
        """
        self.config = config
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.device = device
        
        # Determine which approach to use
        self.use_image_pixels = self.config.model.use_image_pixels
        
        if self.use_image_pixels:
            # For image-based approach
            self.image_pixel_indices = None
            self.image_pixel_count = self.config.model.image_pixel_count
            self.image_pixel_set_seed = self.config.model.image_pixel_set_seed
            if self.rank == 0:
                logging.info(f"Using image-based approach with {self.image_pixel_count} pixels and seed {self.image_pixel_set_seed}")
        else:
            # For latent-based approach
            self.latent_indices = None
            
            # Check if explicit indices are provided
            if hasattr(self.config.model, 'selected_indices') and self.config.model.selected_indices is not None:
                if isinstance(self.config.model.selected_indices, str):
                    self.latent_indices = [int(idx) for idx in self.config.model.selected_indices.split(',')]
                else:
                    self.latent_indices = self.config.model.selected_indices
                if self.rank == 0:
                    logging.info(f"Using manually specified latent indices: {self.latent_indices}")
            else:
                # We'll generate random indices later once we know the latent dimension
                # For backward compatibility, default to 32 if w_partial_length not present
                self.w_partial_length = getattr(self.config.model, 'w_partial_length', 32)
                # Default seed for backward compatibility
                self.w_partial_set_seed = getattr(self.config.model, 'w_partial_set_seed', 42)
                if self.rank == 0:
                    logging.info(f"Will generate {self.w_partial_length} latent indices with seed {self.w_partial_set_seed}")
        
        # Setup models
        self.setup_models()
        
        # Load additional pretrained models if needed
        self.pretrained_models = {}
        if hasattr(config.evaluate, 'evaluate_pretrained') and config.evaluate.evaluate_pretrained:
            self.pretrained_models = load_pretrained_models(config, device, rank)
        
        # Setup the quantized model if we're evaluating quantization
        self.quantized_model = None
        self.quantized_watermarked_model = None
        if getattr(self.config.evaluate, 'evaluate_quantization', False):
            # Setup quantized version of the original model
            if self.rank == 0:
                logging.info("Setting up quantized model...")
            self.quantized_model = quantize_model_weights(self.gan_model)
            self.quantized_model.eval()
            
            # Setup quantized version of the watermarked model
            if getattr(self.config.evaluate, 'evaluate_quantization_watermarked', False):
                if self.rank == 0:
                    logging.info("Setting up quantized watermarked model...")
                self.quantized_watermarked_model = quantize_model_weights(self.watermarked_model)
                self.quantized_watermarked_model.eval()
    
    def _generate_pixel_indices(self) -> None:
        """
        Generate random pixel indices for image-based approach.
        """
        # Set seed for reproducibility
        np.random.seed(self.image_pixel_set_seed)
        
        # Calculate total number of pixels
        img_size = self.config.model.img_size
        channels = 3  # RGB image
        total_pixels = channels * img_size * img_size
        
        # Generate random indices (without replacement)
        if self.image_pixel_count > total_pixels:
            if self.rank == 0:
                logging.warning(f"Requested {self.image_pixel_count} pixels exceeds total pixels {total_pixels}. Using all pixels.")
            self.image_pixel_count = total_pixels
            self.image_pixel_indices = np.arange(total_pixels)
        else:
            self.image_pixel_indices = np.random.choice(
                total_pixels, 
                size=self.image_pixel_count, 
                replace=False
            )
        if self.rank == 0:
            logging.info(f"Generated {len(self.image_pixel_indices)} pixel indices with seed {self.image_pixel_set_seed}")

    def extract_image_partial(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract partial image using selected pixel indices.
        
        Args:
            images (torch.Tensor): Batch of images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Batch of flattened pixel values at selected indices
        """
        batch_size = images.shape[0]
        
        # Flatten the spatial dimensions: [batch_size, channels*height*width]
        flattened = images.reshape(batch_size, -1)
        
        # Get values at selected indices: [batch_size, pixel_count]
        image_partial = flattened[:, self.image_pixel_indices]
        
        return image_partial
    
    def _generate_latent_indices(self, latent_dim: int) -> None:
        """
        Generate random latent indices for latent-based approach.
        
        Args:
            latent_dim (int): Dimension of the latent space.
        """
        # Set seed for reproducibility
        np.random.seed(self.w_partial_set_seed)
        
        # Generate random indices (without replacement)
        if self.w_partial_length > latent_dim:
            if self.rank == 0:
                logging.warning(f"Requested {self.w_partial_length} indices exceeds latent dimension {latent_dim}. Using all dimensions.")
            self.w_partial_length = latent_dim
            self.latent_indices = np.arange(latent_dim)
        else:
            self.latent_indices = np.random.choice(
                latent_dim, 
                size=self.w_partial_length, 
                replace=False
            )
        if self.rank == 0:
            logging.info(f"Generated {len(self.latent_indices)} latent indices with seed {self.w_partial_set_seed}")
            
    def setup_models(self):
        """
        Setup models for evaluation.
        """
        # Load the original StyleGAN2 model
        if self.rank == 0:
            logging.info("Loading StyleGAN2 model...")
        self.gan_model = load_stylegan2_model(
            self.config.model.stylegan2_url,
            self.config.model.stylegan2_local_path,
            self.device
        )
        self.gan_model.eval()
        self.latent_dim = self.gan_model.z_dim
        
        # Generate latent indices if not provided explicitly
        if not self.use_image_pixels and self.latent_indices is None:
            self._generate_latent_indices(self.latent_dim)
        
        # Clone it to create watermarked model
        self.watermarked_model = clone_model(self.gan_model)
        self.watermarked_model.to(self.device)
        self.watermarked_model.eval()
        
        # Initialize decoder model
        self.decoder = Decoder(
            image_size=self.config.model.img_size,
            channels=3,
            output_dim=self.config.model.key_length
        ).to(self.device)
        self.decoder.eval()
        
        # Initialize key mapper with specified seed
        if self.rank == 0:
            logging.info(f"Initializing key mapper with seed {self.config.model.key_mapper_seed}")
        
        # Determine key mapper input dimension based on approach
        if self.use_image_pixels:
            # Generate pixel indices for image-based approach
            self._generate_pixel_indices()
            input_dim = self.image_pixel_count
        else:
            input_dim = len(self.latent_indices)
        
        self.key_mapper = KeyMapper(
            input_dim=input_dim,
            output_dim=self.config.model.key_length,
            seed=self.config.model.key_mapper_seed
        ).to(self.device)
        self.key_mapper.eval()
        
        # Load checkpoint (excluding key mapper)
        if self.rank == 0:
            logging.info(f"Loading checkpoint from {self.config.checkpoint_path}...")
        load_checkpoint(
            checkpoint_path=self.config.checkpoint_path,
            watermarked_model=self.watermarked_model,
            decoder=self.decoder,
            key_mapper=None,  # Set to None to skip loading key mapper state
            device=self.device
        )
        
        # Initialize LPIPS loss with the same network as in training
        self.lpips_loss_fn = lpips.LPIPS(net='alex').to(self.device)
        
        # Setup quantized models
        self.quantized_models = {}
        self.quantized_watermarked_models = {}
        
        # Setup int8 quantized versions if needed
        if getattr(self.config.evaluate, 'evaluate_quantization', False):
            # Setup int8 quantized version of the original model
            if self.rank == 0:
                logging.info("Setting up int8 quantized model...")
            self.quantized_models['int8'] = quantize_model_weights(self.gan_model, 'int8')
            self.quantized_models['int8'].eval()
            
            # Setup int8 quantized version of the watermarked model
            if getattr(self.config.evaluate, 'evaluate_quantization_watermarked', False):
                if self.rank == 0:
                    logging.info("Setting up int8 quantized watermarked model...")
                self.quantized_watermarked_models['int8'] = quantize_model_weights(self.watermarked_model, 'int8')
                self.quantized_watermarked_models['int8'].eval()
                
        # Setup int4 quantized versions if needed
        if getattr(self.config.evaluate, 'evaluate_quantization_int4', False):
            # Setup int4 quantized version of the original model
            if self.rank == 0:
                logging.info("Setting up int4 quantized model...")
            self.quantized_models['int4'] = quantize_model_weights(self.gan_model, 'int4')
            self.quantized_models['int4'].eval()
            
            # Setup int4 quantized version of the watermarked model
            if getattr(self.config.evaluate, 'evaluate_quantization_int4_watermarked', False):
                if self.rank == 0:
                    logging.info("Setting up int4 quantized watermarked model...")
                self.quantized_watermarked_models['int4'] = quantize_model_weights(self.watermarked_model, 'int4')
                self.quantized_watermarked_models['int4'].eval()
                
        # Setup int2 quantized versions if needed
        if getattr(self.config.evaluate, 'evaluate_quantization_int2', False):
            # Setup int2 quantized version of the original model
            if self.rank == 0:
                logging.info("Setting up int2 quantized model...")
            self.quantized_models['int2'] = quantize_model_weights(self.gan_model, 'int2')
            self.quantized_models['int2'].eval()
            
            # Setup int2 quantized version of the watermarked model
            if getattr(self.config.evaluate, 'evaluate_quantization_int2_watermarked', False):
                if self.rank == 0:
                    logging.info("Setting up int2 quantized watermarked model...")
                self.quantized_watermarked_models['int2'] = quantize_model_weights(self.watermarked_model, 'int2')
                self.quantized_watermarked_models['int2'].eval()
        
        # For backward compatibility
        if 'int8' in self.quantized_models:
            self.quantized_model = self.quantized_models['int8']
        else:
            self.quantized_model = None
            
        if 'int8' in self.quantized_watermarked_models:
            self.quantized_watermarked_model = self.quantized_watermarked_models['int8'] 
        else:
            self.quantized_watermarked_model = None
    
    def process_batch(self, z, model_name=None, transformation=None):
        """
        Process a batch of latent vectors for evaluation.
        
        Args:
            z (torch.Tensor): Batch of latent vectors.
            model_name (str, optional): Name of the pretrained model to use. If None, uses the watermarked model.
            transformation (str, optional): Name of the transformation to apply to the original model/images.
        
        Returns:
            dict: Dictionary containing evaluation results.
        """
        with torch.no_grad():
            try:
                # Determine if this is a comparison case (original evaluation) or a negative sample case
                is_comparison_case = model_name is None and transformation is None
                
                # For negative cases, we only need to evaluate the specific model/transformation
                if not is_comparison_case:
                    return self._process_negative_sample_batch(z, model_name, transformation)
                
                # This is the comparison case (original evaluation)
                # Use watermarked model for watermarked images and original model for original images
                watermarked_model = self.watermarked_model
                original_model = self.gan_model
                
                # Generate watermarked images
                w_water = watermarked_model.mapping(z, None)
                x_water = watermarked_model.synthesis(w_water, noise_mode="const")
                
                # Generate original images
                w_orig = original_model.mapping(z, None)
                x_orig = original_model.synthesis(w_orig, noise_mode="const")
                
                # Extract features based on the approach
                if self.use_image_pixels:
                    # Extract pixel values from the watermarked image
                    features = self.extract_image_partial(x_water)
                else:
                    # Extract latent features from w (original approach)
                    if w_water.ndim == 3:
                        w_water_single = w_water[:, 0, :]
                    else:
                        w_water_single = w_water
                    
                    # Convert latent_indices to tensor if it's a numpy array
                    if isinstance(self.latent_indices, np.ndarray):
                        latent_indices = torch.tensor(self.latent_indices, dtype=torch.long, device=self.device)
                    else:
                        latent_indices = self.latent_indices
                        
                    features = w_water_single[:, latent_indices]
                
                # Generate true key
                true_key = self.key_mapper(features)
                
                # Compute key from watermarked image
                pred_key_water_logits = self.decoder(x_water)
                pred_key_water_probs = torch.sigmoid(pred_key_water_logits)
                
                # Compute key from original image
                pred_key_orig_logits = self.decoder(x_orig)
                pred_key_orig_probs = torch.sigmoid(pred_key_orig_logits)
                
                # Calculate MSE distance metrics for watermarked images
                watermarked_mse_distance = torch.mean(torch.pow(pred_key_water_probs - true_key, 2), dim=1)
                
                # Calculate MSE distance metrics for original images
                original_mse_distance = torch.mean(torch.pow(pred_key_orig_probs - true_key, 2), dim=1)
                
                # Calculate LPIPS loss between original and watermarked images
                lpips_losses = self.lpips_loss_fn(x_orig, x_water).squeeze().cpu().numpy()
                
                # Return metrics data
                return {
                    'watermarked_mse_distances': watermarked_mse_distance.cpu().numpy(),
                    'original_mse_distances': original_mse_distance.cpu().numpy(),
                    'batch_size': z.size(0),
                    'lpips_losses': lpips_losses,
                    'x_orig': x_orig,
                    'x_water': x_water
                }
            except Exception as e:
                if self.rank == 0:
                    logging.error(f"Error processing batch with model_name={model_name}, transformation={transformation}: {str(e)}")
                    logging.error(str(e), exc_info=True)
                # Return empty results with same shape as expected
                batch_size = z.size(0)
                return {
                    'watermarked_mse_distances': np.ones(batch_size) * 0.5,
                    'original_mse_distances': np.ones(batch_size) * 0.5,
                    'batch_size': batch_size,
                    'lpips_losses': np.ones(batch_size) * 0.5,
                    'x_orig': torch.zeros_like(z.reshape(z.size(0), 1, 1, 1).expand(-1, 3, self.config.model.img_size, self.config.model.img_size)),
                    'x_water': torch.zeros_like(z.reshape(z.size(0), 1, 1, 1).expand(-1, 3, self.config.model.img_size, self.config.model.img_size))
                }
    
    def _process_negative_sample_batch(self, z, model_name=None, transformation=None):
        """
        Process a batch specifically for negative sample evaluation (pretrained models or transformations).
        This function avoids redundant computation by only evaluating the negative samples themselves.
        
        Args:
            z (torch.Tensor): Batch of latent vectors.
            model_name (str, optional): Name of the pretrained model to use.
            transformation (str, optional): Name of the transformation to apply.
            
        Returns:
            dict: Dictionary containing evaluation results.
        """
        try:
            # Select the negative model to use
            if model_name is not None:
                # Use the pretrained model
                if model_name not in self.pretrained_models:
                    if self.rank == 0:
                        logging.warning(f"Pretrained model '{model_name}' not found. Falling back to original model.")
                    negative_model = self.gan_model
                else:
                    negative_model = self.pretrained_models[model_name]
                    
                # Generate images from the negative model
                w_neg = negative_model.mapping(z, None)
                x_neg = negative_model.synthesis(w_neg, noise_mode="const")
                
            else:  # transformation is not None
                # Determine which base model to use based on the transformation
                is_watermarked = '_watermarked' in transformation
                base_model = self.watermarked_model if is_watermarked else self.gan_model
                transformation_base = transformation.replace('_watermarked', '').replace('_original', '')
                
                # Handle transformations
                if transformation_base == 'truncation':
                    # For truncation, apply truncation to the image generation
                    truncation_psi = getattr(self.config.evaluate, 'truncation_psi', 2.0)
                    x_neg, w_neg = apply_truncation(base_model, z, truncation_psi, return_w=True)
                    
                elif transformation_base in ['quantization', 'quantization_int4', 'quantization_int2']:
                    # Extract the bit precision from the transformation name
                    if transformation_base == 'quantization':
                        precision = 'int8'
                    elif transformation_base == 'quantization_int4':
                        precision = 'int4'
                    else:  # quantization_int2
                        precision = 'int2'
                        
                    # Use the appropriate quantized model
                    if is_watermarked:
                        if precision in self.quantized_watermarked_models:
                            quantized_model = self.quantized_watermarked_models[precision]
                        else:
                            if self.rank == 0:
                                logging.warning(f"Quantized watermarked model with precision {precision} not initialized. Using watermarked model instead.")
                            quantized_model = base_model
                    else:
                        if precision in self.quantized_models:
                            quantized_model = self.quantized_models[precision]
                        else:
                            if self.rank == 0:
                                logging.warning(f"Quantized model with precision {precision} not initialized. Using original model instead.")
                            quantized_model = base_model
                            
                    # Generate images with the quantized model
                    w_neg = quantized_model.mapping(z, None)
                    x_neg = quantized_model.synthesis(w_neg, noise_mode="const")
                        
                else:
                    # First generate image with base model
                    w_neg = base_model.mapping(z, None)
                    x_neg = base_model.synthesis(w_neg, noise_mode="const")
                    
                    # Then apply post-processing transformations
                    if transformation_base == 'downsample':
                        downsample_size = getattr(self.config.evaluate, 'downsample_size', 128)
                        x_neg = downsample_and_upsample(x_neg, downsample_size)
                    elif transformation_base == 'jpeg':
                        jpeg_quality = getattr(self.config.evaluate, 'jpeg_quality', 55)
                        x_neg = apply_jpeg_compression(x_neg, jpeg_quality)
            
            # Extract features based on the approach
            if self.use_image_pixels:
                # Extract pixel values from the image
                features = self.extract_image_partial(x_neg)
            else:
                # Extract latent features from w
                if w_neg.ndim == 3:
                    w_neg_single = w_neg[:, 0, :]
                else:
                    w_neg_single = w_neg
                    
                # Convert latent_indices to tensor if it's a numpy array
                if isinstance(self.latent_indices, np.ndarray):
                    latent_indices = torch.tensor(self.latent_indices, dtype=torch.long, device=self.device)
                else:
                    latent_indices = self.latent_indices
                    
                features = w_neg_single[:, latent_indices]
            
            # Generate true key
            true_key = self.key_mapper(features)
            
            # Compute key from negative sample image
            pred_key_neg_logits = self.decoder(x_neg)
            pred_key_neg_probs = torch.sigmoid(pred_key_neg_logits)
            
            # Calculate MSE distance metrics
            neg_mse_distance = torch.mean(torch.pow(pred_key_neg_probs - true_key, 2), dim=1)
            
            # Use dummy values for watermarked measurements to maintain API compatibility
            batch_size = z.size(0)
            dummy_distances = np.ones(batch_size) * 0.5
            
            # Create a dummy tensor of the same shape as x_neg for the watermarked image placeholder
            dummy_image = torch.zeros_like(x_neg)
            
            # Return metrics data in the same format as the original function for compatibility
            return {
                'watermarked_mse_distances': dummy_distances,  # Dummy value
                'original_mse_distances': neg_mse_distance.cpu().numpy(),  # Actual negative sample distances
                'batch_size': batch_size,
                'lpips_losses': np.zeros(batch_size),  # Zero LPIPS since we're not comparing
                'x_orig': x_neg,  # Store the negative sample image as the "original"
                'x_water': dummy_image  # Dummy image for compatibility
            }
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error processing negative batch with model_name={model_name}, transformation={transformation}: {str(e)}")
                logging.error(str(e), exc_info=True)
            # Return empty results with same shape as expected
            batch_size = z.size(0)
            return {
                'watermarked_mse_distances': np.ones(batch_size) * 0.5,
                'original_mse_distances': np.ones(batch_size) * 0.5,
                'batch_size': batch_size,
                'lpips_losses': np.zeros(batch_size),
                'x_orig': torch.zeros_like(z.reshape(z.size(0), 1, 1, 1).expand(-1, 3, self.config.model.img_size, self.config.model.img_size)),
                'x_water': torch.zeros_like(z.reshape(z.size(0), 1, 1, 1).expand(-1, 3, self.config.model.img_size, self.config.model.img_size))
            }
    
    def evaluate_batch(self) -> Dict[str, float]:
        """
        Perform batch evaluation on multiple samples.
        
        Returns:
            dict: Evaluation metrics.
        """
        if self.rank == 0:
            logging.info(f"Running batch evaluation with {self.config.evaluate.num_samples} samples...")
        
        # Initialize metrics collection
        results = {}
        
        # First, evaluate original and watermarked models
        results['original'] = self.evaluate_model_batch(None, None)
        
        # Calculate the threshold at 95% TPR using watermarked samples
        threshold = self.calculate_threshold_at_tpr(
            results['original']['all_watermarked_mse_distances'], 
            target_tpr=0.95
        )
        
        if self.rank == 0:
            logging.info(f"Calculated threshold at 95% TPR: {threshold:.6f}")
            
        # Save threshold for use in all evaluations
        results['original']['threshold_95tpr'] = threshold
        
        # Calculate FPR for original model (negative samples)
        original_fpr = self.calculate_fpr_at_threshold(
            results['original']['all_original_mse_distances'], 
            threshold
        )
        results['original']['original_fpr_at_95tpr'] = original_fpr
        
        # Evaluate negative samples if needed
        if hasattr(self.config.evaluate, 'evaluate_neg_samples') and self.config.evaluate.evaluate_neg_samples:
            # Evaluate pretrained models
            if hasattr(self.config.evaluate, 'evaluate_pretrained') and self.config.evaluate.evaluate_pretrained:
                for model_name in self.pretrained_models:
                    results[f'pretrained_{model_name}'] = self.evaluate_model_batch(model_name, None)
                    # Calculate FPR at the same threshold
                    fpr = self.calculate_fpr_at_threshold(
                        results[f'pretrained_{model_name}']['all_original_mse_distances'], 
                        threshold
                    )
                    results[f'pretrained_{model_name}']['original_fpr_at_95tpr'] = fpr
            
            # Evaluate transformations
            if hasattr(self.config.evaluate, 'evaluate_transforms') and self.config.evaluate.evaluate_transforms:
                # Evaluate truncation
                if hasattr(self.config.evaluate, 'evaluate_truncation') and self.config.evaluate.evaluate_truncation:
                    results['truncation_original'] = self.evaluate_model_batch(None, 'truncation_original')
                    fpr = self.calculate_fpr_at_threshold(
                        results['truncation_original']['all_original_mse_distances'], 
                        threshold
                    )
                    results['truncation_original']['original_fpr_at_95tpr'] = fpr
                    
                    if hasattr(self.config.evaluate, 'evaluate_truncation_watermarked') and self.config.evaluate.evaluate_truncation_watermarked:
                        results['truncation_watermarked'] = self.evaluate_model_batch(None, 'truncation_watermarked')
                        fpr = self.calculate_fpr_at_threshold(
                            results['truncation_watermarked']['all_original_mse_distances'], 
                            threshold
                        )
                        results['truncation_watermarked']['original_fpr_at_95tpr'] = fpr
                
                # Evaluate quantization (int8)
                if hasattr(self.config.evaluate, 'evaluate_quantization') and self.config.evaluate.evaluate_quantization:
                    results['quantization_original'] = self.evaluate_model_batch(None, 'quantization_original')
                    fpr = self.calculate_fpr_at_threshold(
                        results['quantization_original']['all_original_mse_distances'], 
                        threshold
                    )
                    results['quantization_original']['original_fpr_at_95tpr'] = fpr
                    
                    if hasattr(self.config.evaluate, 'evaluate_quantization_watermarked') and self.config.evaluate.evaluate_quantization_watermarked:
                        results['quantization_watermarked'] = self.evaluate_model_batch(None, 'quantization_watermarked')
                        fpr = self.calculate_fpr_at_threshold(
                            results['quantization_watermarked']['all_original_mse_distances'], 
                            threshold
                        )
                        results['quantization_watermarked']['original_fpr_at_95tpr'] = fpr
                
                # Evaluate quantization (int4)
                if hasattr(self.config.evaluate, 'evaluate_quantization_int4') and self.config.evaluate.evaluate_quantization_int4:
                    results['quantization_int4_original'] = self.evaluate_model_batch(None, 'quantization_int4_original')
                    fpr = self.calculate_fpr_at_threshold(
                        results['quantization_int4_original']['all_original_mse_distances'], 
                        threshold
                    )
                    results['quantization_int4_original']['original_fpr_at_95tpr'] = fpr
                    
                    if hasattr(self.config.evaluate, 'evaluate_quantization_int4_watermarked') and self.config.evaluate.evaluate_quantization_int4_watermarked:
                        results['quantization_int4_watermarked'] = self.evaluate_model_batch(None, 'quantization_int4_watermarked')
                        fpr = self.calculate_fpr_at_threshold(
                            results['quantization_int4_watermarked']['all_original_mse_distances'], 
                            threshold
                        )
                        results['quantization_int4_watermarked']['original_fpr_at_95tpr'] = fpr
                        
                # Evaluate quantization (int2)
                if hasattr(self.config.evaluate, 'evaluate_quantization_int2') and self.config.evaluate.evaluate_quantization_int2:
                    results['quantization_int2_original'] = self.evaluate_model_batch(None, 'quantization_int2_original')
                    fpr = self.calculate_fpr_at_threshold(
                        results['quantization_int2_original']['all_original_mse_distances'], 
                        threshold
                    )
                    results['quantization_int2_original']['original_fpr_at_95tpr'] = fpr
                    
                    if hasattr(self.config.evaluate, 'evaluate_quantization_int2_watermarked') and self.config.evaluate.evaluate_quantization_int2_watermarked:
                        results['quantization_int2_watermarked'] = self.evaluate_model_batch(None, 'quantization_int2_watermarked')
                        fpr = self.calculate_fpr_at_threshold(
                            results['quantization_int2_watermarked']['all_original_mse_distances'], 
                            threshold
                        )
                        results['quantization_int2_watermarked']['original_fpr_at_95tpr'] = fpr
                
                # Evaluate downsample/upsample
                if hasattr(self.config.evaluate, 'evaluate_downsample') and self.config.evaluate.evaluate_downsample:
                    results['downsample_original'] = self.evaluate_model_batch(None, 'downsample_original')
                    fpr = self.calculate_fpr_at_threshold(
                        results['downsample_original']['all_original_mse_distances'], 
                        threshold
                    )
                    results['downsample_original']['original_fpr_at_95tpr'] = fpr
                    
                    if hasattr(self.config.evaluate, 'evaluate_downsample_watermarked') and self.config.evaluate.evaluate_downsample_watermarked:
                        results['downsample_watermarked'] = self.evaluate_model_batch(None, 'downsample_watermarked')
                        fpr = self.calculate_fpr_at_threshold(
                            results['downsample_watermarked']['all_original_mse_distances'], 
                            threshold
                        )
                        results['downsample_watermarked']['original_fpr_at_95tpr'] = fpr
                
                # Evaluate JPEG compression
                if hasattr(self.config.evaluate, 'evaluate_jpeg') and self.config.evaluate.evaluate_jpeg:
                    results['jpeg_original'] = self.evaluate_model_batch(None, 'jpeg_original')
                    fpr = self.calculate_fpr_at_threshold(
                        results['jpeg_original']['all_original_mse_distances'], 
                        threshold
                    )
                    results['jpeg_original']['original_fpr_at_95tpr'] = fpr
                    
                    if hasattr(self.config.evaluate, 'evaluate_jpeg_watermarked') and self.config.evaluate.evaluate_jpeg_watermarked:
                        results['jpeg_watermarked'] = self.evaluate_model_batch(None, 'jpeg_watermarked')
                        fpr = self.calculate_fpr_at_threshold(
                            results['jpeg_watermarked']['all_original_mse_distances'], 
                            threshold
                        )
                        results['jpeg_watermarked']['original_fpr_at_95tpr'] = fpr
        
        # Generate combined metrics summary
        if self.rank == 0:
            logging.info("=== Evaluation Summary ===")
            logging.info(f"Threshold at 95% TPR: {threshold:.6f}")
            
            # Create a table format for results
            header = ["Model/Transform", "FPR at 95% TPR", "ROC-AUC (MSE)", "LPIPS Avg", "LPIPS Std"]
            table_data = []
            
            # Add original model results (first row)
            table_data.append([
                "Watermarked (Positive)", 
                f"95.00%", 
                f"{results['original']['roc_auc_score_mse']:.4f}",
                f"{results['original']['watermarked_lpips_loss_avg']:.4f}",
                f"{results['original']['watermarked_lpips_loss_std']:.4f}"
            ])
            
            # Add original model non-watermarked results (second row)
            table_data.append([
                "Original (Negative)", 
                f"{results['original']['original_fpr_at_95tpr']:.2f}%",
                "-",
                "-",
                "-"
            ])
            
            # Add other negative samples
            for result_name, metrics in results.items():
                if result_name == 'original':
                    continue  # Already added above
                
                table_data.append([
                    result_name,
                    f"{metrics['original_fpr_at_95tpr']:.2f}%",
                    "-",
                    "-",
                    "-"
                ])
            
            # Print table
            logging.info("Results Table:")
            col_widths = [max(len(row[i]) for row in [header] + table_data) for i in range(len(header))]
            
            # Print header
            header_str = " | ".join(f"{h:{w}s}" for h, w in zip(header, col_widths))
            logging.info(f"| {header_str} |")
            
            # Print separator
            sep = " | ".join("-" * w for w in col_widths)
            logging.info(f"| {sep} |")
            
            # Print data rows
            for row in table_data:
                row_str = " | ".join(f"{cell:{w}s}" for cell, w in zip(row, col_widths))
                logging.info(f"| {row_str} |")
            
            # Save all metrics to files
            for result_name, metrics in results.items():
                output_dir = os.path.join(self.config.output_dir, result_name)
                os.makedirs(output_dir, exist_ok=True)
                save_metrics_text(metrics, output_dir)
                
                # Save plots only for original evaluation
                if result_name == 'original':
                    # Get ROC data from metrics or create a fallback
                    if 'roc_data' in metrics:
                        y_data = metrics['roc_data']
                    else:
                        y_data = self._create_fallback_roc_data(metrics)
                    
                    save_metrics_plots(
                        metrics,
                        y_data,
                        metrics['all_watermarked_mse_distances'],
                        metrics['all_original_mse_distances'],
                        None,  # No MAE for watermarked
                        None,  # No MAE for original
                        output_dir
                    )
        
        return results['original']  # Return original results for backward compatibility
    
    def evaluate_model_batch(self, model_name=None, transformation=None):
        """
        Evaluate a specific model or transformation on batch of samples.
        
        Args:
            model_name (str, optional): Name of the pretrained model to evaluate.
            transformation (str, optional): Name of the transformation to apply.
            
        Returns:
            dict: Evaluation metrics.
        """
        # Create appropriate description for what's being evaluated
        if model_name is None and transformation is None:
            name_str = "watermarked vs. original models"
        elif model_name is not None:
            name_str = f"pretrained model '{model_name}'"
        elif transformation is not None:
            name_str = f"{transformation} transformation"
            
        if self.rank == 0:
            logging.info(f"Evaluating {name_str}...")
        
        try:
            # Initialize metrics collection
            all_watermarked_mse_distances = []
            all_original_mse_distances = []
            
            num_batches = (self.config.evaluate.num_samples + self.config.evaluate.batch_size - 1) // self.config.evaluate.batch_size
            total_samples = 0
            all_lpips_losses = []
            
            # Only use tqdm progress bar on rank 0
            batch_iterator = tqdm(range(num_batches), desc=f"Evaluating {name_str}") if self.rank == 0 else range(num_batches)
            
            for batch_idx in batch_iterator:
                try:
                    current_batch_size = min(self.config.evaluate.batch_size, self.config.evaluate.num_samples - total_samples)
                    if current_batch_size <= 0:
                        break
                        
                    # Sample latent vectors
                    z = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                    
                    # Process batch
                    batch_results = self.process_batch(z, model_name, transformation)
                    
                    # Accumulate metrics (only MSE, no MAE)
                    all_watermarked_mse_distances.extend(batch_results['watermarked_mse_distances'])
                    all_original_mse_distances.extend(batch_results['original_mse_distances'])
                    total_samples += batch_results['batch_size']
                    all_lpips_losses.extend(batch_results['lpips_losses'])
                    
                except Exception as e:
                    if self.rank == 0:
                        logging.error(f"Error processing batch {batch_idx} for {name_str}: {str(e)}")
                    # Continue with next batch
                    continue
            
            # Check if we have enough data to calculate metrics
            if total_samples == 0:
                if self.rank == 0:
                    logging.error(f"No valid samples were processed for {name_str}. Returning empty metrics.")
                # Return empty metrics
                return self._create_empty_metrics()
            
            # Calculate ROC data (only for MSE, no MAE)
            try:
                # Prepare data for ROC curve calculation
                y_true = np.concatenate([
                    np.ones(len(all_watermarked_mse_distances)), 
                    np.zeros(len(all_original_mse_distances))
                ])
                
                # Negate distances since lower is better for watermarked images
                y_score_mse = np.concatenate([
                    -np.array(all_watermarked_mse_distances), 
                    -np.array(all_original_mse_distances)
                ])
                
                # Calculate ROC-AUC for MSE
                roc_auc_mse = skmetrics.roc_auc_score(y_true, y_score_mse)
                roc_auc_score = roc_auc_mse  # Use MSE as the only metric
            except Exception as e:
                if self.rank == 0:
                    logging.error(f"Error calculating ROC metrics for {name_str}: {str(e)}")
                # Use fallback values
                roc_auc_mse = 0.5
                roc_auc_score = 0.5
                y_true = np.array([1, 0])
                y_score_mse = np.array([-0.4, -0.6])
            
            # Calculate MSE distance statistics
            watermarked_mse_distance_avg = np.mean(all_watermarked_mse_distances)
            watermarked_mse_distance_std = np.std(all_watermarked_mse_distances)
            
            original_mse_distance_avg = np.mean(all_original_mse_distances)
            original_mse_distance_std = np.std(all_original_mse_distances)
            
            # Calculate LPIPS statistics
            lpips_loss_avg = np.mean(all_lpips_losses)
            lpips_loss_std = np.std(all_lpips_losses)
            
            # Create metrics dictionary
            metrics = {
                'watermarked_lpips_loss_avg': lpips_loss_avg,
                'watermarked_lpips_loss_std': lpips_loss_std,
                'watermarked_mse_distance_avg': watermarked_mse_distance_avg,
                'watermarked_mse_distance_std': watermarked_mse_distance_std,
                'original_mse_distance_avg': original_mse_distance_avg,
                'original_mse_distance_std': original_mse_distance_std,
                'roc_auc_score_mse': roc_auc_mse,
                'roc_auc_score': roc_auc_score,
                'num_samples_processed': total_samples,
                # Store raw data for plotting
                'all_watermarked_mse_distances': all_watermarked_mse_distances,
                'all_original_mse_distances': all_original_mse_distances,
                # Store ROC data
                'roc_data': (y_true, y_score_mse)
            }
            
            return metrics
            
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error evaluating {name_str}: {str(e)}")
            # Return empty metrics
            return self._create_empty_metrics()
    
    def _create_empty_metrics(self):
        """Create empty metrics for error cases."""
        # Create minimal fallback ROC data
        y_true = np.array([1, 0])
        y_score_mse = np.array([-0.4, -0.6])
        
        return {
            'watermarked_mse_distance_avg': 0.5,
            'watermarked_mse_distance_std': 0.0,
            'original_mse_distance_avg': 0.5,
            'original_mse_distance_std': 0.0,
            'roc_auc_score_mse': 0.5,
            'roc_auc_score': 0.5,
            'watermarked_lpips_loss_avg': 0.5,
            'watermarked_lpips_loss_std': 0.0,
            'all_watermarked_mse_distances': [0.5],
            'all_original_mse_distances': [0.5],
            'num_samples_processed': 1,
            'roc_data': (y_true, y_score_mse)
        }
    
    def _create_fallback_roc_data(self, metrics):
        """Create fallback ROC data when metrics might be incomplete."""
        try:
            # Try to create ROC data from metrics
            y_true = np.concatenate([
                np.ones(len(metrics['all_watermarked_mse_distances'])), 
                np.zeros(len(metrics['all_original_mse_distances']))
            ])
            
            y_score_mse = np.concatenate([
                -np.array(metrics['all_watermarked_mse_distances']), 
                -np.array(metrics['all_original_mse_distances'])
            ])
            
            return (y_true, y_score_mse)
        except (KeyError, ValueError) as e:
            if self.rank == 0:
                logging.warning(f"Error creating ROC data: {str(e)}. Using fallback data.")
            # Return fallback data
            return (np.array([1, 0]), np.array([-0.4, -0.6]))
    
    def visualize_samples(self):
        """
        Generate and visualize samples for qualitative evaluation.
        Only perform on rank 0.
        """
        if self.rank != 0:
            return
            
        logging.info(f"Running visual evaluation with {self.config.evaluate.num_vis_samples} samples...")
        
        # First visualize original vs. watermarked
        self.visualize_model_samples(None, None, "watermarked")
        
        # Visualize negative samples if needed
        if hasattr(self.config.evaluate, 'evaluate_neg_samples') and self.config.evaluate.evaluate_neg_samples:
            # Visualize pretrained models
            if hasattr(self.config.evaluate, 'evaluate_pretrained') and self.config.evaluate.evaluate_pretrained:
                for model_name in self.pretrained_models:
                    self.visualize_model_samples(model_name, None, f"pretrained_{model_name}")
            
            # Visualize transformations
            if hasattr(self.config.evaluate, 'evaluate_transforms') and self.config.evaluate.evaluate_transforms:
                # Visualize truncation
                if hasattr(self.config.evaluate, 'evaluate_truncation') and self.config.evaluate.evaluate_truncation:
                    self.visualize_model_samples(None, 'truncation_original', "truncation_original")
                    if hasattr(self.config.evaluate, 'evaluate_truncation_watermarked') and self.config.evaluate.evaluate_truncation_watermarked:
                        self.visualize_model_samples(None, 'truncation_watermarked', "truncation_watermarked")
                
                # Visualize quantization (int8)
                if hasattr(self.config.evaluate, 'evaluate_quantization') and self.config.evaluate.evaluate_quantization:
                    self.visualize_model_samples(None, 'quantization_original', "quantization_original")
                    if hasattr(self.config.evaluate, 'evaluate_quantization_watermarked') and self.config.evaluate.evaluate_quantization_watermarked:
                        self.visualize_model_samples(None, 'quantization_watermarked', "quantization_watermarked")
                
                # Visualize quantization (int4)
                if hasattr(self.config.evaluate, 'evaluate_quantization_int4') and self.config.evaluate.evaluate_quantization_int4:
                    self.visualize_model_samples(None, 'quantization_int4_original', "quantization_int4_original")
                    if hasattr(self.config.evaluate, 'evaluate_quantization_int4_watermarked') and self.config.evaluate.evaluate_quantization_int4_watermarked:
                        self.visualize_model_samples(None, 'quantization_int4_watermarked', "quantization_int4_watermarked")
                        
                # Visualize quantization (int2)
                if hasattr(self.config.evaluate, 'evaluate_quantization_int2') and self.config.evaluate.evaluate_quantization_int2:
                    self.visualize_model_samples(None, 'quantization_int2_original', "quantization_int2_original")
                    if hasattr(self.config.evaluate, 'evaluate_quantization_int2_watermarked') and self.config.evaluate.evaluate_quantization_int2_watermarked:
                        self.visualize_model_samples(None, 'quantization_int2_watermarked', "quantization_int2_watermarked")
                
                # Visualize downsample/upsample
                if hasattr(self.config.evaluate, 'evaluate_downsample') and self.config.evaluate.evaluate_downsample:
                    self.visualize_model_samples(None, 'downsample_original', "downsample_original")
                    if hasattr(self.config.evaluate, 'evaluate_downsample_watermarked') and self.config.evaluate.evaluate_downsample_watermarked:
                        self.visualize_model_samples(None, 'downsample_watermarked', "downsample_watermarked")
                
                # Visualize JPEG compression
                if hasattr(self.config.evaluate, 'evaluate_jpeg') and self.config.evaluate.evaluate_jpeg:
                    self.visualize_model_samples(None, 'jpeg_original', "jpeg_original")
                    if hasattr(self.config.evaluate, 'evaluate_jpeg_watermarked') and self.config.evaluate.evaluate_jpeg_watermarked:
                        self.visualize_model_samples(None, 'jpeg_watermarked', "jpeg_watermarked")
    
    def visualize_model_samples(self, model_name=None, transformation=None, output_subdir="watermarked"):
        """
        Generate and visualize samples for a specific model or transformation.
        
        Args:
            model_name (str, optional): Name of the pretrained model to visualize.
            transformation (str, optional): Name of the transformation to apply.
            output_subdir (str): Subdirectory to save visualization results.
        """
        # Create appropriate description for what's being visualized
        if model_name is None and transformation is None:
            name_str = "watermarked vs. original models"
        elif model_name is not None:
            name_str = f"pretrained model '{model_name}'"
        elif transformation is not None:
            name_str = f"{transformation} transformation"
            
        logging.info(f"Visualizing {name_str}...")
        
        with torch.no_grad():
            # Sample latent vectors
            z_vis = torch.randn(self.config.evaluate.num_vis_samples, self.latent_dim, device=self.device)
            
            # Process batch for visualization
            batch_results = self.process_batch(z_vis, model_name, transformation)
            
            # Extract results
            x_orig_vis = batch_results['x_orig']
            x_water_vis = batch_results['x_water']
            
            # Compute difference
            diff_vis = x_water_vis - x_orig_vis
            
            # Create output directory
            output_dir = os.path.join(self.config.output_dir, output_subdir)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save visualizations
            save_visualization(x_orig_vis, x_water_vis, diff_vis, output_dir)
            
            # Generate images for w extraction
            if model_name is None:
                # For watermarked model, extract w from watermarked model
                w_water_vis = self.watermarked_model.mapping(z_vis, None)
            else:
                # For pretrained models, extract w from the pretrained model
                w_water_vis = self.pretrained_models[model_name].mapping(z_vis, None)
            
            # Extract features based on approach
            if self.use_image_pixels:
                # Extract features from the image (watermarked or original)
                if model_name is None and transformation is None:
                    # Use watermarked image for comparison case
                    features = self.extract_image_partial(x_water_vis)
                else:
                    # Use original/negative image for others
                    features = self.extract_image_partial(x_orig_vis)
            else:
                # Extract from w vectors
                if w_water_vis.ndim == 3:
                    w_water_single_vis = w_water_vis[:, 0, :]
                else:
                    w_water_single_vis = w_water_vis
                    
                # Convert latent_indices to tensor if it's a numpy array
                if isinstance(self.latent_indices, np.ndarray):
                    latent_indices = torch.tensor(self.latent_indices, dtype=torch.long, device=self.device)
                else:
                    latent_indices = self.latent_indices
                    
                features = w_water_single_vis[:, latent_indices]
            
            # Generate true key
            true_keys_vis = self.key_mapper(features)
            
            # Predict keys and probabilities for both original and watermarked images
            pred_keys_water_logits_vis = self.decoder(x_water_vis)
            pred_keys_water_probs_vis = torch.sigmoid(pred_keys_water_logits_vis)
            pred_keys_water_vis = pred_keys_water_probs_vis > 0.5
            
            pred_keys_orig_logits_vis = self.decoder(x_orig_vis)
            pred_keys_orig_probs_vis = torch.sigmoid(pred_keys_orig_logits_vis)
            pred_keys_orig_vis = pred_keys_orig_probs_vis > 0.5
            
            # Calculate MSE distance metrics for each sample
            watermarked_mse_distances_vis = torch.mean(torch.pow(pred_keys_water_probs_vis - true_keys_vis, 2), dim=1)
            original_mse_distances_vis = torch.mean(torch.pow(pred_keys_orig_probs_vis - true_keys_vis, 2), dim=1)
            
            # Save key comparison to log
            for i in range(self.config.evaluate.num_vis_samples):
                logging.info(f"Sample {i} ({output_subdir}):")
                logging.info(f"  True key: {true_keys_vis[i].cpu().numpy().astype(int)}")
                logging.info(f"  Watermarked pred: {pred_keys_water_vis[i].cpu().numpy().astype(int)}")
                logging.info(f"  Original pred: {pred_keys_orig_vis[i].cpu().numpy().astype(int)}")
                logging.info(f"  Watermarked match: {(pred_keys_water_vis[i] == true_keys_vis[i]).all().item()}")
                logging.info(f"  Original match: {(pred_keys_orig_vis[i] == true_keys_vis[i]).all().item()}")
                logging.info(f"  Watermarked MSE distance: {watermarked_mse_distances_vis[i].item():.6f}")
                logging.info(f"  Original MSE distance: {original_mse_distances_vis[i].item():.6f}")
    
    def calculate_threshold_at_tpr(self, watermarked_distances, target_tpr=0.95):
        """
        Calculate the threshold where target_tpr (e.g. 95%) of watermarked images have MSE distances below it.
        
        Args:
            watermarked_distances (np.ndarray): Array of MSE distances for watermarked images
            target_tpr (float): Target true positive rate (default: 0.95)
            
        Returns:
            float: Threshold value
        """
        # Sort distances in ascending order
        sorted_distances = np.sort(watermarked_distances)
        
        # Find the index corresponding to the target TPR
        idx = int(np.ceil(len(sorted_distances) * target_tpr)) - 1
        
        # Ensure index is valid
        idx = max(0, min(idx, len(sorted_distances) - 1))
        
        # Return the threshold
        return sorted_distances[idx]
    
    def calculate_fpr_at_threshold(self, negative_distances, threshold):
        """
        Calculate the false positive rate (FPR) for negative samples at the given threshold.
        
        Args:
            negative_distances (np.ndarray): Array of MSE distances for negative images
            threshold (float): Threshold value
            
        Returns:
            float: False positive rate (percentage of negative samples below threshold)
        """
        # Count how many negative samples are below the threshold
        below_threshold = np.sum(negative_distances <= threshold)
        
        # Calculate FPR
        fpr = (below_threshold / len(negative_distances)) * 100
        
        return fpr
    
    def evaluate(self, evaluation_mode='both'):
        """
        Run evaluation based on specified mode.
        
        Args:
            evaluation_mode (str): Evaluation mode 'batch', 'visual', or 'both'.
            
        Returns:
            dict: Evaluation metrics.
        """
        metrics = {}
        
        if evaluation_mode in ['batch', 'both']:
            metrics = self.evaluate_batch()
            
        if evaluation_mode in ['visual', 'both']:
            self.visualize_samples()
            
        return metrics 