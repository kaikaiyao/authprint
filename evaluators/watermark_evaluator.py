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
from models.decoder import Decoder, FeatureDecoder
from models.key_mapper import KeyMapper
from models.model_utils import clone_model, load_stylegan2_model
from utils.checkpoint import load_checkpoint
from utils.metrics import calculate_metrics, save_metrics_plots, save_metrics_text
from utils.visualization import save_visualization, save_comparison_visualization
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
        
        # Track if we're using direct feature decoder mode
        self.direct_feature_decoder = getattr(self.config.model, 'direct_feature_decoder', False)
        if self.direct_feature_decoder and self.rank == 0:
            logging.info(f"Using direct feature decoder mode: decoder takes pixel features directly as input")
        
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
            # Setup the quantized models will happen in setup_models
            pass
    
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
        
        # Generate pixel indices if using image-based approach (needed regardless of decoder type)
        if self.use_image_pixels:
            self._generate_pixel_indices()
        
        # Clone it to create watermarked model
        self.watermarked_model = clone_model(self.gan_model)
        self.watermarked_model.to(self.device)
        self.watermarked_model.eval()
        
        # Initialize appropriate decoder model based on mode
        if self.direct_feature_decoder:
            # Use FeatureDecoder that takes pixel features directly
            if not self.use_image_pixels:
                if self.rank == 0:
                    logging.warning("direct_feature_decoder is enabled but use_image_pixels is False. "
                                   "This configuration is not supported and may cause errors.")
            
            input_dim = self.image_pixel_count  # Number of selected pixels
            if self.rank == 0:
                logging.info(f"Initializing enhanced FeatureDecoder with input_dim={input_dim}, output_dim={self.config.model.key_length}")
            
            self.decoder = FeatureDecoder(
                input_dim=input_dim,
                output_dim=self.config.model.key_length,
                hidden_dims=self.config.decoder.hidden_dims,
                activation=self.config.decoder.activation,
                dropout_rate=self.config.decoder.dropout_rate,
                num_residual_blocks=self.config.decoder.num_residual_blocks,
                use_spectral_norm=self.config.decoder.use_spectral_norm,
                use_layer_norm=self.config.decoder.use_layer_norm,
                use_attention=self.config.decoder.use_attention
            ).to(self.device)
        else:
            # Use standard image-based Decoder
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
            input_dim = self.image_pixel_count
        else:
            input_dim = len(self.latent_indices)
        
        self.key_mapper = KeyMapper(
            input_dim=input_dim,
            output_dim=self.config.model.key_length,
            seed=self.config.model.key_mapper_seed,
            use_sine=getattr(self.config.model, 'key_mapper_use_sine', False),
            sensitivity=getattr(self.config.model, 'key_mapper_sensitivity', 10.0)
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
            model_name (str, optional): Name of the pretrained model to use.
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
                
                # Calculate difference images
                diff_images = x_water - x_orig
                
                # Save comparison visualization if in visualization mode
                if getattr(self.config.evaluate, 'save_comparisons', True):
                    save_comparison_visualization(
                        orig_images=x_orig,
                        watermarked_images=x_water,
                        diff_images=diff_images,
                        output_dir=os.path.join(self.config.output_dir, "comparisons"),
                        prefix=f"batch_{self.batch_counter}" if hasattr(self, 'batch_counter') else "batch"
                    )
                    if hasattr(self, 'batch_counter'):
                        self.batch_counter += 1
                
                # Extract features based on the approach
                if self.use_image_pixels:
                    # Extract pixel values from the watermarked image
                    features_water = self.extract_image_partial(x_water)
                    # Also extract features from original images for comparison
                    features_orig = self.extract_image_partial(x_orig)
                else:
                    # Extract latent features from w (original approach)
                    if w_water.ndim == 3:
                        w_water_single = w_water[:, 0, :]
                        w_orig_single = w_orig[:, 0, :]
                    else:
                        w_water_single = w_water
                        w_orig_single = w_orig
                    
                    # Convert latent_indices to tensor if it's a numpy array
                    if isinstance(self.latent_indices, np.ndarray):
                        latent_indices = torch.tensor(self.latent_indices, dtype=torch.long, device=self.device)
                    else:
                        latent_indices = self.latent_indices
                        
                    features_water = w_water_single[:, latent_indices]
                    features_orig = w_orig_single[:, latent_indices]
                
                # Generate true key using features from watermarked images
                true_key = self.key_mapper(features_water)
                
                # Compute key based on decoder mode
                if self.direct_feature_decoder:
                    # Use features directly as input to the decoder
                    pred_key_water_logits = self.decoder(features_water)
                    pred_key_orig_logits = self.decoder(features_orig)
                else:
                    # Use the full image as input
                    pred_key_water_logits = self.decoder(x_water)
                    pred_key_orig_logits = self.decoder(x_orig)
                
                # Convert to probabilities
                pred_key_water_probs = torch.sigmoid(pred_key_water_logits)
                pred_key_orig_probs = torch.sigmoid(pred_key_orig_logits)
                
                # Calculate MSE distance metrics for watermarked images
                watermarked_mse_distance = torch.mean(torch.pow(pred_key_water_probs - true_key, 2), dim=1)
                
                # Calculate MSE distance metrics for original images
                original_mse_distance = torch.mean(torch.pow(pred_key_orig_probs - true_key, 2), dim=1)
                
                # Calculate LPIPS loss between original and watermarked images
                lpips_losses = self.lpips_loss_fn(x_orig, x_water).squeeze().detach().cpu().numpy()
                
                # Return metrics data
                return {
                    'watermarked_mse_distances': watermarked_mse_distance.detach().cpu().numpy(),
                    'original_mse_distances': original_mse_distance.detach().cpu().numpy(),
                    'batch_size': z.size(0),
                    'lpips_losses': lpips_losses,
                    'x_orig': x_orig,
                    'x_water': x_water,
                    'features_water': features_water,
                    'features_orig': features_orig,
                    'diff_images': diff_images
                }
            except Exception as e:
                if self.rank == 0:
                    logging.error(f"Error processing batch with model_name={model_name}, transformation={transformation}: {str(e)}")
                    logging.error(str(e), exc_info=True)
                # Return empty results with same shape as expected
                batch_size = z.size(0)
                empty_features = torch.zeros(batch_size, self.image_pixel_count if self.use_image_pixels else len(self.latent_indices), device=self.device)
                empty_image = torch.zeros_like(z.reshape(z.size(0), 1, 1, 1).expand(-1, 3, self.config.model.img_size, self.config.model.img_size))
                return {
                    'watermarked_mse_distances': np.ones(batch_size) * 0.5,
                    'original_mse_distances': np.ones(batch_size) * 0.5,
                    'batch_size': batch_size,
                    'lpips_losses': np.ones(batch_size) * 0.5,
                    'x_orig': empty_image,
                    'x_water': empty_image,
                    'features_water': empty_features,
                    'features_orig': empty_features,
                    'diff_images': torch.zeros_like(empty_image)
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
                        if precision not in self.quantized_watermarked_models:
                            if self.rank == 0:
                                logging.warning(f"Quantized watermarked model with precision {precision} not found. Falling back to original watermarked model.")
                            w_neg = self.watermarked_model.mapping(z, None)
                            x_neg = self.watermarked_model.synthesis(w_neg, noise_mode="const")
                        else:
                            w_neg = self.quantized_watermarked_models[precision].mapping(z, None)
                            x_neg = self.quantized_watermarked_models[precision].synthesis(w_neg, noise_mode="const")
                    else:
                        if precision not in self.quantized_models:
                            if self.rank == 0:
                                logging.warning(f"Quantized model with precision {precision} not found. Falling back to original model.")
                            w_neg = self.gan_model.mapping(z, None)
                            x_neg = self.gan_model.synthesis(w_neg, noise_mode="const")
                        else:
                            w_neg = self.quantized_models[precision].mapping(z, None)
                            x_neg = self.quantized_models[precision].synthesis(w_neg, noise_mode="const")
                            
                elif transformation_base == 'downsample':
                    # Generate image from the base model first
                    w_neg = base_model.mapping(z, None)
                    x_orig = base_model.synthesis(w_neg, noise_mode="const")
                    
                    # Then apply downsampling and upsampling
                    downsample_size = getattr(self.config.evaluate, 'downsample_size', 128)
                    x_neg = downsample_and_upsample(x_orig, downsample_size)
                    
                elif transformation_base == 'jpeg':
                    # Generate image from the base model first
                    w_neg = base_model.mapping(z, None)
                    x_orig = base_model.synthesis(w_neg, noise_mode="const")
                    
                    # Then apply JPEG compression
                    jpeg_quality = getattr(self.config.evaluate, 'jpeg_quality', 75)
                    x_neg = apply_jpeg_compression(x_orig, jpeg_quality)
                    
                else:
                    # Fallback for unknown transformation - use the base model directly
                    if self.rank == 0:
                        logging.warning(f"Unknown transformation '{transformation_base}'. Using base model directly.")
                    w_neg = base_model.mapping(z, None)
                    x_neg = base_model.synthesis(w_neg, noise_mode="const")
            
            # Extract features from the image if using image-based approach or needed for direct feature decoder
            if self.use_image_pixels:
                features_neg = self.extract_image_partial(x_neg)
            else:
                # Extract latent features
                if w_neg.ndim == 3:
                    w_neg_single = w_neg[:, 0, :]
                else:
                    w_neg_single = w_neg
                
                # Convert latent_indices to tensor if it's a numpy array
                if isinstance(self.latent_indices, np.ndarray):
                    latent_indices = torch.tensor(self.latent_indices, dtype=torch.long, device=self.device)
                else:
                    latent_indices = self.latent_indices
                    
                features_neg = w_neg_single[:, latent_indices]
            
            # Generate the reference key using the watermarked model
            # We need to generate a watermarked image first to get its features
            w_water = self.watermarked_model.mapping(z, None)
            x_water = self.watermarked_model.synthesis(w_water, noise_mode="const")
            
            # Extract features from the watermarked image
            if self.use_image_pixels:
                features_water = self.extract_image_partial(x_water)
            else:
                # Extract latent features
                if w_water.ndim == 3:
                    w_water_single = w_water[:, 0, :]
                else:
                    w_water_single = w_water
                
                features_water = w_water_single[:, latent_indices]
            
            # Generate true key using features from the watermarked model
            true_key = self.key_mapper(features_water)
            
            # Compute the predicted key based on the decoder mode
            if self.direct_feature_decoder:
                # Use features directly as input to the decoder
                pred_key_neg_logits = self.decoder(features_neg)
            else:
                # Use the full image as input
                pred_key_neg_logits = self.decoder(x_neg)
                
            # Convert to probabilities
            pred_key_neg_probs = torch.sigmoid(pred_key_neg_logits)
            
            # Calculate MSE distance metrics
            negative_mse_distance = torch.mean(torch.pow(pred_key_neg_probs - true_key, 2), dim=1)
            
            # Return metrics data
            return {
                'negative_mse_distances': negative_mse_distance.detach().cpu().numpy(),
                'batch_size': z.size(0),
                'x_neg': x_neg,
                'features_neg': features_neg
            }
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error processing negative sample batch with model_name={model_name}, transformation={transformation}: {str(e)}")
                logging.error(str(e), exc_info=True)
            # Return empty results with same shape as expected
            batch_size = z.size(0)
            empty_features = torch.zeros(batch_size, self.image_pixel_count if self.use_image_pixels else len(self.latent_indices), device=self.device)
            # Create a properly sized empty tensor for x_neg
            empty_image = torch.zeros((batch_size, 3, self.config.model.img_size, self.config.model.img_size), device=self.device)
            return {
                'negative_mse_distances': np.ones(batch_size) * 0.5,
                'batch_size': batch_size,
                'x_neg': empty_image,
                'features_neg': empty_features
            }
    
    def evaluate_batch(self) -> Dict[str, float]:
        """
        Run batch evaluation to compute metrics.
        
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        if self.rank == 0:
            logging.info(f"Running batch evaluation with {self.config.evaluate.num_samples} samples...")
        
        # Set seed for reproducibility
        if hasattr(self.config.evaluate, 'seed') and self.config.evaluate.seed is not None:
            np.random.seed(self.config.evaluate.seed)
            torch.manual_seed(self.config.evaluate.seed)
            if self.rank == 0:
                logging.info(f"Using fixed random seed {self.config.evaluate.seed} for evaluation")
        
        # Create empty accumulators
        batch_size = self.config.evaluate.batch_size
        num_samples = self.config.evaluate.num_samples
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
        
        # Overall accumulators
        watermarked_distances_all = []
        original_distances_all = []
        lpips_losses_all = []
        
        # Negative sample accumulators
        negative_distances_all = {}
        
        # Process data in batches
        for i in range(num_batches):
            # Determine batch size for the last batch which might be smaller
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # Generate batch of latent vectors
            z = torch.randn(current_batch_size, self.latent_dim, device=self.device)
            
            # Process batch - comparing original vs. watermarked
            batch_results = self.process_batch(z)
            
            # Accumulate results
            watermarked_distances_all.append(batch_results['watermarked_mse_distances'])
            original_distances_all.append(batch_results['original_mse_distances'])
            lpips_losses_all.append(batch_results['lpips_losses'])
        
        # Process negative samples if enabled
        if getattr(self.config.evaluate, 'evaluate_neg_samples', True):
            # Process negative samples from pretrained models
            if getattr(self.config.evaluate, 'evaluate_pretrained', True):
                if getattr(self.config.evaluate, 'evaluate_ffhq1k', True) and 'ffhq1k' in self.pretrained_models:
                    negative_distances_all['ffhq1k'] = self.evaluate_model_batch(model_name='ffhq1k')
                
                if getattr(self.config.evaluate, 'evaluate_ffhq30k', True) and 'ffhq30k' in self.pretrained_models:
                    negative_distances_all['ffhq30k'] = self.evaluate_model_batch(model_name='ffhq30k')
                
                if getattr(self.config.evaluate, 'evaluate_ffhq70k_bcr', True) and 'ffhq70k-bcr' in self.pretrained_models:
                    negative_distances_all['ffhq70k_bcr'] = self.evaluate_model_batch(model_name='ffhq70k-bcr')
                
                if getattr(self.config.evaluate, 'evaluate_ffhq70k_noaug', True) and 'ffhq70k-noaug' in self.pretrained_models:
                    negative_distances_all['ffhq70k_noaug'] = self.evaluate_model_batch(model_name='ffhq70k-noaug')
                
            # Process negative samples from transformed images
            if getattr(self.config.evaluate, 'evaluate_transforms', True):
                # Truncation transformation
                if getattr(self.config.evaluate, 'evaluate_truncation', True):
                    negative_distances_all['truncation_original'] = self.evaluate_model_batch(transformation='truncation_original')
                    
                if getattr(self.config.evaluate, 'evaluate_truncation_watermarked', True):
                    negative_distances_all['truncation_watermarked'] = self.evaluate_model_batch(transformation='truncation_watermarked')
                
                # Int8 Quantization
                if getattr(self.config.evaluate, 'evaluate_quantization', True):
                    negative_distances_all['quantization_original'] = self.evaluate_model_batch(transformation='quantization_original')
                    
                if getattr(self.config.evaluate, 'evaluate_quantization_watermarked', True):
                    negative_distances_all['quantization_watermarked'] = self.evaluate_model_batch(transformation='quantization_watermarked')
                
                # Int4 Quantization
                if getattr(self.config.evaluate, 'evaluate_quantization_int4', True):
                    negative_distances_all['quantization_int4_original'] = self.evaluate_model_batch(transformation='quantization_int4_original')
                    
                if getattr(self.config.evaluate, 'evaluate_quantization_int4_watermarked', True):
                    negative_distances_all['quantization_int4_watermarked'] = self.evaluate_model_batch(transformation='quantization_int4_watermarked')
                
                # Int2 Quantization
                if getattr(self.config.evaluate, 'evaluate_quantization_int2', True):
                    negative_distances_all['quantization_int2_original'] = self.evaluate_model_batch(transformation='quantization_int2_original')
                    
                if getattr(self.config.evaluate, 'evaluate_quantization_int2_watermarked', True):
                    negative_distances_all['quantization_int2_watermarked'] = self.evaluate_model_batch(transformation='quantization_int2_watermarked')
                
                # Downsampling
                if getattr(self.config.evaluate, 'evaluate_downsample', True):
                    negative_distances_all['downsample_original'] = self.evaluate_model_batch(transformation='downsample_original')
                    
                if getattr(self.config.evaluate, 'evaluate_downsample_watermarked', True):
                    negative_distances_all['downsample_watermarked'] = self.evaluate_model_batch(transformation='downsample_watermarked')
                
                # JPEG compression
                if getattr(self.config.evaluate, 'evaluate_jpeg', True):
                    negative_distances_all['jpeg_original'] = self.evaluate_model_batch(transformation='jpeg_original')
                    
                if getattr(self.config.evaluate, 'evaluate_jpeg_watermarked', True):
                    negative_distances_all['jpeg_watermarked'] = self.evaluate_model_batch(transformation='jpeg_watermarked')
        
        # Concatenate results across batches
        watermarked_distances = np.concatenate(watermarked_distances_all)
        original_distances = np.concatenate(original_distances_all)
        lpips_losses = np.concatenate(lpips_losses_all)
        
        # Calculate number of correct key matches (we consider any match with MSE < 0.25 as correct)
        # We would ideally use binary match rates, but this approximation works for evaluation
        threshold = 0.25  # Typical threshold for correct key detection
        watermarked_correct = np.sum(watermarked_distances < threshold)
        original_correct = np.sum(original_distances < threshold)
        total_samples = len(watermarked_distances)
        
        # For now, use MSE distances as MAE distances since we don't calculate them directly
        # In a future implementation, we could calculate MAE distances separately
        watermarked_mae_distances = watermarked_distances
        original_mae_distances = original_distances
        
        # Calculate metrics
        if len(negative_distances_all) > 0:
            # Include negative samples in metrics calculation
            metrics, roc_data = calculate_metrics(
                watermarked_distances, 
                original_distances,
                watermarked_mae_distances, 
                original_mae_distances,
                watermarked_correct,
                original_correct,
                total_samples,
                lpips_losses
            )
            metrics['negative_distances_all'] = negative_distances_all
        else:
            # Only compare watermarked with original 
            empty_negative = self._create_empty_metrics()
            metrics, roc_data = calculate_metrics(
                watermarked_distances, 
                original_distances,
                watermarked_mae_distances, 
                original_mae_distances,
                watermarked_correct,
                original_correct,
                total_samples,
                lpips_losses
            )
            # Create empty ROC data
            metrics['roc_data'] = roc_data
        
        if self.rank == 0:
            # Save metrics
            save_metrics_text(metrics, self.config.output_dir)
            save_metrics_plots(metrics, roc_data, watermarked_distances, original_distances, 
                              watermarked_mae_distances, original_mae_distances, self.config.output_dir)
        
        return metrics
    
    def evaluate_model_batch(self, model_name=None, transformation=None):
        """
        Evaluate a specific model or transformation.
        
        Args:
            model_name (str, optional): Name of the pretrained model to evaluate.
            transformation (str, optional): Name of the transformation to apply.
            
        Returns:
            np.ndarray: Distances for the negative samples.
        """
        # Initialize accumulators
        batch_size = self.config.evaluate.batch_size
        num_samples = self.config.evaluate.num_samples
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
        
        negative_distances_all = []
        
        # Process data in batches
        for i in range(num_batches):
            # Determine batch size for the last batch which might be smaller
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # Generate batch of latent vectors
            z = torch.randn(current_batch_size, self.latent_dim, device=self.device)
            
            # Process batch using the specified model or transformation
            batch_results = self._process_negative_sample_batch(z, model_name, transformation)
            
            # Accumulate negative distances
            negative_distances_all.append(batch_results['negative_mse_distances'])
        
        # Concatenate results across batches
        negative_distances = np.concatenate(negative_distances_all)
        
        return negative_distances
    
    def _create_empty_metrics(self):
        """Create empty metrics for error cases."""
        # Create minimal fallback ROC data
        y_true = np.array([1, 0])
        y_score_mse = np.array([-0.4, -0.6])
        y_score_mae = np.array([-0.4, -0.6])
        
        empty_dict = {
            'watermarked_match_rate': 50.0,
            'original_match_rate': 0.0,
            'watermarked_mse_distance_avg': 0.5,
            'watermarked_mse_distance_std': 0.0,
            'watermarked_mae_distance_avg': 0.5,
            'watermarked_mae_distance_std': 0.0,
            'original_mse_distance_avg': 0.5,
            'original_mse_distance_std': 0.0,
            'original_mae_distance_avg': 0.5,
            'original_mae_distance_std': 0.0,
            'roc_auc_score_mse': 0.5,
            'roc_auc_score_mae': 0.5,
            'roc_auc_score': 0.5,
            'watermarked_lpips_loss_avg': 0.5,
            'watermarked_lpips_loss_std': 0.0,
            'all_watermarked_mse_distances': [0.5],
            'all_original_mse_distances': [0.5],
            'num_samples_processed': 1
        }
        
        return empty_dict
    
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
            
            # Use same data for MAE scores as a fallback
            y_score_mae = y_score_mse.copy()
            
            return (y_true, y_score_mse, y_score_mae)
        except (KeyError, ValueError) as e:
            if self.rank == 0:
                logging.warning(f"Error creating ROC data: {str(e)}. Using fallback data.")
            # Return fallback data
            return (np.array([1, 0]), np.array([-0.4, -0.6]), np.array([-0.4, -0.6]))
    
    def visualize_samples(self):
        """
        Generate and visualize a set of samples with their watermark keys.
        """
        if self.rank != 0:
            return  # Only perform visualization on the master process
            
        # Create main visualization directory
        vis_dir = os.path.join(self.config.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate fixed latent vectors for visualization
        np.random.seed(self.config.evaluate.visualization_seed)
        num_vis_samples = self.config.evaluate.num_vis_samples
        z_vis = torch.from_numpy(
            np.random.randn(num_vis_samples, self.latent_dim)
        ).float().to(self.device)
        
        # Visualize the watermarked model samples
        vis_subdir = os.path.join(vis_dir, "watermarked")
        self.visualize_model_samples(output_subdir=vis_subdir, z_vis=z_vis)
        
        # Visualize the pretrained model samples if enabled
        if hasattr(self.config.evaluate, 'evaluate_pretrained') and self.config.evaluate.evaluate_pretrained:
            for model_name in self.pretrained_models:
                vis_subdir = os.path.join(vis_dir, f"pretrained_{model_name}")
                self.visualize_model_samples(model_name=model_name, output_subdir=vis_subdir, z_vis=z_vis)
        
        # Visualize transformations if enabled
        if hasattr(self.config.evaluate, 'evaluate_transforms') and self.config.evaluate.evaluate_transforms:
            # Truncation
            if getattr(self.config.evaluate, 'evaluate_truncation', False):
                vis_subdir = os.path.join(vis_dir, "truncation_original")
                self.visualize_model_samples(transformation='truncation_original', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
                
            if getattr(self.config.evaluate, 'evaluate_truncation_watermarked', False):
                vis_subdir = os.path.join(vis_dir, "truncation_watermarked")
                self.visualize_model_samples(transformation='truncation_watermarked', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
            
            # Int8 Quantization
            if getattr(self.config.evaluate, 'evaluate_quantization', False):
                vis_subdir = os.path.join(vis_dir, "quantization_original")
                self.visualize_model_samples(transformation='quantization_original', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
                
            if getattr(self.config.evaluate, 'evaluate_quantization_watermarked', False):
                vis_subdir = os.path.join(vis_dir, "quantization_watermarked")
                self.visualize_model_samples(transformation='quantization_watermarked', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
            
            # Int4 Quantization
            if getattr(self.config.evaluate, 'evaluate_quantization_int4', False):
                vis_subdir = os.path.join(vis_dir, "quantization_int4_original")
                self.visualize_model_samples(transformation='quantization_int4_original', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
                
            if getattr(self.config.evaluate, 'evaluate_quantization_int4_watermarked', False):
                vis_subdir = os.path.join(vis_dir, "quantization_int4_watermarked")
                self.visualize_model_samples(transformation='quantization_int4_watermarked', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
            
            # Int2 Quantization
            if getattr(self.config.evaluate, 'evaluate_quantization_int2', False):
                vis_subdir = os.path.join(vis_dir, "quantization_int2_original")
                self.visualize_model_samples(transformation='quantization_int2_original', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
                
            if getattr(self.config.evaluate, 'evaluate_quantization_int2_watermarked', False):
                vis_subdir = os.path.join(vis_dir, "quantization_int2_watermarked")
                self.visualize_model_samples(transformation='quantization_int2_watermarked', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
            
            # Downsampling
            if getattr(self.config.evaluate, 'evaluate_downsample', False):
                vis_subdir = os.path.join(vis_dir, "downsample_original")
                self.visualize_model_samples(transformation='downsample_original', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
                
            if getattr(self.config.evaluate, 'evaluate_downsample_watermarked', False):
                vis_subdir = os.path.join(vis_dir, "downsample_watermarked")
                self.visualize_model_samples(transformation='downsample_watermarked', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
            
            # JPEG compression
            if getattr(self.config.evaluate, 'evaluate_jpeg', False):
                vis_subdir = os.path.join(vis_dir, "jpeg_original")
                self.visualize_model_samples(transformation='jpeg_original', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
                
            if getattr(self.config.evaluate, 'evaluate_jpeg_watermarked', False):
                vis_subdir = os.path.join(vis_dir, "jpeg_watermarked")
                self.visualize_model_samples(transformation='jpeg_watermarked', 
                                           output_subdir=vis_subdir, z_vis=z_vis)
    
    def visualize_model_samples(self, model_name=None, transformation=None, output_subdir="watermarked", z_vis=None):
        """
        Generate and visualize a set of samples for a specific model or transformation.
        
        Args:
            model_name (str, optional): Name of the pretrained model to use.
            transformation (str, optional): Name of the transformation to apply.
            output_subdir (str): Subdirectory to save visualizations in.
            z_vis (torch.Tensor, optional): Fixed latent vectors for visualization.
        """
        # Create visualization directory
        os.makedirs(output_subdir, exist_ok=True)
        
        # Get number of samples for visualization
        num_vis_samples = self.config.evaluate.num_vis_samples
        
        # Generate latent vectors if not provided
        if z_vis is None:
            np.random.seed(self.config.evaluate.visualization_seed)
            z_vis = torch.from_numpy(
                np.random.randn(num_vis_samples, self.latent_dim)
            ).float().to(self.device)
        
        with torch.no_grad():
            # Determine which model to use for comparison
            if model_name is not None or transformation is not None:
                # Negative sample case
                batch_result = self._process_negative_sample_batch(z_vis, model_name, transformation)
                
                # Extract images and features
                x_neg = batch_result['x_neg']
                features_neg = batch_result['features_neg']
                
                # Convert to numpy for visualization
                x_neg_np = x_neg.detach().cpu().numpy()
                
                # Get the watermarked model images for reference comparison
                w_water = self.watermarked_model.mapping(z_vis, None)
                x_water = self.watermarked_model.synthesis(w_water, noise_mode="const")
                
                # Also extract original images for comparison
                w_orig = self.gan_model.mapping(z_vis, None)
                x_orig = self.gan_model.synthesis(w_orig, noise_mode="const")
                
                # Extract features
                if self.use_image_pixels:
                    features_water = self.extract_image_partial(x_water)
                    features_orig = self.extract_image_partial(x_orig)
                else:
                    # Extract latent features
                    if w_water.ndim == 3:
                        w_water_single = w_water[:, 0, :]
                        w_orig_single = w_orig[:, 0, :]
                    else:
                        w_water_single = w_water
                        w_orig_single = w_orig
                    
                    # Convert indices to tensor if needed
                    if isinstance(self.latent_indices, np.ndarray):
                        latent_indices = torch.tensor(self.latent_indices, dtype=torch.long, device=self.device)
                    else:
                        latent_indices = self.latent_indices
                    
                    features_water = w_water_single[:, latent_indices]
                    features_orig = w_orig_single[:, latent_indices]
                
                # Generate true key using watermarked model features
                true_keys = self.key_mapper(features_water)
                
                # Predict keys based on decoder mode
                if self.direct_feature_decoder:
                    # Use features directly as input to the decoder
                    pred_keys_neg_logits = self.decoder(features_neg)
                    pred_keys_water_logits = self.decoder(features_water)
                    pred_keys_orig_logits = self.decoder(features_orig)
                else:
                    # Use the full image as input to the decoder
                    pred_keys_neg_logits = self.decoder(x_neg)
                    pred_keys_water_logits = self.decoder(x_water)
                    pred_keys_orig_logits = self.decoder(x_orig)
                
                # Convert to probabilities and binary keys
                pred_keys_neg_probs = torch.sigmoid(pred_keys_neg_logits)
                pred_keys_neg = pred_keys_neg_probs > 0.5
                
                pred_keys_water_probs = torch.sigmoid(pred_keys_water_logits)
                pred_keys_water = pred_keys_water_probs > 0.5
                
                pred_keys_orig_probs = torch.sigmoid(pred_keys_orig_logits)
                pred_keys_orig = pred_keys_orig_probs > 0.5
                
                # Save images with annotations
                for i in range(num_vis_samples):
                    # Determine match status
                    matches_reference = torch.all(pred_keys_neg[i] == true_keys[i]).item()
                    matches_watermarked = torch.all(pred_keys_water[i] == true_keys[i]).item()
                    matches_original = torch.all(pred_keys_orig[i] == true_keys[i]).item()
                    
                    # Log details
                    if getattr(self.config.evaluate, 'verbose_visualization', False):
                        logging.info(f"{output_subdir.split('/')[-1]} - Sample {i+1}:")
                        logging.info(f"  True key: {true_keys[i].detach().cpu().numpy().tolist()}")
                        logging.info(f"  Predicted key: {pred_keys_neg[i].detach().cpu().numpy().tolist()}")
                        logging.info(f"  Matches reference: {matches_reference}")
                    
                    # Save image with annotations
                    sample_filename = os.path.join(output_subdir, f"sample_{i+1}.png")
                    save_visualization(
                        image=x_neg_np[i],
                        true_key=true_keys[i].detach().cpu().numpy(),
                        pred_key=pred_keys_neg[i].detach().cpu().numpy(),
                        pred_probs=pred_keys_neg_probs[i].detach().cpu().numpy(),
                        output_path=sample_filename,
                        title=f"{output_subdir.split('/')[-1]} - Sample {i+1}",
                        match_status=matches_reference
                    )
                
            else:
                # Standard comparison case - watermarked vs original
                # Generate images from watermarked model
                w_water = self.watermarked_model.mapping(z_vis, None)
                x_water = self.watermarked_model.synthesis(w_water, noise_mode="const")
                
                # Generate images from original model
                w_orig = self.gan_model.mapping(z_vis, None)
                x_orig = self.gan_model.synthesis(w_orig, noise_mode="const")
                
                # Extract features
                if self.use_image_pixels:
                    features_water = self.extract_image_partial(x_water)
                    features_orig = self.extract_image_partial(x_orig)
                else:
                    # Extract latent features
                    if w_water.ndim == 3:
                        w_water_single = w_water[:, 0, :]
                        w_orig_single = w_orig[:, 0, :]
                    else:
                        w_water_single = w_water
                        w_orig_single = w_orig
                    
                    # Convert indices to tensor if needed
                    if isinstance(self.latent_indices, np.ndarray):
                        latent_indices = torch.tensor(self.latent_indices, dtype=torch.long, device=self.device)
                    else:
                        latent_indices = self.latent_indices
                    
                    features_water = w_water_single[:, latent_indices]
                    features_orig = w_orig_single[:, latent_indices]
                
                # Generate true key using watermarked image features
                true_keys = self.key_mapper(features_water)
                
                # Predict keys based on decoder mode
                if self.direct_feature_decoder:
                    # Use features directly as input to the decoder
                    pred_keys_water_logits = self.decoder(features_water)
                    pred_keys_orig_logits = self.decoder(features_orig)
                else:
                    # Use the full image as input to the decoder
                    pred_keys_water_logits = self.decoder(x_water)
                    pred_keys_orig_logits = self.decoder(x_orig)
                
                # Convert to probabilities and binary keys
                pred_keys_water_probs = torch.sigmoid(pred_keys_water_logits)
                pred_keys_water = pred_keys_water_probs > 0.5
                
                pred_keys_orig_probs = torch.sigmoid(pred_keys_orig_logits)
                pred_keys_orig = pred_keys_orig_probs > 0.5
                
                # Convert images to numpy for visualization
                x_water_np = x_water.detach().cpu().numpy()
                x_orig_np = x_orig.detach().cpu().numpy()
                
                # Create subdirectories for watermarked vs original
                watermarked_dir = os.path.join(output_subdir, "watermarked")
                original_dir = os.path.join(output_subdir, "original")
                os.makedirs(watermarked_dir, exist_ok=True)
                os.makedirs(original_dir, exist_ok=True)
                
                # Save images with annotations
                for i in range(num_vis_samples):
                    # Determine match status
                    matches_watermarked = torch.all(pred_keys_water[i] == true_keys[i]).item()
                    matches_original = torch.all(pred_keys_orig[i] == true_keys[i]).item()
                    
                    # Log details
                    if getattr(self.config.evaluate, 'verbose_visualization', False):
                        logging.info(f"Sample {i+1}:")
                        logging.info(f"  True key: {true_keys[i].detach().cpu().numpy().tolist()}")
                        logging.info(f"  Watermarked pred: {pred_keys_water[i].detach().cpu().numpy().tolist()} (Match: {matches_watermarked})")
                        logging.info(f"  Original pred: {pred_keys_orig[i].detach().cpu().numpy().tolist()} (Match: {matches_original})")
                    
                    # Save watermarked image with annotations
                    water_filename = os.path.join(watermarked_dir, f"sample_{i+1}.png")
                    save_visualization(
                        image=x_water_np[i],
                        true_key=true_keys[i].detach().cpu().numpy(),
                        pred_key=pred_keys_water[i].detach().cpu().numpy(),
                        pred_probs=pred_keys_water_probs[i].detach().cpu().numpy(),
                        output_path=water_filename,
                        title=f"Watermarked - Sample {i+1}",
                        match_status=matches_watermarked
                    )
                    
                    # Save original image with annotations
                    orig_filename = os.path.join(original_dir, f"sample_{i+1}.png")
                    save_visualization(
                        image=x_orig_np[i],
                        true_key=true_keys[i].detach().cpu().numpy(),
                        pred_key=pred_keys_orig[i].detach().cpu().numpy(),
                        pred_probs=pred_keys_orig_probs[i].detach().cpu().numpy(),
                        output_path=orig_filename,
                        title=f"Original - Sample {i+1}",
                        match_status=matches_original
                    )
    
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
        Run evaluation based on the specified mode.
        
        Args:
            evaluation_mode (str): Evaluation mode, one of 'batch', 'visual', or 'both'.
            
        Returns:
            dict: Evaluation metrics (if batch evaluation was performed).
        """
        metrics = None
        
        if self.rank == 0:
            # Log configuration information
            logging.info("Starting evaluation with the following configuration:")
            logging.info(f"  Evaluation mode: {evaluation_mode}")
            logging.info(f"  Approach: {'Image-based' if self.use_image_pixels else 'Latent-based'}")
            logging.info(f"  Direct feature decoder: {self.direct_feature_decoder}")
            if self.direct_feature_decoder and not self.use_image_pixels:
                logging.warning("  WARNING: direct_feature_decoder is enabled but use_image_pixels is False. "
                               "This configuration is not supported.")
            
            # Additional configuration information
            if self.use_image_pixels:
                logging.info(f"  Image pixel count: {self.image_pixel_count}")
                logging.info(f"  Image pixel seed: {self.image_pixel_set_seed}")
            else:
                logging.info(f"  Latent indices length: {len(self.latent_indices)}")
                logging.info(f"  Latent indices seed: {self.w_partial_set_seed}")
            
            logging.info(f"  Key length: {self.config.model.key_length}")
            logging.info(f"  Key mapper seed: {self.config.model.key_mapper_seed}")
        
        if evaluation_mode in ['batch', 'both']:
            metrics = self.evaluate_batch()
            
        if evaluation_mode in ['visual', 'both']:
            self.visualize_samples()
            
        return metrics 