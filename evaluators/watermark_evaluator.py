"""
Evaluator for StyleGAN watermarking.
"""
import logging
import os
from typing import Dict, Optional

import torch
import numpy as np

from config.default_config import Config
from models.decoder import Decoder
from models.model_utils import load_stylegan2_model
from utils.checkpoint import load_checkpoint
from utils.metrics import save_metrics_text, calculate_fid
from utils.image_transforms import (
    quantize_model_weights, 
    downsample_and_upsample
)
from utils.model_loading import load_pretrained_models


class WatermarkEvaluator:
    """
    Evaluator for StyleGAN watermarking with direct pixel prediction.
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
        
        # Enable timing logs
        self.enable_timing = getattr(self.config.evaluate, 'enable_timing_logs', True)
        if self.rank == 0 and self.enable_timing:
            logging.info("Timing logs are enabled for detailed process monitoring")
        
        # Initialize timing dictionary for tracking durations
        self.timing_stats = {}
        
        # Initialize models
        self.gan_model = None
        self.decoder = None
        
        # Initialize pixel selection parameters
        self.image_pixel_indices = None
        self.image_pixel_count = self.config.model.image_pixel_count
        self.image_pixel_set_seed = self.config.model.image_pixel_set_seed
        
        if self.rank == 0:
            logging.info(f"Using direct pixel prediction with {self.image_pixel_count} pixels and seed {self.image_pixel_set_seed}")
        
        # Initialize quantized models dictionary
        self.quantized_models = {}
        
        # Setup models
        self.setup_models()
        
        # Load retrained models
        self.pretrained_models = load_pretrained_models(device, rank)
    
    def _generate_pixel_indices(self) -> None:
        """
        Generate random pixel indices for image-based approach.
        """
        # Set seed for reproducibility
        torch.manual_seed(self.image_pixel_set_seed)
        
        # Calculate total number of pixels
        img_size = self.config.model.img_size
        channels = 3  # RGB image
        total_pixels = channels * img_size * img_size
        
        # Generate random indices (without replacement) on CPU first
        if self.image_pixel_count > total_pixels:
            if self.rank == 0:
                logging.warning(f"Requested {self.image_pixel_count} pixels exceeds total pixels {total_pixels}. Using all pixels.")
            self.image_pixel_count = total_pixels
            self.image_pixel_indices = torch.arange(total_pixels)
        else:
            self.image_pixel_indices = torch.randperm(total_pixels)[:self.image_pixel_count]
        
        # Move to device after generation
        self.image_pixel_indices = self.image_pixel_indices.to(self.device)
        
        if self.rank == 0:
            logging.info(f"Generated {len(self.image_pixel_indices)} pixel indices with seed {self.image_pixel_set_seed}")
            logging.info(f"Selected pixel indices: {self.image_pixel_indices.tolist()}")

    def extract_image_partial(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract partial image using selected pixel indices.
        
        Args:
            images (torch.Tensor): Batch of images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Batch of flattened pixel values at selected indices
        """
        batch_size = images.shape[0]
        
        # Flatten the spatial dimensions using a view operation
        flattened = images.view(batch_size, -1)
        
        # Get values at selected indices: [batch_size, pixel_count]
        image_partial = flattened.index_select(1, self.image_pixel_indices)
        
        return image_partial
    
    def _mask_selected_pixels(self, images: torch.Tensor) -> torch.Tensor:
        """
        Mask selected pixels in the images by setting them to -1.
        
        Args:
            images (torch.Tensor): Input images [batch_size, channels, height, width].
            
        Returns:
            torch.Tensor: Images with selected pixels masked.
        """
        # Create a copy of the images to avoid modifying the original
        masked_images = images.clone()
        
        # Flatten images for masking
        batch_size = images.size(0)
        flattened = masked_images.view(batch_size, -1)
        
        # Set selected pixels to -1
        flattened[:, self.image_pixel_indices] = -1
        
        # Reshape back to original shape
        return flattened.view_as(images)
    
    def setup_models(self):
        """
        Initialize and set up all models.
        """
        # Load pretrained StyleGAN2 model
        self.gan_model = load_stylegan2_model(
            self.config.model.stylegan2_url,
            self.config.model.stylegan2_local_path,
            self.device
        )
        
        # Initialize decoder for direct pixel prediction
        self.decoder = Decoder(
            image_size=self.config.model.img_size,
            channels=3,
            output_dim=self.image_pixel_count  # Output dimension is number of pixels to predict
        ).to(self.device)
        self.decoder.eval()
        
        # Generate pixel indices
        self._generate_pixel_indices()
        
        # Load checkpoint
        if self.rank == 0:
            logging.info(f"Loading checkpoint from {self.config.checkpoint_path}...")
        
        load_checkpoint(
            checkpoint_path=self.config.checkpoint_path,
            decoder=self.decoder,
            device=self.device
        )
        
        try:
            # Setup quantized models - always set up int8 and int4
            if self.rank == 0:
                logging.info("Setting up int8 quantized model...")
            
            try:
                int8_model = quantize_model_weights(self.gan_model, 'int8')
                int8_model.eval()
                self.quantized_models['int8'] = int8_model
                if self.rank == 0:
                    logging.info("Successfully created int8 quantized model")
            except Exception as e:
                if self.rank == 0:
                    logging.error(f"Failed to create int8 model: {str(e)}")
                    logging.error("int8 quantization error:", exc_info=True)
            
            if self.rank == 0:
                logging.info("Setting up int4 quantized model...")
            
            try:
                int4_model = quantize_model_weights(self.gan_model, 'int4')
                int4_model.eval()
                self.quantized_models['int4'] = int4_model
                if self.rank == 0:
                    logging.info("Successfully created int4 quantized model")
            except Exception as e:
                if self.rank == 0:
                    logging.error(f"Failed to create int4 model: {str(e)}")
                    logging.error("int4 quantization error:", exc_info=True)
            
            if self.rank == 0:
                logging.info(f"Available quantized models: {list(self.quantized_models.keys())}")
        
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error in quantization setup: {str(e)}")
                logging.error("Quantization setup error:", exc_info=True)
                logging.error("Continuing without quantized models...")
            self.quantized_models = {}
    
    def evaluate_batch(self) -> Dict[str, float]:
        """
        Run batch evaluation to compute metrics.
        
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        if self.rank == 0:
            logging.info(f"Running batch evaluation with {self.config.evaluate.num_samples} samples...")
        
        try:
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
            
            # Generate all latent vectors upfront for consistency
            all_z_original = torch.randn(num_samples, self.gan_model.z_dim, device=self.device)
            all_z_negative = torch.randn(num_samples, self.gan_model.z_dim, device=self.device)
            
            # Process batches for original model
            mse_per_sample = []  # Changed from mse_values to mse_per_sample
            
            with torch.no_grad():
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_samples)
                    
                    # Extract batch of latent vectors
                    z = all_z_original[start_idx:end_idx]
                    
                    # Generate images
                    if hasattr(self.gan_model, 'module'):
                        w = self.gan_model.module.mapping(z, None)
                        x = self.gan_model.module.synthesis(w, noise_mode="const")
                    else:
                        w = self.gan_model.mapping(z, None)
                        x = self.gan_model.synthesis(w, noise_mode="const")
                    
                    # Extract features (real pixel values)
                    features = self.extract_image_partial(x)
                    true_values = features
                    
                    # Mask selected pixels and predict values
                    masked_x = self._mask_selected_pixels(x)
                    pred_values = self.decoder(masked_x)
                    
                    # Calculate metrics - now calculating MSE per sample
                    mse = torch.mean(torch.pow(pred_values - true_values, 2), dim=1).cpu().numpy()
                    mse_per_sample.extend(mse.tolist())  # extend list with individual sample MSEs
                    
                    # Progress reporting
                    if self.rank == 0 and num_batches > 10 and (i+1) % max(1, num_batches//10) == 0:
                        logging.info(f"Processed {i+1}/{num_batches} batches")
            
            # Convert to numpy array for calculations
            mse_per_sample = np.array(mse_per_sample)
            
            # Combine results - now taking mean and std of per-sample MSEs
            mse_all = np.mean(mse_per_sample)
            mse_std = np.std(mse_per_sample)
            
            # Calculate threshold for 95% TPR - using per-sample MSEs
            threshold = np.percentile(mse_per_sample, 95)
            if self.rank == 0:
                logging.info(f"Threshold at 95% TPR: {threshold:.6f}")
            
            # Calculate metrics
            metrics = {
                'pixel_mse_mean': mse_all,
                'pixel_mse_std': mse_std,
                'pixel_mse_values': mse_per_sample,  # Now storing all per-sample MSEs
                'threshold_95tpr': threshold
            }
            
            # Evaluate negative samples
            negative_results = self._evaluate_negative_samples(all_z_original,all_z_negative, threshold)
            if negative_results:
                metrics['negative_results'] = negative_results
            
            # Save metrics
            if self.rank == 0:
                save_metrics_text(metrics, self.config.output_dir)
            
            return metrics
                
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error in batch evaluation: {str(e)}")
                logging.error(str(e), exc_info=True)
            return {}
    
    def _evaluate_negative_samples(self, all_z_original, all_z_negative, threshold):
        """
        Evaluate negative samples by comparing against pretrained models and transformations.
        Computes FPR at 95% TPR threshold for each negative case.
        
        Args:
            all_z (torch.Tensor): All latent vectors for evaluation
            threshold (float): MSE threshold at 95% TPR from original model
            
        Returns:
            dict: Dictionary mapping negative sample types to their metrics
        """
        negative_results = {}
        batch_size = self.config.evaluate.batch_size
        num_samples = self.config.evaluate.num_samples
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        # Define the evaluations to run
        evaluations_to_run = []
        
        # Add pretrained model evaluations
        for model_name in ['ffhq70k-ada', 'ffhq1k', 'ffhq30k', 'ffhq70k-bcr', 'ffhq70k-noaug']:
            if model_name in self.pretrained_models:
                evaluations_to_run.append((model_name, None))
        
        # Add transformations - only add quantization if models are available
        if self.quantized_models:
            if 'int8' in self.quantized_models:
                evaluations_to_run.append((None, 'quantization_int8'))
                if self.rank == 0:
                    logging.info(f"Added int8 quantization evaluation")
            if 'int4' in self.quantized_models:
                evaluations_to_run.append((None, 'quantization_int4'))
                if self.rank == 0:
                    logging.info(f"Added int4 quantization evaluation")
            if self.rank == 0:
                logging.info(f"Added quantization evaluations for: {list(self.quantized_models.keys())}")
        else:
            if self.rank == 0:
                logging.warning("No quantized models available, skipping quantization evaluation")
        
        # Add downsample evaluations for both sizes
        evaluations_to_run.append((None, 'downsample_128'))
        evaluations_to_run.append((None, 'downsample_224'))
        if self.rank == 0:
            logging.info("Added downsample evaluations for sizes 128 and 224")
        
        # Run evaluations
        total_evals = len(evaluations_to_run)
        if self.rank == 0:
            logging.info(f"Running {total_evals} evaluations...")
            logging.info(f"Evaluations to run: {[e[1] if e[1] else e[0] for e in evaluations_to_run]}")
        
        # Generate original model images for FID comparison
        original_images = []
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                z = all_z_original[start_idx:end_idx]
                
                if hasattr(self.gan_model, 'module'):
                    w = self.gan_model.module.mapping(z, None)
                    x = self.gan_model.module.synthesis(w, noise_mode="const")
                else:
                    w = self.gan_model.mapping(z, None)
                    x = self.gan_model.synthesis(w, noise_mode="const")
                
                original_images.append(x)
        
        original_images = torch.cat(original_images, dim=0)
        
        # Generate negative case images for FID comparison
        with torch.no_grad():
            for idx, (model_name, transformation) in enumerate(evaluations_to_run):
                key = model_name if model_name else transformation
                if self.rank == 0:
                    logging.info(f"Starting evaluation for: {key}")
                
                mse_per_sample = []  # Changed from mse_per_batch to mse_per_sample
                negative_images = []
                
                # Process in batches
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_samples)
                    
                    # Extract batch of latent vectors
                    z = all_z_negative[start_idx:end_idx]
                    
                    # Generate negative sample images
                    if model_name is not None:
                        # Use pretrained model
                        model = self.pretrained_models[model_name]
                        
                        if hasattr(model, 'module'):
                            w = model.module.mapping(z, None)
                            x = model.module.synthesis(w, noise_mode="const")
                        else:
                            w = model.mapping(z, None)
                            x = model.synthesis(w, noise_mode="const")
                    else:
                        # Generate base images
                        if hasattr(self.gan_model, 'module'):
                            w = self.gan_model.module.mapping(z, None)
                            x = self.gan_model.module.synthesis(w, noise_mode="const")
                        else:
                            w = self.gan_model.mapping(z, None)
                            x = self.gan_model.synthesis(w, noise_mode="const")
                        
                        # Apply transformations
                        if transformation and transformation.startswith('quantization'):
                            # For quantization, use the pre-quantized models
                            precision = transformation.split('_')[-1]
                            if precision in self.quantized_models:
                                model = self.quantized_models[precision]
                                if hasattr(model, 'module'):
                                    w = model.module.mapping(z, None)
                                    x = model.module.synthesis(w, noise_mode="const")
                                else:
                                    w = model.mapping(z, None)
                                    x = model.synthesis(w, noise_mode="const")
                                if self.rank == 0 and i == 0:
                                    logging.info(f"Using {precision} quantized model for batch")
                            else:
                                if self.rank == 0 and i == 0:
                                    logging.warning(f"Quantized model for precision {precision} not found")
                        elif transformation.startswith('downsample'):
                            if self.rank == 0 and i == 0:
                                downsample_size = int(transformation.split('_')[1])
                                logging.info(f"Applying downsample transformation with size {downsample_size}")
                            downsample_size = int(transformation.split('_')[1])
                            x = downsample_and_upsample(x, downsample_size=downsample_size)
                    
                    # Store images for FID calculation
                    negative_images.append(x)
                    
                    # Extract features (real pixel values)
                    features = self.extract_image_partial(x)
                    true_values = features
                    
                    # Mask selected pixels and predict values
                    masked_x = self._mask_selected_pixels(x)
                    pred_values = self.decoder(masked_x)
                    
                    # Calculate per-sample MSE - mean over features dimension only, not batch
                    mse = torch.mean(torch.pow(pred_values - true_values, 2), dim=1).cpu().numpy()
                    mse_per_sample.extend(mse.tolist())  # extend list with individual sample MSEs
                
                # Combine all negative images
                negative_images = torch.cat(negative_images, dim=0)
                
                # Calculate FID score
                fid_score = calculate_fid(original_images, negative_images, batch_size=batch_size, device=self.device)
                
                # Convert to numpy array for calculations
                mse_per_sample = np.array(mse_per_sample)
                
                # Calculate metrics
                mse_all = np.mean(mse_per_sample)
                mse_std = np.std(mse_per_sample)
                
                # Calculate FPR at 95% TPR threshold - now using per-sample MSEs
                fpr = np.mean(mse_per_sample <= threshold)  # Proportion of negative samples below threshold
                
                # Store metrics
                negative_results[key] = {
                    'mse_mean': mse_all,
                    'mse_std': mse_std,
                    'mse_values': mse_per_sample,  # Now storing all per-sample MSEs
                    'fpr_at_95tpr': fpr,
                    'fid_score': fid_score
                }
                
                if self.rank == 0:
                    logging.info(f"FPR at 95% TPR for {key}: {fpr:.4f} (computed over {len(mse_per_sample)} samples)")
                    logging.info(f"FID score for {key}: {fid_score:.4f}")
                
                # Progress reporting
                if self.rank == 0 and (idx+1) % max(1, total_evals//5) == 0:
                    logging.info(f"Completed {idx+1}/{total_evals} evaluations")
        
        return negative_results
    
    def evaluate(self):
        """
        Run batch evaluation to compute metrics.
        
        Returns:
            dict: Evaluation metrics.
        """
        # Log evaluation configuration
        if self.rank == 0:
            logging.info("Starting evaluation with the following configuration:")
            logging.info(f"  Pixel count: {self.image_pixel_count}")
            logging.info(f"  Pixel seed: {self.image_pixel_set_seed}")
        
        metrics = self.evaluate_batch()
        
        # Print metrics table if we have results and are on rank 0
        if metrics and self.rank == 0:
            logging.info("\nEvaluation Results:")
            logging.info("-" * 150)
            logging.info(f"{'Model/Transform':<40}{'Avg MSE':>15}{'Std MSE':>15}{'FPR@95%TPR':>15}{'FID Score':>15}")
            logging.info("-" * 150)
            
            # Print original model results and threshold
            logging.info(f"{'Original Model':<40}{metrics['pixel_mse_mean']:>15.6f}{metrics['pixel_mse_std']:>15.6f}{'N/A':>15}{'N/A':>15}")
            logging.info(f"Threshold at 95% TPR: {metrics['threshold_95tpr']:.6f}")
            
            # Print negative sample results if available
            if 'negative_results' in metrics:
                logging.info("\nNegative Sample Results:")
                for name, result in metrics['negative_results'].items():
                    logging.info(f"{name:<40}{result['mse_mean']:>15.6f}{result['mse_std']:>15.6f}{result['fpr_at_95tpr']:>15.4f}{result['fid_score']:>15.4f}")
            
            logging.info("-" * 150)
        
        return metrics