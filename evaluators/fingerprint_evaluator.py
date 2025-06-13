"""
Evaluator for generative model fingerprinting.
"""
import logging
import os
from typing import Dict, Optional, List, Tuple, Any

import torch
import numpy as np
import random

from config.default_config import Config
from models.decoder import DecoderSD_L, DecoderSD_M, DecoderSD_S, StyleGAN2Decoder
from models.base_model import BaseGenerativeModel
from utils.checkpoint import load_checkpoint
from utils.metrics import save_metrics_text, calculate_fid
from utils.image_transforms import (
    quantize_model_weights, 
    downsample_and_upsample
)
from utils.model_loading import (
    load_pretrained_models,
    STYLEGAN2_MODELS,
    STABLE_DIFFUSION_MODELS
)


class FingerprintEvaluator:
    """
    Evaluator for generative model fingerprinting.
    """
    def __init__(
        self,
        config: Config,
        local_rank: int,
        rank: int,
        world_size: int,
        device: torch.device,
        selected_pretrained_models: Optional[List[str]] = None,
        custom_pretrained_models: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            config (Config): Configuration object.
            local_rank (int): Local process rank.
            rank (int): Global process rank.
            world_size (int): Total number of processes.
            device (torch.device): Device to run evaluation on.
            selected_pretrained_models (Optional[List[str]]): List of pretrained model names to use.
                If None or empty, all default models will be used.
            custom_pretrained_models (Optional[Dict[str, Any]]): Dictionary mapping
                custom model names to their configurations.
        """
        self.config = config
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.device = device
        
        # Store pretrained model selections
        self.selected_pretrained_models = selected_pretrained_models
        self.custom_pretrained_models = custom_pretrained_models or {}
        
        # Enable timing logs
        self.enable_timing = getattr(self.config.evaluate, 'enable_timing_logs', True)
        if self.rank == 0 and self.enable_timing:
            logging.info("Timing logs are enabled for detailed process monitoring")
        
        # Initialize timing dictionary for tracking durations
        self.timing_stats = {}
        
        # Initialize models
        self.generative_model = None
        self.decoder = None
        
        # Initialize pixel selection parameters
        self.image_pixel_indices = None
        self.image_pixel_count = self.config.model.image_pixel_count
        self.image_pixel_set_seed = self.config.model.image_pixel_set_seed
        
        # Initialize prompt dataset if multi-prompt mode is enabled
        self.prompts: Optional[List[str]] = None
        if self.config.model.enable_multi_prompt:
            self._load_prompt_dataset()
        
        if self.rank == 0:
            logging.info(f"Using direct pixel prediction with {self.image_pixel_count} pixels and seed {self.image_pixel_set_seed}")
            if self.config.model.enable_multi_prompt:
                logging.info("Multi-prompt evaluation mode is enabled")
        
        # Initialize quantized models dictionary
        self.quantized_models = {}
        
        # Setup models
        self.setup_models()
        
        # Load pretrained models with custom configuration
        if self.selected_pretrained_models or self.custom_pretrained_models:
            # Create combined model dictionary
            model_dict = {}
            
            # Add selected default models
            if not self.selected_pretrained_models:
                # If no models specified, use all default models
                if self.config.model.model_type == "stylegan2":
                    model_dict.update(STYLEGAN2_MODELS)
                else:
                    model_dict.update(STABLE_DIFFUSION_MODELS)
            else:
                # Add only selected default models
                default_models = STYLEGAN2_MODELS if self.config.model.model_type == "stylegan2" else STABLE_DIFFUSION_MODELS
                for model_name in self.selected_pretrained_models:
                    if model_name in default_models:
                        model_dict[model_name] = default_models[model_name]
                    elif self.rank == 0:
                        logging.warning(f"Requested model '{model_name}' not found in default models")
            
            # Add custom models
            model_dict.update(self.custom_pretrained_models)
            
            # Load the models
            self.pretrained_models = load_pretrained_models(
                device=self.device,
                rank=self.rank,
                model_type=self.config.model.model_type,
                selected_models=model_dict,
                img_size=self.config.model.img_size,
                enable_cpu_offload=self.config.model.sd_enable_cpu_offload if self.config.model.model_type == "stable-diffusion" else False,
                dtype=getattr(torch, self.config.model.sd_dtype) if self.config.model.model_type == "stable-diffusion" else torch.float32
            )
        else:
            # Load all default models
            self.pretrained_models = load_pretrained_models(
                device=self.device,
                rank=self.rank,
                model_type=self.config.model.model_type,
                img_size=self.config.model.img_size,
                enable_cpu_offload=self.config.model.sd_enable_cpu_offload if self.config.model.model_type == "stable-diffusion" else False,
                dtype=getattr(torch, self.config.model.sd_dtype) if self.config.model.model_type == "stable-diffusion" else torch.float32
            )
    
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
    
    def setup_models(self):
        """
        Initialize and set up all models.
        """
        # Initialize generative model
        if self.rank == 0:
            logging.info(f"Loading {self.config.model.model_type} model...")
            
        model_class = self.config.model.get_model_class()
        model_kwargs = self.config.model.get_model_kwargs(self.device)
        self.generative_model = model_class(**model_kwargs)
        
        # Initialize decoder based on model type and size
        decoder_output_dim = self.image_pixel_count  # For direct pixel prediction
        
        if self.config.model.model_type == "stylegan2":
            self.decoder = StyleGAN2Decoder(
                image_size=self.config.model.img_size,
                channels=3,
                output_dim=decoder_output_dim
            ).to(self.device)
            if self.rank == 0:
                logging.info(f"Initialized StyleGAN2Decoder with output_dim={decoder_output_dim}")
        else:  # stable-diffusion
            decoder_class = {
                "S": DecoderSD_S,
                "M": DecoderSD_M,
                "L": DecoderSD_L
            }[self.config.model.sd_decoder_size]
            
            self.decoder = decoder_class(
                image_size=self.config.model.img_size,
                channels=3,
                output_dim=decoder_output_dim
            ).to(self.device)
            
            if self.rank == 0:
                logging.info(f"Initialized SD-Decoder-{self.config.model.sd_decoder_size} with output_dim={decoder_output_dim}")
        
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
        
        # Setup quantized models if enabled in evaluation config
        if getattr(self.config.evaluate, 'enable_quantization', True):
            try:
                if self.rank == 0:
                    logging.info("Setting up int8 quantized model...")
                
                try:
                    if self.config.model.model_type == "stylegan2":
                        int8_model = quantize_model_weights(self.generative_model, 'int8')
                    else:  # stable-diffusion
                        int8_model = self.generative_model.quantize('int8')
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
                    if self.config.model.model_type == "stylegan2":
                        int4_model = quantize_model_weights(self.generative_model, 'int4')
                    else:  # stable-diffusion
                        int4_model = self.generative_model.quantize('int4')
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
        else:
            if self.rank == 0:
                logging.info("Quantization disabled in evaluation config")
            self.quantized_models = {}
    
    def evaluate_batch(self) -> Dict[str, float]:
        """
        Run batch evaluation to compute metrics.
        
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        if self.rank == 0:
            logging.info(f"Running batch evaluation with {self.config.evaluate.num_samples} samples...")
            if self.config.model.enable_multi_prompt:
                logging.info("Using multi-prompt evaluation mode")
        
        try:
            # Set seed for reproducibility
            if hasattr(self.config.evaluate, 'seed') and self.config.evaluate.seed is not None:
                np.random.seed(self.config.evaluate.seed)
                torch.manual_seed(self.config.evaluate.seed)
                random.seed(self.config.evaluate.seed)  # For prompt sampling
                if self.rank == 0:
                    logging.info(f"Using fixed random seed {self.config.evaluate.seed} for evaluation")
            
            # Create empty accumulators
            batch_size = self.config.evaluate.batch_size
            num_samples = self.config.evaluate.num_samples
            num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
            
            # Generate latents or prepare generation based on model type
            if self.config.model.model_type == "stylegan2":
                all_z_original = torch.randn(num_samples, self.generative_model.z_dim, device=self.device)
                all_z_negative = torch.randn(num_samples, self.generative_model.z_dim, device=self.device)
                gen_kwargs = {"noise_mode": "const"}
            else:  # stable-diffusion
                all_z_original = None  # Not used for SD
                all_z_negative = None  # Not used for SD
                gen_kwargs = self.config.model.get_generation_kwargs()
            
            # Process batches for original model
            mse_per_sample = []  # Changed from mse_values to mse_per_sample
            
            with torch.no_grad():
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_samples)
                    current_batch_size = end_idx - start_idx
                    
                    # For Stable Diffusion, update prompts if multi-prompt mode is enabled
                    if self.config.model.model_type == "stable-diffusion":
                        prompts = self._sample_prompts(current_batch_size)
                        gen_kwargs["prompt"] = prompts
                        if self.rank == 0 and i == 0:  # Log sample prompts from first batch
                            logging.info(f"Sample prompts for evaluation: {prompts[:3]}")
                    
                    # For Stable Diffusion, update prompts if multi-prompt mode is enabled
                    if self.config.model.model_type == "stable-diffusion":
                        prompts = self._sample_prompts(current_batch_size)
                        gen_kwargs["prompt"] = prompts
                        if self.rank == 0 and i == 0:  # Log sample prompts from first batch
                            logging.info(f"Sample prompts for evaluation: {prompts[:3]}")
                    
                    # Generate images based on model type
                    if self.config.model.model_type == "stylegan2":
                        z = all_z_original[start_idx:end_idx]
                        x = self.generative_model.generate_images(
                            batch_size=current_batch_size,
                            device=self.device,
                            z=z,  # Pass the latent vectors explicitly
                            **gen_kwargs
                        )
                    else:  # stable-diffusion
                        x = self.generative_model.generate_images(
                            batch_size=current_batch_size,
                            device=self.device,
                            **gen_kwargs
                        )
                    
                    # Extract features (real pixel values)
                    features = self.extract_image_partial(x)
                    true_values = features
                    
                    # Predict values
                    pred_values = self.decoder(x)
                    
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
            negative_results = self._evaluate_negative_samples(all_z_original, all_z_negative, threshold, gen_kwargs)
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
    
    def _evaluate_negative_samples(
        self,
        original_z: Optional[torch.Tensor],
        negative_z: Optional[torch.Tensor],
        threshold: float,
        gen_kwargs: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate negative samples by comparing against pretrained models and transformations.
        Computes FPR at 95% TPR threshold for each negative case.
        
        Args:
            original_z (Optional[torch.Tensor]): Original model latent vectors for FID comparison (StyleGAN2 only)
            negative_z (Optional[torch.Tensor]): Negative model latent vectors for evaluation (StyleGAN2 only)
            threshold (float): MSE threshold at 95% TPR from original model
            gen_kwargs (Dict[str, Any]): Generation kwargs for the model
            
        Returns:
            dict: Dictionary mapping negative sample types to their metrics
        """
        negative_results = {}
        batch_size = self.config.evaluate.batch_size
        num_samples = self.config.evaluate.num_samples
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        # Define the evaluations to run
        evaluations_to_run = []
        
        # Add pretrained model evaluations - use all available models
        for model_name in self.pretrained_models.keys():
            evaluations_to_run.append((model_name, None))
        
        # Add transformations based on model type
        if self.config.model.model_type == "stylegan2":
            # Only add quantization for StyleGAN2 models
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
        evaluations_to_run.append((None, 'downsample_16'))
        evaluations_to_run.append((None, 'downsample_224'))
        if self.rank == 0:
            logging.info("Added downsample evaluations for sizes 16 and 224")
        
        # Run evaluations
        total_evals = len(evaluations_to_run)
        if self.rank == 0:
            logging.info(f"Running {total_evals} evaluations...")
            logging.info(f"Evaluations to run: {[e[1] if e[1] else e[0] for e in evaluations_to_run]}")
        
        # Generate original images for FID comparison
        original_images = []
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                current_batch_size = end_idx - start_idx
                
                # Generate images based on model type
                if self.config.model.model_type == "stylegan2":
                    z = original_z[start_idx:end_idx]
                    x = self.generative_model.generate_images(
                        batch_size=current_batch_size,
                        device=self.device,
                        z=z,  # Pass the latent vectors explicitly
                        **gen_kwargs
                    )
                else:  # stable-diffusion
                    x = self.generative_model.generate_images(
                        batch_size=current_batch_size,
                        device=self.device,
                        **gen_kwargs
                    )
                original_images.append(x)
        
        # Concatenate all original images
        original_images = torch.cat(original_images, dim=0)
        
        # Generate negative case images for FID comparison
        with torch.no_grad():
            for idx, (model_name, transformation) in enumerate(evaluations_to_run):
                key = model_name if model_name else transformation
                if self.rank == 0:
                    logging.info(f"Starting evaluation for: {key}")
                
                mse_per_sample = []
                negative_images = []
                
                # Process in batches
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_samples)
                    current_batch_size = end_idx - start_idx
                    
                    # Generate negative sample images
                    if model_name is not None:
                        # Use pretrained model
                        model = self.pretrained_models[model_name]
                        if self.config.model.model_type == "stylegan2":
                            z = negative_z[start_idx:end_idx]
                            x = model.generate_images(
                                batch_size=current_batch_size,
                                device=self.device,
                                z=z,  # Pass the latent vectors explicitly
                                **gen_kwargs
                            )
                        else:  # stable-diffusion
                            x = model.generate_images(
                                batch_size=current_batch_size,
                                device=self.device,
                                **gen_kwargs
                            )
                    else:
                        # Generate base images
                        if self.config.model.model_type == "stylegan2":
                            z = negative_z[start_idx:end_idx]
                            x = self.generative_model.generate_images(
                                batch_size=current_batch_size,
                                device=self.device,
                                z=z,  # Pass the latent vectors explicitly
                                **gen_kwargs
                            )
                        else:  # stable-diffusion
                            x = self.generative_model.generate_images(
                                batch_size=current_batch_size,
                                device=self.device,
                                **gen_kwargs
                            )
                        
                        # Apply transformations
                        if transformation and transformation.startswith('quantization'):
                            # For quantization, use the pre-quantized models
                            precision = transformation.split('_')[-1]
                            if precision in self.quantized_models:
                                model = self.quantized_models[precision]
                                if self.config.model.model_type == "stylegan2":
                                    x = model.generate_images(
                                        batch_size=current_batch_size,
                                        device=self.device,
                                        z=z,  # Pass the latent vectors explicitly
                                        **gen_kwargs
                                    )
                                else:  # stable-diffusion
                                    x = model.generate_images(
                                        batch_size=current_batch_size,
                                        device=self.device,
                                        **gen_kwargs
                                    )
                            else:
                                if self.rank == 0 and i == 0:
                                    logging.warning(f"Quantized model for precision {precision} not found")
                        elif transformation.startswith('downsample'):
                            downsample_size = int(transformation.split('_')[1])
                            x = downsample_and_upsample(x, downsample_size=downsample_size)
                    
                    # Store images for FID calculation
                    negative_images.append(x)
                    
                    # Extract features (real pixel values)
                    features = self.extract_image_partial(x)
                    true_values = features
                    
                    # Predict values
                    pred_values = self.decoder(x)
                    
                    # Calculate per-sample MSE
                    mse = torch.mean(torch.pow(pred_values - true_values, 2), dim=1).cpu().numpy()
                    mse_per_sample.extend(mse.tolist())
                
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

    def _load_prompt_dataset(self) -> None:
        """
        Load prompts from the dataset file.
        """
        if self.config.model.prompt_source == "local":
            self._load_prompt_dataset_local()
        else:  # diffusiondb
            self._load_prompt_dataset_diffusiondb()

    def _load_prompt_dataset_local(self) -> None:
        """
        Load prompts from a local file.
        """
        if self.rank == 0:
            logging.info(f"Loading prompts from local file: {self.config.model.prompt_dataset_path}")
        
        try:
            with open(self.config.model.prompt_dataset_path, 'r', encoding='utf-8') as f:
                all_prompts = [line.strip() for line in f if line.strip()]
            
            # Sample the specified number of prompts
            if len(all_prompts) > self.config.model.prompt_dataset_size:
                self.prompts = random.sample(all_prompts, self.config.model.prompt_dataset_size)
            else:
                self.prompts = all_prompts
                if self.rank == 0:
                    logging.warning(f"Prompt dataset contains fewer prompts ({len(all_prompts)}) "
                                  f"than requested ({self.config.model.prompt_dataset_size})")
            
            if self.rank == 0:
                logging.info(f"Loaded {len(self.prompts)} prompts from local file")
                logging.info(f"Sample prompts: {self.prompts[:3]}")
        
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error loading prompt dataset: {str(e)}")
            raise

    def _load_prompt_dataset_diffusiondb(self) -> None:
        """
        Load prompts from DiffusionDB dataset.
        """
        if self.rank == 0:
            logging.info("Loading prompts from DiffusionDB dataset")
        
        try:
            from datasets import load_dataset
            from trainers.fingerprint_trainer import clean_prompt  # Reuse the clean_prompt function
            
            # Load the metadata table from DiffusionDB
            subset_mapping = {
                "2m_random_10k": "2m_random_10k",  # Using random 10k subset instead of full dataset
                "large_random_10k": "large_random_10k",
                "2m_random_5k": "2m_random_5k",
            }
            subset = subset_mapping[self.config.model.diffusiondb_subset]
            dataset = load_dataset("poloclub/diffusiondb", subset, split="train", trust_remote_code=True)
            
            # Extract and clean all unique prompts
            all_prompts = []
            raw_prompts = list(set(dataset["prompt"]))
            
            if self.rank == 0:
                logging.info(f"Found {len(raw_prompts)} unique prompts before cleaning")
            
            for prompt in raw_prompts:
                if not prompt:  # Skip empty prompts
                    continue
                    
                cleaned_prompt = clean_prompt(prompt)
                if cleaned_prompt and len(cleaned_prompt.split()) <= 50:  # Only keep reasonably sized prompts
                    all_prompts.append(cleaned_prompt)
            
            if self.rank == 0:
                logging.info(f"Retained {len(all_prompts)} prompts after cleaning")
            
            # Sample the specified number of prompts
            if len(all_prompts) > self.config.model.prompt_dataset_size:
                self.prompts = random.sample(all_prompts, self.config.model.prompt_dataset_size)
            else:
                self.prompts = all_prompts
                if self.rank == 0:
                    logging.warning(f"DiffusionDB contains fewer clean prompts ({len(all_prompts)}) "
                                  f"than requested ({self.config.model.prompt_dataset_size})")
            
            if self.rank == 0:
                logging.info(f"Loaded {len(self.prompts)} prompts from DiffusionDB")
                logging.info("Sample prompts after cleaning:")
                for i, prompt in enumerate(self.prompts[:10]):  # Show first 10 prompts
                    logging.info(f"  {i+1}. {prompt}")
        
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error loading DiffusionDB dataset: {str(e)}")
            raise

    def _sample_prompts(self, batch_size: int) -> List[str]:
        """
        Sample prompts for the current batch.
        
        Args:
            batch_size (int): Number of prompts to sample.
            
        Returns:
            List[str]: List of sampled prompts.
        """
        if not self.config.model.enable_multi_prompt or not self.prompts:
            return [self.config.model.sd_prompt] * batch_size
        
        return random.sample(self.prompts, min(batch_size, len(self.prompts)))