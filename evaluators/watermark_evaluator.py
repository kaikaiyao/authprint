"""
Evaluator for StyleGAN watermarking.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import lpips
import numpy as np
from tqdm import tqdm

from config.default_config import Config
from models.decoder import Decoder
from models.key_mapper import KeyMapper
from models.model_utils import clone_model, load_stylegan2_model
from utils.checkpoint import load_checkpoint
from utils.metrics import calculate_metrics, save_metrics_plots, save_metrics_text
from utils.visualization import save_visualization


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
        
        # Parse latent indices for partial vector
        if isinstance(config.model.selected_indices, str):
            self.latent_indices = [int(idx) for idx in config.model.selected_indices.split(',')]
        else:
            self.latent_indices = config.model.selected_indices
        
        # Setup models
        self.setup_models()
    
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
        self.key_mapper = KeyMapper(
            input_dim=len(self.latent_indices),
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
    
    def evaluate_batch(self) -> Dict[str, float]:
        """
        Perform batch evaluation on multiple samples.
        
        Returns:
            dict: Evaluation metrics.
        """
        if self.rank == 0:
            logging.info(f"Running batch evaluation with {self.config.evaluate.num_samples} samples...")
        
        # Initialize metrics collection
        all_lpips_losses = []
        all_watermarked_mse_distances = []
        all_original_mse_distances = []
        all_watermarked_mae_distances = []
        all_original_mae_distances = []
        
        num_batches = (self.config.evaluate.num_samples + self.config.evaluate.batch_size - 1) // self.config.evaluate.batch_size
        watermarked_correct = 0
        original_correct = 0
        total_samples = 0
        
        # Only use tqdm progress bar on rank 0
        batch_iterator = tqdm(range(num_batches), desc="Evaluating batches") if self.rank == 0 else range(num_batches)
        
        for _ in batch_iterator:
            current_batch_size = min(self.config.evaluate.batch_size, self.config.evaluate.num_samples - total_samples)
            if current_batch_size <= 0:
                break
                
            with torch.no_grad():
                # Sample latent vectors
                z = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                
                # Generate original images
                w_orig = self.gan_model.mapping(z, None)
                x_orig = self.gan_model.synthesis(w_orig, noise_mode="const")
                
                # Generate watermarked images
                w_water = self.watermarked_model.mapping(z, None)
                x_water = self.watermarked_model.synthesis(w_water, noise_mode="const")
                
                # Extract latent partial and compute true key
                if w_orig.ndim == 3:
                    w_orig_single = w_orig[:, 0, :]
                    w_water_single = w_water[:, 0, :]
                else:
                    w_orig_single = w_orig
                    w_water_single = w_water
                
                w_partial = w_water_single[:, self.latent_indices]
                true_key = self.key_mapper(w_partial)
                
                # Compute key from watermarked image
                pred_key_water_logits = self.decoder(x_water)
                pred_key_water_probs = torch.sigmoid(pred_key_water_logits)
                pred_key_water = (pred_key_water_probs > 0.5).float()
                
                # Compute key from original image
                pred_key_orig_logits = self.decoder(x_orig)
                pred_key_orig_probs = torch.sigmoid(pred_key_orig_logits)
                pred_key_orig = (pred_key_orig_probs > 0.5).float()
                
                # Calculate distance metrics for watermarked images
                watermarked_mse_distance = torch.mean(torch.pow(pred_key_water_probs - true_key, 2), dim=1)
                watermarked_mae_distance = torch.mean(torch.abs(pred_key_water_probs - true_key), dim=1)
                
                # Calculate distance metrics for original images
                original_mse_distance = torch.mean(torch.pow(pred_key_orig_probs - true_key, 2), dim=1)
                original_mae_distance = torch.mean(torch.abs(pred_key_orig_probs - true_key), dim=1)
                
                # Store distances for ROC-AUC calculation
                all_watermarked_mse_distances.extend(watermarked_mse_distance.cpu().numpy())
                all_original_mse_distances.extend(original_mse_distance.cpu().numpy())
                all_watermarked_mae_distances.extend(watermarked_mae_distance.cpu().numpy())
                all_original_mae_distances.extend(original_mae_distance.cpu().numpy())
                
                # Calculate match rates (all bits must match)
                water_matches = (pred_key_water == true_key).all(dim=1).sum().item()
                orig_matches = (pred_key_orig == true_key).all(dim=1).sum().item()
                
                watermarked_correct += water_matches
                original_correct += orig_matches
                total_samples += current_batch_size
                
                # Calculate LPIPS loss between original and watermarked images
                lpips_losses = self.lpips_loss_fn(x_orig, x_water).squeeze().cpu().numpy()
                all_lpips_losses.extend(lpips_losses)
        
        # Calculate metrics
        metrics, y_data = calculate_metrics(
            all_watermarked_mse_distances,
            all_original_mse_distances,
            all_watermarked_mae_distances,
            all_original_mae_distances,
            watermarked_correct,
            original_correct,
            total_samples,
            all_lpips_losses
        )
        
        # Generate plots and visualizations - only on rank 0
        if self.rank == 0:
            # Log metrics
            logging.info(f"Watermarked match rate: {metrics['watermarked_match_rate']:.2f}%")
            logging.info(f"Original match rate: {metrics['original_match_rate']:.2f}%")
            logging.info(f"LPIPS loss avg: {metrics['watermarked_lpips_loss_avg']:.6f}, std: {metrics['watermarked_lpips_loss_std']:.6f}")
            logging.info(f"Watermarked MSE distance: {metrics['watermarked_mse_distance_avg']:.6f}±{metrics['watermarked_mse_distance_std']:.6f}")
            logging.info(f"Original MSE distance: {metrics['original_mse_distance_avg']:.6f}±{metrics['original_mse_distance_std']:.6f}")
            logging.info(f"Watermarked MAE distance: {metrics['watermarked_mae_distance_avg']:.6f}±{metrics['watermarked_mae_distance_std']:.6f}")
            logging.info(f"Original MAE distance: {metrics['original_mae_distance_avg']:.6f}±{metrics['original_mae_distance_std']:.6f}")
            logging.info(f"ROC-AUC score (MSE): {metrics['roc_auc_score_mse']:.6f}")
            logging.info(f"ROC-AUC score (MAE): {metrics['roc_auc_score_mae']:.6f}")
            
            # Save plots
            save_metrics_plots(
                metrics,
                y_data,
                all_watermarked_mse_distances,
                all_original_mse_distances,
                all_watermarked_mae_distances,
                all_original_mae_distances,
                self.config.output_dir
            )
            
            # Save metrics to file
            save_metrics_text(metrics, self.config.output_dir)
        
        return metrics
    
    def visualize_samples(self):
        """
        Generate and visualize samples for qualitative evaluation.
        Only perform on rank 0.
        """
        if self.rank != 0:
            return
            
        logging.info(f"Running visual evaluation with {self.config.evaluate.num_vis_samples} samples...")
        
        with torch.no_grad():
            # Sample latent vectors
            z_vis = torch.randn(self.config.evaluate.num_vis_samples, self.latent_dim, device=self.device)
            
            # Generate original images
            w_orig_vis = self.gan_model.mapping(z_vis, None)
            x_orig_vis = self.gan_model.synthesis(w_orig_vis, noise_mode="const")
            
            # Generate watermarked images
            w_water_vis = self.watermarked_model.mapping(z_vis, None)
            x_water_vis = self.watermarked_model.synthesis(w_water_vis, noise_mode="const")
            
            # Compute difference
            diff_vis = x_water_vis - x_orig_vis
            
            # Save visualizations
            save_visualization(x_orig_vis, x_water_vis, diff_vis, self.config.output_dir)
            
            # Extract latent partial and compute true key for each sample
            if w_water_vis.ndim == 3:
                w_water_single_vis = w_water_vis[:, 0, :]
            else:
                w_water_single_vis = w_water_vis
            
            w_partial_vis = w_water_single_vis[:, self.latent_indices]
            true_keys_vis = self.key_mapper(w_partial_vis)
            
            # Predict keys and probabilities for both original and watermarked images
            pred_keys_water_logits_vis = self.decoder(x_water_vis)
            pred_keys_water_probs_vis = torch.sigmoid(pred_keys_water_logits_vis)
            pred_keys_water_vis = pred_keys_water_probs_vis > 0.5
            
            pred_keys_orig_logits_vis = self.decoder(x_orig_vis)
            pred_keys_orig_probs_vis = torch.sigmoid(pred_keys_orig_logits_vis)
            pred_keys_orig_vis = pred_keys_orig_probs_vis > 0.5
            
            # Calculate distance metrics for each sample
            watermarked_mse_distances_vis = torch.mean(torch.pow(pred_keys_water_probs_vis - true_keys_vis, 2), dim=1)
            watermarked_mae_distances_vis = torch.mean(torch.abs(pred_keys_water_probs_vis - true_keys_vis), dim=1)
            
            original_mse_distances_vis = torch.mean(torch.pow(pred_keys_orig_probs_vis - true_keys_vis, 2), dim=1)
            original_mae_distances_vis = torch.mean(torch.abs(pred_keys_orig_probs_vis - true_keys_vis), dim=1)
            
            # Save key comparison to log
            for i in range(self.config.evaluate.num_vis_samples):
                logging.info(f"Sample {i}:")
                logging.info(f"  True key: {true_keys_vis[i].cpu().numpy().astype(int)}")
                logging.info(f"  Watermarked pred: {pred_keys_water_vis[i].cpu().numpy().astype(int)}")
                logging.info(f"  Original pred: {pred_keys_orig_vis[i].cpu().numpy().astype(int)}")
                logging.info(f"  Watermarked match: {(pred_keys_water_vis[i] == true_keys_vis[i]).all().item()}")
                logging.info(f"  Original match: {(pred_keys_orig_vis[i] == true_keys_vis[i]).all().item()}")
                logging.info(f"  Watermarked MSE distance: {watermarked_mse_distances_vis[i].item():.6f}")
                logging.info(f"  Original MSE distance: {original_mse_distances_vis[i].item():.6f}")
                logging.info(f"  Watermarked MAE distance: {watermarked_mae_distances_vis[i].item():.6f}")
                logging.info(f"  Original MAE distance: {original_mae_distances_vis[i].item():.6f}")
    
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