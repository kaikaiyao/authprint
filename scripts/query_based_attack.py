#!/usr/bin/env python
"""
Query-based attack script for StyleGAN watermarking.
"""
import argparse
import logging
import os
import sys
import lpips
import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

# Add the parent directory (project root) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.default_config import get_default_config
from models.model_utils import load_stylegan2_model
from utils.distributed import setup_distributed, cleanup_distributed
from utils.logging_utils import setup_logging
from utils.image_transforms import quantize_model_weights, downsample_and_upsample
from utils.model_loading import load_pretrained_models
from utils.metrics import calculate_fid, InceptionV3


class ImageQualityMetrics:
    """Wrapper for various image quality metrics."""
    def __init__(self, device):
        self.device = device
        # Initialize LPIPS
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        # Initialize SSIM
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        # Initialize PSNR
        self.psnr = PeakSignalNoiseRatio().to(device)
    
    def compute_metrics(self, original, perturbed):
        """Compute all image quality metrics."""
        return {
            'lpips': self.lpips_fn(original, perturbed).item(),
            'psnr': self.psnr(original, perturbed).item(),
            'ssim': self.ssim(original, perturbed).item()
        }


class DecoderWrapper:
    """Wrapper for the decoder to provide binary prediction interface."""
    def __init__(self, decoder, threshold, image_pixel_indices):
        self.decoder = decoder
        self.threshold = threshold
        self.image_pixel_indices = image_pixel_indices
    
    def extract_features(self, x):
        """Extract features using the same method as the evaluator."""
        batch_size = x.shape[0]
        flattened = x.view(batch_size, -1)
        return flattened[:, self.image_pixel_indices]
    
    def predict(self, x):
        """Return True if image is detected as original, False if undetected.
        Calculates MSE per sample, matching evaluator's implementation."""
        with torch.no_grad():
            features = self.extract_features(x)
            pred_values = self.decoder(x)
            # Calculate MSE per sample (dim=1)
            mse = torch.mean(torch.pow(pred_values - features, 2), dim=1)
            # Compare each sample's MSE with threshold
            return mse <= self.threshold


class QueryBasedAttack:
    """Query-based attack implementation for StyleGAN watermarking."""
    def __init__(
        self,
        original_model,
        decoder,
        epsilon=0.1,
        max_queries=1000,
        binary_search_steps=10,
        device=None,
        rank=0,
        batch_size=16
    ):
        self.original_model = original_model
        self.decoder = decoder
        self.epsilon = epsilon
        self.max_queries = max_queries
        self.binary_search_steps = binary_search_steps
        self.device = device or next(decoder.parameters()).device
        self.rank = rank
        self.batch_size = batch_size
        
        # Initialize quality metrics for evaluation only
        self.quality_metrics = ImageQualityMetrics(self.device)
    
    def apply_perturbation(self, image, alpha, z):
        """Apply controlled perturbation to image.
        
        Args:
            image: The negative case image to perturb
            alpha: Interpolation coefficient
            z: The latent vector used to generate both negative and original images
            negative_case_type: Type of negative case being attacked
        """
        # Generate original image using the same z vector
        if hasattr(self.original_model, 'module'):
            w = self.original_model.module.mapping(z, None)
            original_img = self.original_model.module.synthesis(w, noise_mode="const")
        else:
            w = self.original_model.mapping(z, None)
            original_img = self.original_model.synthesis(w, noise_mode="const")
        
        # Interpolate between negative and original image
        perturbed = (1 - alpha) * image + alpha * original_img
        
        # Ensure perturbation is within epsilon bound
        delta = perturbed - image
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        perturbed = image + delta
        
        return perturbed
    
    def binary_search_perturbation(self, image, z, low=0.0, high=1.0):
        """Binary search for minimal perturbation.
        For all cases (both original and negative), we want to find a perturbation
        that makes the decoder predict True (think it's an original image).
        For original model (control case), this should be easy as they already pass.
        For negative cases, we need to find perturbation that makes them pass.
        """
        best_perturbed = None
        best_alpha = None
        
        # Check if the image already passes
        initial_prediction = self.decoder.predict(image)
        if self.rank == 0:
            with torch.no_grad():
                initial_mse = torch.mean(torch.pow(self.decoder.decoder(image) - self.decoder.extract_features(image), 2), dim=1)
                logging.info(f"Initial state - Prediction: {initial_prediction.cpu().numpy()[0]}, MSE: {initial_mse.cpu().numpy()[0]:.6f} (threshold: {self.decoder.threshold:.6f})")
        
        if initial_prediction.all():
            # If it already passes, just return the original image
            if self.rank == 0:
                logging.info("Image already passes detection, no perturbation needed")
            return image, 0.0
        
        for step in range(self.binary_search_steps):
            alpha = (low + high) / 2
            perturbed = self.apply_perturbation(image, alpha, z)
            current_prediction = self.decoder.predict(perturbed)
            
            if self.rank == 0:
                with torch.no_grad():
                    current_mse = torch.mean(torch.pow(self.decoder.decoder(perturbed) - self.decoder.extract_features(perturbed), 2), dim=1)
                    logging.info(f"Step {step+1}/{self.binary_search_steps} - "
                               f"Alpha: {alpha:.4f}, "
                               f"Prediction: {current_prediction.cpu().numpy()[0]}, "
                               f"MSE: {current_mse.cpu().numpy()[0]:.6f}")
            
            if current_prediction.all():  # Successfully fooled - decoder thinks it's original
                high = alpha
                best_perturbed = perturbed
                best_alpha = alpha
                if self.rank == 0:
                    logging.info(f"Found successful perturbation at alpha={alpha:.4f}")
            else:
                low = alpha
        
        if best_perturbed is not None and self.rank == 0:
            with torch.no_grad():
                final_mse = torch.mean(torch.pow(self.decoder.decoder(best_perturbed) - self.decoder.extract_features(best_perturbed), 2), dim=1)
                logging.info(f"Final result - Alpha: {best_alpha:.4f}, MSE: {final_mse.cpu().numpy()[0]:.6f}")
        elif self.rank == 0:
            logging.info("Failed to find successful perturbation")
        
        return best_perturbed, best_alpha
    
    def attack_negative_case(self, negative_model, num_samples, negative_case_type=None):
        """Attack a specific negative case."""
        successful_attacks = 0
        total_queries = 0
        results = []
        
        # Lists to store images for FID calculation
        original_images = []
        perturbed_images = []
        
        for i in range(num_samples):
            if self.rank == 0:
                logging.info(f"\nProcessing sample {i+1}/{num_samples}")
                if negative_case_type:
                    logging.info(f"Negative case type: {negative_case_type}")
            
            # Generate z vector and image from negative model
            z = torch.randn(1, negative_model.z_dim, device=self.device)
            if hasattr(negative_model, 'module'):
                w = negative_model.module.mapping(z, None)
                negative_img = negative_model.module.synthesis(w, noise_mode="const")
            else:
                w = negative_model.mapping(z, None)
                negative_img = negative_model.synthesis(w, noise_mode="const")
            
            # Apply transformations for special cases
            if negative_case_type and negative_case_type.startswith('downsample'):
                size = int(negative_case_type.split('_')[1])
                negative_img = downsample_and_upsample(negative_img, downsample_size=size)
                if self.rank == 0:
                    logging.info(f"Applied downsampling transformation with size {size}")
            
            # Try to find fooling perturbation using same z vector
            if self.rank == 0:
                logging.info("Starting binary search for perturbation...")
            perturbed, alpha = self.binary_search_perturbation(negative_img, z)
            
            if perturbed is not None:
                successful_attacks += 1
                # Compute quality metrics only for successful attacks
                metrics = self.quality_metrics.compute_metrics(negative_img, perturbed)
                
                # Store images for FID calculation
                original_images.append(negative_img)
                perturbed_images.append(perturbed)
                
                results.append({
                    'alpha': alpha,
                    'metrics': metrics
                })
                
                if self.rank == 0:
                    logging.info(f"Attack successful - Alpha: {alpha:.4f}, Metrics: LPIPS={metrics['lpips']:.4f}, PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
            else:
                if self.rank == 0:
                    logging.info("Attack failed for this sample")
            
            total_queries += self.max_queries
            
            # Log progress
            if (i + 1) % 10 == 0 and self.rank == 0:
                logging.info(f"Progress: {i+1}/{num_samples} samples. Current success rate: {successful_attacks/(i+1):.2%}")
        
        # Calculate FID score for all successful attacks
        if successful_attacks > 0:
            # Concatenate all images
            original_images = torch.cat(original_images, dim=0)
            perturbed_images = torch.cat(perturbed_images, dim=0)
            
            # Convert from [-1, 1] to [0, 1] range for FID calculation
            original_images = (original_images + 1) / 2
            perturbed_images = (perturbed_images + 1) / 2
            
            try:
                if self.rank == 0:
                    logging.info(f"Computing FID score for {len(original_images)} successful attacks...")
                
                fid = calculate_fid(
                    original_images,
                    perturbed_images,
                    batch_size=self.batch_size,
                    device=self.device
                )
                
                if self.rank == 0:
                    logging.info(f"FID score computed: {fid:.4f}")
            except Exception as e:
                if self.rank == 0:
                    logging.error(f"Error computing FID score: {str(e)}")
                fid = float('inf')
        else:
            fid = float('inf')
        
        # Aggregate results
        success_rate = successful_attacks / num_samples
        avg_queries = total_queries / num_samples
        
        # Calculate average metrics for successful attacks
        avg_metrics = {
            'lpips': np.mean([r['metrics']['lpips'] for r in results]) if results else float('inf'),
            'psnr': np.mean([r['metrics']['psnr'] for r in results]) if results else 0,
            'ssim': np.mean([r['metrics']['ssim'] for r in results]) if results else 0,
            'fid': fid
        }
        
        return {
            'success_rate': success_rate,
            'avg_queries': avg_queries,
            'avg_metrics': avg_metrics,
            'results': results
        }
    
    def attack_all_cases(self, pretrained_models, quantized_models, num_samples):
        """Attack all negative cases and return combined results."""
        all_results = {}
        
        # Attack pretrained models
        for model_name, model in pretrained_models.items():
            if self.rank == 0:
                logging.info(f"\nAttacking pretrained model: {model_name}")
            all_results[model_name] = self.attack_negative_case(model, num_samples)
        
        # Attack quantized models
        for precision, model in quantized_models.items():
            case_name = f"quantization_{precision}"
            if self.rank == 0:
                logging.info(f"\nAttacking quantized model: {precision}")
            all_results[case_name] = self.attack_negative_case(model, num_samples, case_name)
        
        # Attack downsample cases
        for size in [128, 224]:
            case_name = f"downsample_{size}"
            if self.rank == 0:
                logging.info(f"\nAttacking downsample case: {size}")
            all_results[case_name] = self.attack_negative_case(self.original_model, num_samples, case_name)
        
        return all_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Query-based attack for StyleGAN2 Watermarking")
    
    # Model configuration
    parser.add_argument("--stylegan2_url", type=str,
                        default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl",
                        help="URL for the pretrained StyleGAN2 model")
    parser.add_argument("--stylegan2_local_path", type=str,
                        default="ffhq70k-paper256-ada.pkl",
                        help="Local path to store/load the StyleGAN2 model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to decoder checkpoint to attack")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image resolution")
    parser.add_argument("--image_pixel_set_seed", type=int, default=42,
                        help="Random seed for selecting pixel indices")
    parser.add_argument("--image_pixel_count", type=int, default=32,
                        help="Number of pixels to select from the image")
    
    # Attack configuration
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to attack")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Maximum perturbation size")
    parser.add_argument("--max_queries", type=int, default=1000,
                        help="Maximum number of queries per sample")
    parser.add_argument("--binary_search_steps", type=int, default=10,
                        help="Number of binary search steps")
    parser.add_argument("--detection_threshold", type=float, default=0.002883,
                        help="MSE threshold for detection (95% TPR threshold)")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="query_based_attack_results",
                        help="Directory to save attack results")
    
    return parser.parse_args()


def format_results_table(all_results):
    """Format results into a nice table using fixed-width columns."""
    # Print table header
    table_str = "\nAttack Results Summary:\n"
    table_str += "-" * 150 + "\n"
    table_str += f"{'Negative Case':<40}{'Success Rate':>15}{'Avg Queries':>15}{'LPIPS':>15}{'PSNR':>15}{'SSIM':>15}{'FID':>15}\n"
    table_str += "-" * 150 + "\n"
    
    # Add rows
    for case_name, results in all_results.items():
        row = f"{case_name:<40}"
        row += f"{results['success_rate']:.2%:>15}"
        row += f"{results['avg_queries']:.1f:>15}"
        row += f"{results['avg_metrics']['lpips']:.4f:>15}"
        row += f"{results['avg_metrics']['psnr']:.2f:>15}"
        row += f"{results['avg_metrics']['ssim']:.4f:>15}"
        row += f"{results['avg_metrics']['fid']:.2f:>15}\n"
        table_str += row
    
    table_str += "-" * 150 + "\n"
    return table_str


def main():
    """Main entry point for attack."""
    args = parse_args()
    
    # Setup distributed environment
    local_rank, rank, world_size, device = setup_distributed()
    
    # Load default configuration and update with args
    config = get_default_config()
    config.update_from_args(args, mode='query_based_attack')
    
    # Create output directory and setup logging
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
        setup_logging(config.output_dir, rank)
        
        # Log configuration
        logging.info(f"Configuration:\n{config}")
        logging.info(f"Distributed setup: local_rank={local_rank}, rank={rank}, world_size={world_size}, device={device}")
    
    try:
        # Load models
        if rank == 0:
            logging.info("Loading models...")
        
        # Load original StyleGAN2 model
        original_model = load_stylegan2_model(
            config.model.stylegan2_url,
            config.model.stylegan2_local_path,
            device
        )
        
        # Load pretrained models
        pretrained_models = load_pretrained_models(device, rank)
        
        # Setup quantized models
        quantized_models = {}
        try:
            if rank == 0:
                logging.info("Setting up quantized models...")
            for precision in ['int8', 'int4']:
                try:
                    quantized_model = quantize_model_weights(original_model, precision)
                    quantized_model.eval()
                    quantized_models[precision] = quantized_model
                    if rank == 0:
                        logging.info(f"Successfully created {precision} model")
                except Exception as e:
                    if rank == 0:
                        logging.error(f"Failed to create {precision} model: {str(e)}")
        except Exception as e:
            if rank == 0:
                logging.error(f"Error in quantization setup: {str(e)}")
        
        # Load decoder and create wrapper
        from models.decoder import Decoder
        from utils.checkpoint import load_checkpoint
        
        decoder = Decoder(
            image_size=config.model.img_size,
            channels=3,
            output_dim=config.model.image_pixel_count
        ).to(device)
        
        load_checkpoint(
            checkpoint_path=config.checkpoint_path,
            decoder=decoder,
            device=device
        )
        decoder.eval()
        
        # Generate pixel indices (same as in evaluator)
        torch.manual_seed(config.model.image_pixel_set_seed)
        total_pixels = config.model.img_size * config.model.img_size * 3
        image_pixel_indices = torch.randperm(total_pixels)[:config.model.image_pixel_count].to(device)
        
        if rank == 0:
            logging.info(f"Generated {len(image_pixel_indices)} pixel indices with seed {config.model.image_pixel_set_seed}")
            logging.info(f"Selected pixel indices: {image_pixel_indices.tolist()}")
        
        # Create decoder wrapper with threshold from config
        decoder_wrapper = DecoderWrapper(
            decoder=decoder,
            threshold=config.query_based_attack.detection_threshold,
            image_pixel_indices=image_pixel_indices
        )
        
        # Create attacker
        attacker = QueryBasedAttack(
            original_model=original_model,
            decoder=decoder_wrapper,
            epsilon=config.query_based_attack.epsilon,
            max_queries=config.query_based_attack.max_queries,
            binary_search_steps=config.query_based_attack.binary_search_steps,
            device=device,
            rank=rank,
            batch_size=config.query_based_attack.batch_size
        )
        
        # Run attack on all cases
        if rank == 0:
            logging.info("Starting attacks on all negative cases...")
        
        all_results = attacker.attack_all_cases(
            pretrained_models=pretrained_models,
            quantized_models=quantized_models,
            num_samples=config.query_based_attack.num_samples
        )
        
        # Print results table
        if rank == 0:
            table = format_results_table(all_results)
            logging.info(table)
    
    except Exception as e:
        if rank == 0:
            logging.error(f"Error in attack: {str(e)}")
            logging.error("Attack error:", exc_info=True)
        raise
    finally:
        # Clean up distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main() 