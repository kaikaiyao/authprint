import torch
import numpy as np
import math
import logging
import statistics
import matplotlib.pyplot as plt
import lpips
from typing import Tuple, Dict, List
from collections import defaultdict

from utils.image_utils import (
    constrain_image, downsample_and_upsample, 
    apply_jpeg_compression, quantize_model_weights,
    apply_truncation
)
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision.transforms.functional import to_pil_image
from models.stylegan2 import is_stylegan2, load_stylegan2_model
from torchmetrics.image.fid import FrechetInceptionDistance
from key.key import generate_mask_secret_key, mask_image_with_key
from utils.logging import LogRankFilter


def evaluate_model(
    num_images: int,
    gan_model: torch.nn.Module,
    watermarked_model: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    plotting: bool,
    latent_dim: int,
    max_delta: float,
    mask_switch_on: bool,
    seed_key: int,
    flip_key_type: str,
    batch_size: int = 8,
    key_type: str = "csprng",
    compute_fid: bool = False,
    rank: int = 0
) -> Dict:
    """
    Evaluate watermarking model performance with comprehensive metrics across multiple test cases.
    Only runs on rank 0 for distributed setups.
    
    Args:
        ...
        rank: Process rank in distributed setup. Function only runs on rank 0.
    
    Returns:
        Dictionary containing metrics for all test cases (on rank 0) or empty dict (on other ranks)
    """
    # Configure logging to filter based on rank
    root_logger = logging.getLogger()
    
    # First remove existing rank filters if any
    for handler in root_logger.handlers:
        for filter in handler.filters[:]:
            if isinstance(filter, LogRankFilter):
                handler.removeFilter(filter)
        
        # Add our rank filter to each handler
        handler.addFilter(LogRankFilter(rank))
    
    # Only run evaluation on rank 0
    if rank != 0:
        return {}

    # Initialize models and metrics
    gan_model.eval()
    watermarked_model.eval()
    decoder.eval()
    
    total_decoder_params = sum(p.numel() for p in decoder.parameters())
    lpips_loss = lpips.LPIPS(net="vgg").to(device)
    
    # Only initialize FID metric if needed
    fid_metric = FrechetInceptionDistance().to(device) if compute_fid else None

    # Load alternative pre-trained models - only on rank 0
    alternative_models = {}
    if rank == 0:
        model_configs = {
            'ffhq1k': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq1k-paper256-ada.pkl",
                      "ffhq1k-paper256-ada.pkl"),
            'ffhq30k': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq30k-paper256-ada.pkl",
                       "ffhq30k-paper256-ada.pkl"),
            'ffhq70k-bcr': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada-bcr.pkl",
                           "ffhq70k-paper256-ada-bcr.pkl"),
            'ffhq70k-noaug': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-noaug.pkl",
                             "ffhq70k-paper256-noaug.pkl")
        }
        
        for model_name, (url, local_path) in model_configs.items():
            try:
                logging.info(f"Loading alternative model: {model_name}")
                alternative_models[model_name] = load_stylegan2_model(url=url, local_path=local_path, device=device)
            except Exception as e:
                logging.warning(f"Failed to load {model_name}: {str(e)}")
                continue

    # Data collection structures for each test case
    test_cases = {
        'watermarked': {'scores': [], 'images': []},
        'original': {'scores': [], 'images': []},
        'random': {'scores': [], 'images': []},
        'truncated': {'scores': [], 'images': []},
        'quantized': {'scores': [], 'images': []},
        'downsampled': {'scores': [], 'images': []},
        'compressed': {'scores': [], 'images': []}
    }
    
    # Add alternative model test cases
    for model_name in alternative_models.keys():
        test_cases[f'alt_{model_name}'] = {'scores': [], 'images': []}

    # Process images in batches
    num_batches = math.ceil(num_images / batch_size)
    for batch_idx in range(num_batches):
        process_comprehensive_batch(
            batch_idx=batch_idx,
            num_batches=num_batches,
            gan_model=gan_model,
            watermarked_model=watermarked_model,
            decoder=decoder,
            device=device,
            latent_dim=latent_dim,
            max_delta=max_delta,
            mask_switch_on=mask_switch_on,
            seed_key=seed_key,
            batch_size=batch_size,
            lpips_loss=lpips_loss,
            fid_metric=fid_metric,
            test_cases=test_cases,
            alternative_models=alternative_models,
            plotting=plotting,
            flip_key_type=flip_key_type,
            num_images=num_images,
            key_type=key_type,
            compute_fid=compute_fid
        )

    # Calculate final metrics
    results = calculate_comprehensive_metrics(test_cases, fid_metric, compute_fid)
    
    if plotting:
        generate_comprehensive_plots(test_cases, results)
    
    return results


def process_comprehensive_batch(
    batch_idx: int,
    num_batches: int,
    gan_model: torch.nn.Module,
    watermarked_model: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    latent_dim: int,
    max_delta: float,
    mask_switch_on: bool,
    seed_key: int,
    batch_size: int,
    lpips_loss: torch.nn.Module,
    fid_metric: FrechetInceptionDistance,
    test_cases: Dict,
    alternative_models: Dict,
    plotting: bool,
    flip_key_type: str,
    num_images: int,
    key_type: str = "csprng",
    compute_fid: bool = False
) -> None:
    """Process a batch for all test cases"""
    current_batch_size = min(batch_size, num_images - batch_idx * batch_size)
    logging.info(f"Processing batch {batch_idx+1}/{num_batches} ({current_batch_size} images)")

    # Generate base images
    with torch.no_grad():
        z = torch.randn((current_batch_size, latent_dim), device=device)
        x_M = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
        x_M_hat = watermarked_model(z, None, truncation_psi=1.0, noise_mode="const")
        x_M_hat = constrain_image(x_M_hat, x_M, max_delta)
        
        # Generate images for all test cases
        x_rand = torch.rand_like(x_M) * 2 - 1
        x_truncated = apply_truncation(
            model=gan_model,
            z=z,
            truncation_psi=0.5
        )
        x_quantized = quantize_model_weights(
            model=gan_model,
            precision='int8'
        )(z, None, truncation_psi=1.0, noise_mode="const")
        x_downsampled = downsample_and_upsample(
            images=x_M,
            downsample_size=128
        )
        x_compressed = apply_jpeg_compression(
            images=x_M,
            quality=55
        )
        
        # Generate images from alternative models
        alt_images = {
            name: model(z, None, truncation_psi=1.0, noise_mode="const")
            for name, model in alternative_models.items()
        }

    # Process watermark detection for all cases
    process_detection_for_case(
        images=x_M,
        decoder=decoder,
        mask_switch_on=mask_switch_on,
        seed_key=seed_key,
        device=device,
        case_data=test_cases['original'],
        batch_size=current_batch_size,
        flip_key_type=flip_key_type,
        key_type=key_type
    )
    process_detection_for_case(
        images=x_M_hat,
        decoder=decoder,
        mask_switch_on=mask_switch_on,
        seed_key=seed_key,
        device=device,
        case_data=test_cases['watermarked'],
        batch_size=current_batch_size,
        flip_key_type=flip_key_type,
        key_type=key_type
    )
    process_detection_for_case(
        images=x_rand,
        decoder=decoder,
        mask_switch_on=mask_switch_on,
        seed_key=seed_key,
        device=device,
        case_data=test_cases['random'],
        batch_size=current_batch_size,
        flip_key_type=flip_key_type,
        key_type=key_type
    )
    process_detection_for_case(
        images=x_truncated,
        decoder=decoder,
        mask_switch_on=mask_switch_on,
        seed_key=seed_key,
        device=device,
        case_data=test_cases['truncated'],
        batch_size=current_batch_size,
        flip_key_type=flip_key_type,
        key_type=key_type
    )
    process_detection_for_case(
        images=x_quantized,
        decoder=decoder,
        mask_switch_on=mask_switch_on,
        seed_key=seed_key,
        device=device,
        case_data=test_cases['quantized'],
        batch_size=current_batch_size,
        flip_key_type=flip_key_type,
        key_type=key_type
    )
    process_detection_for_case(
        images=x_downsampled,
        decoder=decoder,
        mask_switch_on=mask_switch_on,
        seed_key=seed_key,
        device=device,
        case_data=test_cases['downsampled'],
        batch_size=current_batch_size,
        flip_key_type=flip_key_type,
        key_type=key_type
    )
    process_detection_for_case(
        images=x_compressed,
        decoder=decoder,
        mask_switch_on=mask_switch_on,
        seed_key=seed_key,
        device=device,
        case_data=test_cases['compressed'],
        batch_size=current_batch_size,
        flip_key_type=flip_key_type,
        key_type=key_type
    )
    
    # Process alternative models
    for name, images in alt_images.items():
        process_detection_for_case(
            images=images,
            decoder=decoder,
            mask_switch_on=mask_switch_on,
            seed_key=seed_key,
            device=device,
            case_data=test_cases[f'alt_{name}'],
            batch_size=current_batch_size,
            flip_key_type=flip_key_type,
            key_type=key_type
        )

    # Update FID metric for original vs watermarked
    if compute_fid and fid_metric is not None:
        update_fid_metric(
            x_M=x_M,
            x_M_hat=x_M_hat,
            fid_metric=fid_metric
        )

    # Store images for plotting if needed
    if plotting:
        store_comprehensive_plot_data(
            test_cases=test_cases,
            current_batch_size=current_batch_size,
            x_M=x_M,
            x_M_hat=x_M_hat,
            x_rand=x_rand,
            x_truncated=x_truncated,
            x_quantized=x_quantized,
            x_downsampled=x_downsampled,
            x_compressed=x_compressed,
            alt_images=alt_images
        )


def process_detection_for_case(images, decoder, mask_switch_on, seed_key, device, case_data, batch_size, flip_key_type, key_type="csprng"):
    """Process watermark detection for a specific test case"""
    if mask_switch_on:
        k_mask = generate_mask_secret_key(images.shape, seed_key, device=device, flip_key_type=flip_key_type, key_type=key_type)
        images = mask_image_with_key(images=images, cnn_key=k_mask)

    with torch.no_grad():
        k = decoder(images)
        scores = torch.norm(k, dim=1)
        
    case_data['scores'].extend(scores.cpu().tolist())
    case_data['images'].append(images.cpu())


def calculate_comprehensive_metrics(test_cases: Dict, fid_metric: FrechetInceptionDistance, compute_fid: bool = False) -> Dict:
    """Calculate comprehensive metrics for all test cases"""
    results = {}
    
    # Calculate threshold at 1% FPR using original vs watermarked
    original_scores = np.array(test_cases['original']['scores'])
    watermarked_scores = np.array(test_cases['watermarked']['scores'])
    
    # Create labels and scores arrays with matching lengths
    labels = np.array([0] * len(original_scores) + [1] * len(watermarked_scores))
    combined_scores = np.concatenate([original_scores, watermarked_scores])
    
    fpr, tpr, thresholds = roc_curve(labels, combined_scores)
    
    # Handle perfect separation case and find appropriate threshold
    if np.all(fpr[:-1] == 0):  # Perfect separation case (all FPR=0 except last point)
        # Set threshold just slightly below the lowest watermarked score
        watermarked_min = min(watermarked_scores)
        epsilon = 1e-6  # Small buffer to ensure we catch all watermarked images
        threshold = watermarked_min - epsilon
        logging.info("Perfect separation detected! Setting threshold just below lowest watermarked score")
    else:
        # Find threshold at 1% FPR - we want the lowest threshold that gives us ≤1% FPR
        # This ensures we catch all high-confidence watermarked images while maintaining FPR
        target_fpr = 0.01
        threshold_candidates = [(fpr[i], thresholds[i]) for i in range(len(fpr)) if fpr[i] <= target_fpr]
        if threshold_candidates:
            threshold = min(threshold_candidates, key=lambda x: x[1])[1]  # Use minimum valid threshold
        else:
            # If no threshold gives exactly 1% FPR, take the one closest to it
            threshold_idx = np.argmin(np.abs(fpr - target_fpr))
            threshold = thresholds[threshold_idx]
    
    logging.info(f"Selected threshold: {threshold:.10f}")
    logging.info(f"Number of original images: {len(original_scores)}")
    logging.info(f"Number of watermarked images: {len(watermarked_scores)}")
    
    # Calculate metrics for each test case
    for case_name, case_data in test_cases.items():
        scores = np.array(case_data['scores'])
        
        if case_name == 'watermarked':
            # Calculate TPR@1%FPR for watermarked case
            # Higher scores indicate watermarked images
            tpr_at_threshold = np.mean(scores >= threshold)
            results[case_name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'tpr_at_1_fpr': tpr_at_threshold
            }
        else:
            # Calculate TNR@1%FPR for other cases
            # Lower scores indicate non-watermarked images
            tnr_at_threshold = np.mean(scores < threshold)
            results[case_name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'tnr_at_1_fpr': tnr_at_threshold
            }
    
    # Calculate original metrics
    if compute_fid and fid_metric is not None:
        results['fid_score'] = fid_metric.compute().item()
    else:
        results['fid_score'] = None
        
    # Calculate AUC using the same arrays used for ROC curve
    results['auc'] = roc_auc_score(labels, combined_scores)
    
    return results


def generate_comprehensive_plots(test_cases: Dict, results: Dict) -> None:
    """Generate comprehensive plots for all test cases"""
    if not test_cases['watermarked']['images']:
        return

    logging.info("Generating comprehensive evaluation plots...")
    
    # Plot score distributions
    plt.figure(figsize=(15, 8))
    for case_name, case_data in test_cases.items():
        plt.hist(case_data['scores'], bins=50, alpha=0.5, label=case_name)
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.title("Score Distributions Across Test Cases")
    plt.legend()
    plt.savefig("score_distributions.png")
    plt.close()

    # Plot example images from each case
    num_cases = len(test_cases)
    fig, axs = plt.subplots(num_cases, 3, figsize=(15, 5*num_cases))
    
    for idx, (case_name, case_data) in enumerate(test_cases.items()):
        if case_data['images']:
            # Get first image from first batch
            img = case_data['images'][0][0]
            score = case_data['scores'][0]
            
            # Convert to display format
            img_display = ((img + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            
            axs[idx, 0].imshow(to_pil_image(img_display))
            axs[idx, 0].set_title(f"{case_name}\nScore: {score:.3f}")
            
            # Add metric values
            metric_name = 'tpr_at_1_fpr' if case_name == 'watermarked' else 'tnr_at_1_fpr'
            metric_value = results[case_name][metric_name]
            axs[idx, 1].text(0.5, 0.5, f"{metric_name}: {metric_value:.3f}", ha='center')
            axs[idx, 1].axis('off')
            
            # Add score statistics
            stats_text = f"Mean: {results[case_name]['mean_score']:.3f}\nStd: {results[case_name]['std_score']:.3f}"
            axs[idx, 2].text(0.5, 0.5, stats_text, ha='center')
            axs[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("comprehensive_evaluation.png")
    plt.close()

def store_comprehensive_plot_data(
    test_cases: Dict,
    current_batch_size: int,
    x_M: torch.Tensor,
    x_M_hat: torch.Tensor,
    x_rand: torch.Tensor,
    x_truncated: torch.Tensor,
    x_quantized: torch.Tensor,
    x_downsampled: torch.Tensor,
    x_compressed: torch.Tensor,
    alt_images: Dict[str, torch.Tensor]
) -> None:
    """Store image data for plotting"""
    # Store original and watermarked images
    test_cases['original']['images'].append(x_M.cpu())
    test_cases['watermarked']['images'].append(x_M_hat.cpu())
    test_cases['random']['images'].append(x_rand.cpu())
    test_cases['truncated']['images'].append(x_truncated.cpu())
    test_cases['quantized']['images'].append(x_quantized.cpu())
    test_cases['downsampled']['images'].append(x_downsampled.cpu())
    test_cases['compressed']['images'].append(x_compressed.cpu())
    
    # Store alternative model images
    for name, images in alt_images.items():
        test_cases[f'alt_{name}']['images'].append(images.cpu())

def update_fid_metric(x_M, x_M_hat, fid_metric):
    """Update FID metric with new images"""
    x_M_normalized = ((x_M + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    x_M_hat_normalized = ((x_M_hat + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    fid_metric.update(x_M_normalized, real=True)
    fid_metric.update(x_M_hat_normalized, real=False)