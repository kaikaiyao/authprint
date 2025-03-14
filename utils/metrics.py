"""
Metrics utilities for StyleGAN watermarking evaluation.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as skmetrics


def calculate_metrics(
    all_watermarked_mse_distances, 
    all_original_mse_distances,
    all_watermarked_mae_distances, 
    all_original_mae_distances,
    watermarked_correct,
    original_correct,
    total_samples,
    all_lpips_losses
):
    """
    Calculate evaluation metrics.
    
    Args:
        all_watermarked_mse_distances (list): MSE distances for watermarked images
        all_original_mse_distances (list): MSE distances for original images
        all_watermarked_mae_distances (list): MAE distances for watermarked images
        all_original_mae_distances (list): MAE distances for original images
        watermarked_correct (int): Number of correctly matched watermarked keys
        original_correct (int): Number of correctly matched original keys
        total_samples (int): Total number of samples
        all_lpips_losses (list): LPIPS losses between original and watermarked images
        
    Returns:
        dict: Dictionary of metrics
    """
    # Calculate match rates
    watermarked_match_rate = (watermarked_correct / total_samples) * 100
    original_match_rate = (original_correct / total_samples) * 100
    
    # Calculate distance statistics
    watermarked_mse_distance_avg = np.mean(all_watermarked_mse_distances)
    watermarked_mse_distance_std = np.std(all_watermarked_mse_distances)
    watermarked_mae_distance_avg = np.mean(all_watermarked_mae_distances)
    watermarked_mae_distance_std = np.std(all_watermarked_mae_distances)
    
    original_mse_distance_avg = np.mean(all_original_mse_distances)
    original_mse_distance_std = np.std(all_original_mse_distances)
    original_mae_distance_avg = np.mean(all_original_mae_distances)
    original_mae_distance_std = np.std(all_original_mae_distances)
    
    # Calculate LPIPS statistics
    lpips_loss_avg = np.mean(all_lpips_losses)
    lpips_loss_std = np.std(all_lpips_losses)
    
    # Calculate ROC-AUC scores
    # For watermarked vs original detection based on distance:
    # Lower distance is better for watermarked images, so we need to negate the distances
    # 1 = watermarked (positive class), 0 = original (negative class)
    y_true = np.concatenate([
        np.ones(len(all_watermarked_mse_distances)), 
        np.zeros(len(all_original_mse_distances))
    ])
    
    # Negate distances since lower is better for watermarked images
    # (ROC-AUC expects higher scores for positive class)
    y_score_mse = np.concatenate([
        -np.array(all_watermarked_mse_distances), 
        -np.array(all_original_mse_distances)
    ])
    
    y_score_mae = np.concatenate([
        -np.array(all_watermarked_mae_distances), 
        -np.array(all_original_mae_distances)
    ])
    
    # Calculate ROC-AUC
    roc_auc_mse = skmetrics.roc_auc_score(y_true, y_score_mse)
    roc_auc_mae = skmetrics.roc_auc_score(y_true, y_score_mae)
    
    # Use MSE as the primary ROC-AUC metric
    roc_auc_score = roc_auc_mse
    
    # Create metrics dictionary
    metrics = {
        'watermarked_match_rate': watermarked_match_rate,
        'original_match_rate': original_match_rate,
        'watermarked_lpips_loss_avg': lpips_loss_avg,
        'watermarked_lpips_loss_std': lpips_loss_std,
        'watermarked_mse_distance_avg': watermarked_mse_distance_avg,
        'watermarked_mse_distance_std': watermarked_mse_distance_std,
        'watermarked_mae_distance_avg': watermarked_mae_distance_avg,
        'watermarked_mae_distance_std': watermarked_mae_distance_std,
        'original_mse_distance_avg': original_mse_distance_avg,
        'original_mse_distance_std': original_mse_distance_std,
        'original_mae_distance_avg': original_mae_distance_avg,
        'original_mae_distance_std': original_mae_distance_std,
        'roc_auc_score_mse': roc_auc_mse,
        'roc_auc_score_mae': roc_auc_mae,
        'roc_auc_score': roc_auc_score,
        'num_samples_processed': total_samples
    }
    
    return metrics, (y_true, y_score_mse, y_score_mae)


def save_metrics_plots(
    metrics, 
    y_data,
    all_watermarked_mse_distances,
    all_original_mse_distances,
    all_watermarked_mae_distances,
    all_original_mae_distances,
    output_dir
):
    """
    Save metrics plots.
    
    Args:
        metrics (dict): Dictionary of metrics
        y_data (tuple): y_true, y_score_mse, y_score_mae for ROC curves
        all_watermarked_mse_distances (list): MSE distances for watermarked images
        all_original_mse_distances (list): MSE distances for original images
        all_watermarked_mae_distances (list): MAE distances for watermarked images
        all_original_mae_distances (list): MAE distances for original images
        output_dir (str): Output directory
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Unpack y_data
    y_true, y_score_mse, y_score_mae = y_data
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    # ROC curve for MSE distance
    fpr_mse, tpr_mse, _ = skmetrics.roc_curve(y_true, y_score_mse)
    plt.plot(fpr_mse, tpr_mse, label=f'MSE Distance (AUC = {metrics["roc_auc_score_mse"]:.4f})')
    
    # ROC curve for MAE distance
    fpr_mae, tpr_mae, _ = skmetrics.roc_curve(y_true, y_score_mae)
    plt.plot(fpr_mae, tpr_mae, label=f'MAE Distance (AUC = {metrics["roc_auc_score_mae"]:.4f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Watermark Detection')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    
    # Plot histograms of distances
    plt.figure(figsize=(12, 10))
    
    # MSE distance histogram
    plt.subplot(2, 1, 1)
    plt.hist(all_watermarked_mse_distances, bins=50, alpha=0.5, label='Watermarked')
    plt.hist(all_original_mse_distances, bins=50, alpha=0.5, label='Original')
    plt.xlabel('MSE Distance')
    plt.ylabel('Count')
    plt.title('Histogram of MSE Distances')
    plt.legend()
    plt.grid(True)
    
    # MAE distance histogram
    plt.subplot(2, 1, 2)
    plt.hist(all_watermarked_mae_distances, bins=50, alpha=0.5, label='Watermarked')
    plt.hist(all_original_mae_distances, bins=50, alpha=0.5, label='Original')
    plt.xlabel('MAE Distance')
    plt.ylabel('Count')
    plt.title('Histogram of MAE Distances')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'distance_histograms.png'), dpi=300)
    plt.close()


def save_metrics_text(metrics, output_dir):
    """
    Save metrics to a text file.
    
    Args:
        metrics (dict): Dictionary of metrics
        output_dir (str): Output directory
    """
    metrics_path = os.path.join(output_dir, "evaluation_metrics.txt")
    with open(metrics_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n") 