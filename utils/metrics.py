"""
Metrics utilities for StyleGAN watermarking evaluation.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as skmetrics
import logging
import seaborn as sns


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


def save_metrics_plots(metrics, y_data, watermarked_mse_distances, original_mse_distances, 
                   watermarked_mae_distances=None, original_mae_distances=None, output_dir=None):
    """Save metrics plots to output directory."""
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)

    try:
        from sklearn.metrics import roc_curve, auc
        
        plt.style.use('seaborn-v0_8-darkgrid')

        # Unpack y_data
        if len(y_data) == 3:
            y_true, y_score_mse, y_score_mae = y_data
        else:
            y_true, y_score_mse = y_data
            y_score_mae = None
        
        # Create ROC curve figure
        plt.figure(figsize=(10, 8))
        
        # Calculate ROC curve for MSE
        fpr_mse, tpr_mse, _ = roc_curve(y_true, y_score_mse)
        roc_auc_mse = auc(fpr_mse, tpr_mse)
        
        # Plot MSE ROC curve
        plt.plot(fpr_mse, tpr_mse, color='blue', lw=2, 
                label=f'MSE (AUC = {roc_auc_mse:.4f})')
        
        # Calculate and plot MAE ROC curve if available
        if y_score_mae is not None and watermarked_mae_distances is not None and original_mae_distances is not None:
            fpr_mae, tpr_mae, _ = roc_curve(y_true, y_score_mae)
            roc_auc_mae = auc(fpr_mae, tpr_mae)
            plt.plot(fpr_mae, tpr_mae, color='green', lw=2, 
                    label=f'MAE (AUC = {roc_auc_mae:.4f})')
        
        # Plot random classifier
        plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--',
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create distance distribution plot
        plt.figure(figsize=(12, 10))
        
        # MSE distance distribution
        plt.subplot(2, 1, 1)
        sns.histplot(watermarked_mse_distances, bins=50, alpha=0.5, color='blue', label='Watermarked')
        sns.histplot(original_mse_distances, bins=50, alpha=0.5, color='red', label='Original')
        plt.title('MSE Distance Distribution')
        plt.xlabel('MSE Distance')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        
        # MAE distance distribution if available
        if watermarked_mae_distances is not None and original_mae_distances is not None:
            plt.subplot(2, 1, 2)
            sns.histplot(watermarked_mae_distances, bins=50, alpha=0.5, color='blue', label='Watermarked')
            sns.histplot(original_mae_distances, bins=50, alpha=0.5, color='red', label='Original')
            plt.title('MAE Distance Distribution')
            plt.xlabel('MAE Distance')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distance_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        logging.warning("matplotlib and/or seaborn not available. Skipping plot generation.")
    except Exception as e:
        logging.warning(f"Error generating plots: {str(e)}")


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