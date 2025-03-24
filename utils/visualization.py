"""
Visualization utilities for StyleGAN watermarking.
"""
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


def save_image_grid(images, filename, nrow=None, scale=True):
    """
    Save batch of images as a grid in a single file.
    
    Args:
        images (torch.Tensor): Batch of images [B, C, H, W]
        filename (str): Output filename
        nrow (int): Number of images per row
        scale (bool): Whether to scale pixel values to [0, 255]
                     If True: assumes input is in [-1,1] range
                     If False: assumes input is in [0,1] range
    """
    if nrow is None:
        nrow = int(np.sqrt(images.shape[0]))
    
    # Convert to numpy and transpose from [B, C, H, W] to [B, H, W, C]
    images_np = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    # Scale to [0, 255] if needed
    if scale:
        # Scale from [-1,1] to [0,255]
        images_np = ((images_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
    else:
        # Scale from [0,1] to [0,255]
        images_np = (images_np * 255).clip(0, 255).astype(np.uint8)
    
    # Create grid
    h, w = images_np.shape[1:3]
    grid_h = images_np.shape[0] // nrow if images_np.shape[0] % nrow == 0 else images_np.shape[0] // nrow + 1
    grid_w = nrow
    
    grid = np.zeros((grid_h * h, grid_w * w, 3), dtype=np.uint8)
    
    for idx, img in enumerate(images_np):
        i = idx // nrow
        j = idx % nrow
        grid[i * h:(i + 1) * h, j * w:(j + 1) * w] = img
    
    Image.fromarray(grid).save(filename)


def save_visualization(image, true_key, pred_key, pred_probs, output_path, title, match_status):
    """
    Save a single image with watermark key visualization.
    
    Args:
        image (numpy.ndarray): Image array in [H, W, C] format, values in [-1, 1]
        true_key (numpy.ndarray): True binary key
        pred_key (numpy.ndarray): Predicted binary key
        pred_probs (numpy.ndarray): Predicted probabilities
        output_path (str): Path to save the visualization
        title (str): Title for the visualization
        match_status (bool): Whether predicted key matches true key
    """
    # Create figure with two subplots: image and key visualization
    fig = plt.figure(figsize=(12, 6))
    
    # Plot image
    ax1 = plt.subplot(1, 2, 1)
    # Convert from [-1, 1] to [0, 255]
    img_display = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)
    ax1.imshow(img_display)
    ax1.axis('off')
    ax1.set_title('Generated Image')
    
    # Plot key visualization
    ax2 = plt.subplot(1, 2, 2)
    key_len = len(true_key)
    
    # Create bar positions
    positions = np.arange(key_len)
    bar_width = 0.35
    
    # Plot true key bars
    ax2.bar(positions - bar_width/2, true_key, bar_width, 
            label='True Key', color='blue', alpha=0.5)
    
    # Plot predicted probabilities and binary predictions
    ax2.bar(positions + bar_width/2, pred_probs, bar_width,
            label='Pred Probs', color='red', alpha=0.5)
    
    # Add markers for binary predictions
    ax2.scatter(positions + bar_width/2, pred_key, 
               color='darkred', marker='o', s=50,
               label='Binary Pred', zorder=3)
    
    # Add match status and predicted key
    match_color = 'green' if match_status else 'red'
    match_text = 'MATCH' if match_status else 'NO MATCH'
    plt.text(0.5, 1.15, f'Key Status: {match_text}', 
             horizontalalignment='center',
             transform=ax2.transAxes,
             color=match_color,
             fontsize=12,
             fontweight='bold')
    
    # Customize the plot
    ax2.set_xticks(positions)
    ax2.set_xticklabels([str(i) for i in range(key_len)])
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_xlabel('Key Bit Index')
    ax2.set_ylabel('Bit Value / Probability')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_title('Watermark Key Visualization')
    
    # Add overall title
    plt.suptitle(title, fontsize=14, y=1.05)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_comparison_visualization(orig_images, watermarked_images, diff_images, output_dir, prefix="sample"):
    """
    Save original, watermarked, and difference images.
    
    Args:
        orig_images (torch.Tensor): Original images
        watermarked_images (torch.Tensor): Watermarked images
        diff_images (torch.Tensor): Difference images
        output_dir (str): Output directory
        prefix (str): Prefix for filenames
    """
    num_samples = orig_images.shape[0]
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save individual images
    for i in range(num_samples):
        # Save original image
        orig_path = os.path.join(vis_dir, f"{prefix}_{i}_original.png")
        orig_img = ((orig_images[i].detach().cpu() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        orig_img = orig_img.permute(1, 2, 0).numpy()
        Image.fromarray(orig_img).save(orig_path)
        
        # Save watermarked image
        watermarked_path = os.path.join(vis_dir, f"{prefix}_{i}_watermarked.png")
        watermarked_img = ((watermarked_images[i].detach().cpu() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        watermarked_img = watermarked_img.permute(1, 2, 0).numpy()
        Image.fromarray(watermarked_img).save(watermarked_path)
        
        # Save difference image with proper scaling for visibility
        diff_path = os.path.join(vis_dir, f"{prefix}_{i}_difference.png")
        # Scale difference for better visibility (amplify the subtle differences)
        diff_img = diff_images[i].detach().cpu()
        # Normalize and scale to full range for better visualization
        diff_min = diff_img.min()
        diff_max = diff_img.max()
        if diff_max > diff_min:  # Avoid division by zero
            diff_img = (diff_img - diff_min) / (diff_max - diff_min)
        diff_img = (diff_img * 255).clamp(0, 255).to(torch.uint8)
        diff_img = diff_img.permute(1, 2, 0).numpy()
        Image.fromarray(diff_img).save(diff_path)
    
    # Save grid images
    save_image_grid(orig_images, os.path.join(vis_dir, f"{prefix}_all_original.png"))
    save_image_grid(watermarked_images, os.path.join(vis_dir, f"{prefix}_all_watermarked.png"))
    
    # For difference grid, rescale each difference image individually for better visibility
    norm_diff_images = []
    for i in range(diff_images.shape[0]):
        diff_img = diff_images[i]
        diff_min = diff_img.min()
        diff_max = diff_img.max()
        if diff_max > diff_min:  # Avoid division by zero
            diff_img = (diff_img - diff_min) / (diff_max - diff_min)
        norm_diff_images.append(diff_img)
    norm_diff_images = torch.stack(norm_diff_images)
    save_image_grid(norm_diff_images, os.path.join(vis_dir, f"{prefix}_all_differences.png"), scale=False) 