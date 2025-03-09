import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
from pathlib import Path
import seaborn as sns
from matplotlib.ticker import MaxNLocator

def parse_log_file(filename):
    losses = []
    with open(filename, 'r') as f:
        for line in f:
            if 'Train Iteration' in line:
                # Extract loss value using regex
                match = re.search(r'loss: (\d+\.\d+)', line)
                if match:
                    losses.append(float(match.group(1)))
    return np.array(losses)

def smooth_losses(losses, window_size=100):
    # If we have fewer points than window_size, adjust window_size
    if len(losses) < window_size:
        window_size = max(1, len(losses) // 10)
    
    # Pad the array to handle edge cases
    pad_size = window_size - (len(losses) % window_size)
    if pad_size < window_size:
        padded_losses = np.pad(losses, (0, pad_size), mode='edge')
    else:
        padded_losses = losses
    
    # Reshape and compute means
    n_windows = len(padded_losses) // window_size
    reshaped = padded_losses[:n_windows * window_size].reshape(-1, window_size)
    return np.mean(reshaped, axis=1)

def plot_loss(filename, output_path='loss_plot.pdf'):
    # Set up the matplotlib parameters for conference-quality plots
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'Roboto', 'Open Sans'],
        'font.size': 9,
        'text.usetex': False,  # Set to True if you have LaTeX installed
        'axes.linewidth': 0.8,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })
    
    # Set a clean, modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.2})
    
    # ACM-style figure size (single column width is about 3.33 inches)
    # For a single-column figure
    fig_width = 3.33  # inches
    # Golden ratio for aesthetically pleasing height-to-width ratio
    golden_ratio = (5**0.5 - 1) / 2  # Approximately 0.618
    fig_height = fig_width * golden_ratio  # inches
    
    # Create figure with conference-appropriate size
    plt.figure(figsize=(fig_width, fig_height))
    
    # Parse and smooth the losses
    losses = parse_log_file(filename)
    
    if len(losses) == 0:
        print(f"Error: No loss values found in {filename}")
        return
        
    # Normalize x-axis from 0 to 1 (training progress)
    x = np.linspace(0, 1, len(losses))
    
    # Plot raw data with low opacity and much smaller points
    plt.plot(x, losses, '.', color='#4292c6', alpha=0.08, markersize=0.5, label='Raw Loss')
    
    # Compute smoothed losses with adaptive window size
    window_size = min(100, max(5, len(losses) // 20))  # Adaptive window size
    smoothed_losses = []
    
    # Moving average for smoothing
    for i in range(len(losses)):
        start = max(0, i - window_size // 2)
        end = min(len(losses), i + window_size // 2)
        smoothed_losses.append(np.mean(losses[start:end]))
    
    # Plot smoothed line with thinner linewidth
    plt.plot(x, smoothed_losses, '-', color='#08519c', linewidth=0.8, label='Smoothed Loss')
    
    # Set labels with proper formatting - use normal weight with sans-serif fonts
    plt.xlabel('Training Progress', fontweight='normal')
    plt.ylabel('Loss', fontweight='normal')
    
    # Set x-axis ticks to only show progress from 0 to 1
    plt.xticks([0, 0.25, 0.5, 0.75, 1], ['0', '0.25', '0.5', '0.75', '1'])
    
    # Customize ticks for y-axis
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(5))
    
    # Customize spines for a cleaner look with rounded fonts
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    
    # Add legend with custom position - small adjustment for sans-serif
    plt.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='none')
    
    # Get min and max for annotations
    min_idx = np.argmin(smoothed_losses)
    min_loss = smoothed_losses[min_idx]
    min_x = x[min_idx]
    
    # Optional: Annotate minimum loss with smaller font
    plt.annotate(f'Min: {min_loss:.4f}', 
                xy=(min_x, min_loss),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=7.5,  # Slightly larger for sans-serif readability
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', 
                               color='#08519c', alpha=0.7, linewidth=0.6))
    
    # Ensure the plot fits nicely in the figure
    plt.tight_layout()
    
    # Save the plot in multiple formats for different use cases
    plt.savefig(output_path, bbox_inches='tight')
    
    # Also save as PNG for easy viewing and PPT inclusion
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    # Save high-resolution version for poster printing
    hires_path = output_path.replace('.pdf', '_hires.png')
    plt.savefig(hires_path, dpi=600, bbox_inches='tight')
    
    plt.close()
    
    print(f"Plots saved as:")
    print(f"- {output_path} (for publication)")
    print(f"- {png_path} (for presentations)")
    print(f"- {hires_path} (high resolution)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "loss1.txt"  # default filename
    
    output_path = 'loss_plot.pdf'
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
        
    plot_loss(log_file, output_path) 