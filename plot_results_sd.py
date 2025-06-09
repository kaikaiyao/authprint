import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import font_manager

def setup_plotting_style():
    """Set up the professional plotting style."""
    plt.rcParams['font.sans-serif'] = ['Segoe UI']
    plt.rcParams['font.family'] = 'sans-serif'
    
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 20,           # Slightly smaller for subplot layout
        'axes.labelsize': 22,      # Adjusted for subplot layout
        'axes.titlesize': 24,      # Adjusted for subplot layout
        'legend.fontsize': 18,     # Adjusted for subplot layout
        'xtick.labelsize': 20,     # Adjusted for subplot layout
        'ytick.labelsize': 20,     # Adjusted for subplot layout
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.facecolor': 'none',
        'axes.facecolor': 'white',
        'grid.linewidth': 1.0,
        'axes.linewidth': 2.0,
        'legend.title_fontsize': 20,
        'font.weight': 'normal',
        'axes.labelweight': 'normal',
        'axes.titleweight': 'normal',
        'text.color': 'black',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'grid.color': '#E5E5E5',   # Light gray for grid
        'figure.edgecolor': 'black'
    })

# Data for Stable Diffusion models
sd_methods = [
    "SD 1.5",
    "SD 1.4",
    "SD 1.3",
    "SD 1.2",
    "SD 1.1"
]

# Updated FPR values for each model
sd_fpr_values = [
    [0.3764, 0.0888, 0.0728, 0.1133, 0.1722, 0.1811, 0.1811],  # SD 1.5
    [0.3650, 0.0509, 0.0509, 0.0820, 0.0934, 0.1019, 0.1053],  # SD 1.4
    [0.3640, 0.0870, 0.0817, 0.1328, 0.1527, 0.1527, 0.1563],  # SD 1.3
    [0.4307, 0.0982, 0.1082, 0.1428, 0.1884, 0.1863, 0.1963],  # SD 1.2
    [0.3160, 0.0089, 0.0018, 0.0000, 0.0178, 0.0284, 0.0266]   # SD 1.1
]

def add_small_offset(values, threshold=0.01):
    """Add small offset to very small values to make them visible in plots."""
    return [max(v, threshold) if v > 0 else v for v in values]

def create_plot(ax, methods, fpr_values):
    """Create a plot for Stable Diffusion results."""
    pixels = [4, 64, 256, 1024, 4096, 16384, 65536]
    
    # ML conference standard colors
    colors = [
        '#0077BB',  # Blue
        '#EE7733',  # Orange
        '#009988',  # Teal
        '#CC3311',  # Red
        '#33BBEE'   # Sky Blue
    ]
    
    lines = []  # Store lines for legend
    labels = []  # Store labels for legend
    
    # Plot FPR vs Pixels
    for idx, method in enumerate(methods):
        marker = 'o' if idx % 2 == 0 else 's'
        label = method
        
        y_values = np.array(fpr_values[idx])
        
        # Plot main line
        line = ax.plot(range(len(pixels)), y_values,
                marker=marker, label=label, color=colors[idx],
                alpha=0.9, markersize=10, linestyle='-', linewidth=2.5,
                markeredgecolor='black', markeredgewidth=1)[0]
        
        lines.append(line)
        labels.append(label)
    
    # Customize the plot
    ax.set_xticks(range(len(pixels)))
    ax.set_xticklabels(pixels)
    ax.set_xlabel('Fingerprint Length', labelpad=15)
    ax.set_ylabel('FPR@95%TPR', labelpad=15)
    ax.set_title('Stable Diffusion 2.1', pad=15)
    ax.set_ylim(-0.05, 1.05)  # Consistent y-axis range
    
    # Add legend inside the plot
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0),
             fontsize=12, ncol=1, handletextpad=0.3, 
             handlelength=1.5, markerscale=1.2, edgecolor='black')
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Set spines to black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    return lines, labels

def create_inference_steps_comparison(ax):
    """Create a plot comparing different inference steps."""
    models = ["SD 1.5", "SD 1.4", "SD 1.3", "SD 1.2", "SD 1.1"]
    steps_25 = [0.1133, 0.082, 0.1328, 0.1428, 0.0]      # 25 steps
    steps_15 = [0.0765, 0.0315, 0.0819, 0.1021, 0.0284]  # 15 steps
    steps_5 = [0.0, 0.0039, 0.0039, 0.0078, 0.0]         # 5 steps
    
    x = np.arange(len(models))
    width = 0.25
    
    # ML conference standard colors
    colors = {
        '25_steps': '#0077BB',  # Blue
        '15_steps': '#EE7733',  # Orange
        '5_steps': '#009988'    # Teal
    }
    
    # Plot bars
    ax.bar(x - width, steps_25, width, label='25 steps', color=colors['25_steps'], edgecolor='black', linewidth=1)
    ax.bar(x, steps_15, width, label='15 steps', color=colors['15_steps'], edgecolor='black', linewidth=1)
    ax.bar(x + width, steps_5, width, label='5 steps', color=colors['5_steps'], edgecolor='black', linewidth=1)
    
    ax.set_ylabel('FPR@95%TPR', labelpad=15)
    ax.set_title('Impact of Number of Inference Steps', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(-0.05, 1.05)  # Consistent y-axis range
    ax.legend(edgecolor='black')
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Set spines to black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

def create_prompt_comparison(ax):
    """Create a plot comparing prompt vs no prompt."""
    models = ["SD 1.5", "SD 1.4", "SD 1.3", "SD 1.2", "SD 1.1"]
    with_prompt = [0.1133, 0.082, 0.1328, 0.1428, 0.0]        # Japanese café prompt
    no_prompt = [0.1535, 0.1364, 0.1006, 0.1832, 0.2959]      # generic prompt
    
    x = np.arange(len(models))
    width = 0.35
    
    # ML conference standard colors
    colors = {
        'no_prompt': '#0077BB',    # Blue
        'with_prompt': '#EE7733'   # Orange
    }
    
    # Plot bars
    ax.bar(x - width/2, no_prompt, width, label='Generic: "a high quality photo"', color=colors['no_prompt'], edgecolor='black', linewidth=1)
    ax.bar(x + width/2, with_prompt, width, label='Specific: "Japanese café at neon night"', color=colors['with_prompt'], edgecolor='black', linewidth=1)
    
    ax.set_ylabel('FPR@95%TPR', labelpad=15)
    ax.set_title('Impact of Prompt Types', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(-0.05, 1.05)  # Consistent y-axis range
    ax.legend(edgecolor='black')
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Set spines to black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

def create_training_samples_comparison(ax):
    """Create a plot comparing number of training samples."""
    models = ["SD 1.5", "SD 1.4", "SD 1.3", "SD 1.2", "SD 1.1"]
    samples_128k = [0.3012, 0.1685, 0.2556, 0.2764, 0.1436]  # 128k samples
    samples_320k = [0.1658, 0.1012, 0.1835, 0.1965, 0.0822]  # 320k samples
    samples_512k = [0.1133, 0.082, 0.1328, 0.1428, 0.0]      # 512k samples
    
    x = np.arange(len(models))
    width = 0.25  # Adjusted width to fit three bars
    
    # ML conference standard colors
    colors = {
        '128k_samples': '#0077BB',    # Blue
        '320k_samples': '#EE7733',    # Orange
        '512k_samples': '#009988'     # Teal
    }
    
    # Plot bars
    ax.bar(x - width, samples_128k, width, label='128k samples', color=colors['128k_samples'], edgecolor='black', linewidth=1)
    ax.bar(x, samples_320k, width, label='320k samples', color=colors['320k_samples'], edgecolor='black', linewidth=1)
    ax.bar(x + width, samples_512k, width, label='512k samples', color=colors['512k_samples'], edgecolor='black', linewidth=1)
    
    ax.set_ylabel('FPR@95%TPR', labelpad=15)
    ax.set_title('Impact of Number of Training Samples', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(-0.05, 1.05)  # Consistent y-axis range
    ax.legend(edgecolor='black')
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Set spines to black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

def create_decoder_size_comparison(ax):
    """Create a plot comparing different decoder sizes."""
    models = ["SD 1.5", "SD 1.4", "SD 1.3", "SD 1.2", "SD 1.1"]
    small_decoder = [0.9576, 0.9488, 0.9887, 0.9543, 0.9555]    # 32M params
    medium_decoder = [0.2536, 0.1883, 0.3016, 0.2586, 0.0768]   # 187M params
    large_decoder = [0.1133, 0.082, 0.1328, 0.1428, 0.0]        # 674M params
    
    x = np.arange(len(models))
    width = 0.25
    
    # ML conference standard colors
    colors = {
        'small': '#0077BB',    # Blue
        'medium': '#EE7733',   # Orange
        'large': '#009988'     # Teal
    }
    
    # Plot bars
    ax.bar(x - width, small_decoder, width, label='32M params', color=colors['small'], edgecolor='black', linewidth=1)
    ax.bar(x, medium_decoder, width, label='187M params', color=colors['medium'], edgecolor='black', linewidth=1)
    ax.bar(x + width, large_decoder, width, label='674M params', color=colors['large'], edgecolor='black', linewidth=1)
    
    ax.set_ylabel('FPR@95%TPR', labelpad=15)
    ax.set_title('Impact of Decoder Sizes', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(-0.05, 1.05)  # Consistent y-axis range
    ax.legend(edgecolor='black')
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Set spines to black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

def create_ensemble_decoder_plot(ax):
    """Create a plot showing the impact of number of decoders."""
    # Disable auto-layout to prevent automatic scaling
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['figure.constrained_layout.use'] = False
    
    models = ["SD 1.5", "SD 1.4", "SD 1.3", "SD 1.2", "SD 1.1"]
    num_decoders = [1, 2, 3, 4, 5]
    
    # FPR values for each model and number of decoders
    fpr_values = {
        "SD 1.5": [0.0888, 0.0625, 0.0354, 0.0315, 0.0299],
        "SD 1.4": [0.0509, 0.0251, 0.0172, 0.0165, 0.0110],
        "SD 1.3": [0.0870, 0.0617, 0.0451, 0.0298, 0.0265],
        "SD 1.2": [0.0982, 0.0685, 0.0475, 0.0384, 0.0358],
        "SD 1.1": [0.0089, 0.0053, 0.0035, 0.0019, 0.0021]
    }
    
    # ML conference standard colors
    colors = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#33BBEE']
    
    # Plot lines for each model
    for idx, model in enumerate(models):
        ax.plot(num_decoders, fpr_values[model], 
                marker='o' if idx % 2 == 0 else 's',
                label=model, color=colors[idx],
                alpha=0.9, markersize=10, linestyle='-', linewidth=2.5,
                markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('Number of Decoders', labelpad=15)
    ax.set_ylabel('FPR@95%TPR', labelpad=15)
    ax.set_title('Performance of Ensemble Decoder', pad=15)
    ax.set_xticks(num_decoders)
    
    # Force y-axis limits and ticks
    ax.set_ylim(0, 0.5)
    yticks = np.linspace(0, 0.5, 6)  # 6 ticks from 0 to 0.5
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{x:.1f}' for x in yticks])  # Format as decimals
    
    # Add legend
    ax.legend(edgecolor='black', loc='upper right')
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Set spines to black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

# Set up plotting style
setup_plotting_style()

# Create figure for main FPR plot with example image
fig1 = plt.figure(figsize=(14, 6))  # Slightly narrower overall

# Create subplots with specific width ratios and less spacing
gs = fig1.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.1)  # Reduced wspace for closer plots

# Create subplot for FPR plot
ax1 = fig1.add_subplot(gs[0])  # Left subplot
create_plot(ax1, sd_methods, sd_fpr_values)

# Create subplot for example image
ax_img = fig1.add_subplot(gs[1])  # Right subplot
img = plt.imread('nova_cafe_sd2-1.png')
ax_img.imshow(img)
ax_img.set_title('SD2.1 Example', pad=15)
# Add subtitle below the image
ax_img.text(0.5, -0.1, 'The Japanese Cafe prompt', 
            horizontalalignment='center',
            transform=ax_img.transAxes,
            fontsize=20)
ax_img.axis('off')  # Hide axes for the image

# Save with tight layout
plt.savefig('sd_fpr.png', bbox_inches='tight', dpi=300, transparent=True)
plt.close()

# Create figure for training samples and decoder size comparison
fig4 = plt.figure(figsize=(14, 6))  # Wider figure to accommodate two plots

# Create subplots
ax4_left = fig4.add_subplot(121)  # Left subplot
ax4_right = fig4.add_subplot(122)  # Right subplot

# Create decoder size comparison on left subplot
create_decoder_size_comparison(ax4_left)

# Create training samples comparison on right subplot
create_training_samples_comparison(ax4_right)

# Adjust layout and save
plt.tight_layout()
plt.savefig('sd_training_resource.png', bbox_inches='tight', dpi=300, transparent=True)
plt.close()

# Create figure for inference steps comparison
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111)
create_inference_steps_comparison(ax2)
plt.tight_layout()
plt.savefig('sd_inference_steps_comparison.png', bbox_inches='tight', dpi=300, transparent=True)
plt.close()

# Create figure for prompt comparison
fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111)
create_prompt_comparison(ax3)
plt.tight_layout()
plt.savefig('sd_prompt_comparison.png', bbox_inches='tight', dpi=300, transparent=True)
plt.close()

# Create figure for ensemble decoder comparison
plt.clf()  # Clear any existing plots
fig5 = plt.figure(figsize=(8, 6))
ax5 = fig5.add_subplot(111)

# Create plot with forced y-axis limits
create_ensemble_decoder_plot(ax5)

# Force y-axis limits again right before saving
ax5.set_ylim(0, 0.5)
ax5.set_yticks(np.linspace(0, 0.5, 6))
ax5.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

# Force draw and save
plt.draw()
plt.savefig('sd_ensemble_decoder.png', bbox_inches='tight', dpi=300, transparent=True)
plt.close('all')  # Close all figures to ensure clean state 