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

# FPR values for each model
sd_fpr_values = [
    [0.8281, 0.4844, 0.3594, 0.4336, 0.3789, 0.3984, 0.3984],
    [0.7930, 0.2578, 0.1992, 0.2383, 0.2148, 0.2344, 0.2422],
    [0.8008, 0.3867, 0.3438, 0.3516, 0.3359, 0.3359, 0.3438],
    [0.8398, 0.4844, 0.3906, 0.4062, 0.3672, 0.3633, 0.3828],
    [0.6953, 0.0742, 0.0352, 0.0859, 0.0391, 0.0625, 0.0586]
]

def create_plot(ax, methods, fpr_values):
    """Create a plot for Stable Diffusion results."""
    pixels = [4, 64, 256, 1024, 4096, 16384, 65536]
    
    # NeurIPS-style colors
    colors = [
        '#E64B35',  # Red
        '#4DBBD5',  # Blue
        '#00A087',  # Teal
        '#3C5488',  # Navy
        '#F39B7F'   # Light Red
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
                alpha=0.9, markersize=10, linestyle='-', linewidth=2.5)[0]
        
        lines.append(line)
        labels.append(label)
    
    # Customize the plot
    ax.set_xticks(range(len(pixels)))
    ax.set_xticklabels(pixels)
    ax.set_xlabel('Fingerprint Length (Number of Pixels Selected)', labelpad=15)
    ax.set_ylabel('FPR@95%TPR', labelpad=15)
    ax.set_title('Target Model: Stable Diffusion 2.1', pad=15)
    ax.set_ylim(-0.05, 1.05)
    
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
    """Create a plot comparing different inference steps for 1024 pixels."""
    models = ["SD 1.5", "SD 1.4", "SD 1.3", "SD 1.2", "SD 1.1"]
    steps_25 = [0.4336, 0.2383, 0.3516, 0.4062, 0.0859]  # 25 steps
    steps_15 = [0.1992, 0.1250, 0.2227, 0.3047, 0.0625]  # 15 steps
    steps_5 = [0.0000, 0.0039, 0.0039, 0.0078, 0.0000]   # 5 steps
    
    # Add small offset to 5-steps values to make them visible
    steps_5_visible = [max(val, 0.01) for val in steps_5]  # Minimum height of 0.01
    
    x = np.arange(len(models))
    width = 0.25
    
    # NeurIPS-style colors
    colors = {
        '25_steps': '#E64B35',  # Red
        '15_steps': '#4DBBD5',  # Blue
        '5_steps': '#00A087'    # Teal
    }
    
    # Plot bars with new colors
    bars1 = ax.bar(x - width, steps_25, width, label='25 Steps', color=colors['25_steps'])
    bars2 = ax.bar(x, steps_15, width, label='15 Steps', color=colors['15_steps'])
    bars3 = ax.bar(x + width, steps_5_visible, width, label='5 Steps', color=colors['5_steps'])
    
    # Add value labels for all bars
    for bars, values in [(bars1, steps_25), (bars2, steps_15), (bars3, steps_5)]:
        for bar, val in zip(bars, values):
            # Show all values, including zeros
            height = max(val, 0.01) if bars == bars3 else val
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'{val:.4f}', ha='center', va='bottom', rotation=90,
                   fontsize=10, color='black')
    
    ax.set_ylabel('FPR@95%TPR', labelpad=15)
    ax.set_title('Comparison of Inference Steps\n(1024 Pixels)', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Set spines to black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

def create_prompt_comparison(ax):
    """Create a plot comparing prompt vs no prompt for 1024 pixels with 15 steps."""
    models = ["SD 1.5", "SD 1.4", "SD 1.3", "SD 1.2", "SD 1.1"]
    with_prompt = [0.1992, 0.1250, 0.2227, 0.3047, 0.0625]  # 15 steps with prompt
    no_prompt = [0.2578, 0.2578, 0.1992, 0.4531, 0.7109]    # 15 steps no prompt
    
    x = np.arange(len(models))
    width = 0.35
    
    # NeurIPS-style colors
    colors = {
        'no_prompt': '#E64B35',    # Red
        'with_prompt': '#4DBBD5'   # Blue
    }
    
    # Reversed order: no prompt first (red), then with prompt (blue)
    ax.bar(x - width/2, no_prompt, width, label='No Prompt', color=colors['no_prompt'])
    ax.bar(x + width/2, with_prompt, width, label='With Prompt', color=colors['with_prompt'])
    
    ax.set_ylabel('FPR@95%TPR', labelpad=15)
    ax.set_title('Prompt vs No Prompt Comparison\n(1024 Pixels, 15 Steps)', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Set spines to black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

def create_iteration_comparison(ax):
    """Create a plot comparing training iterations for 256 pixels."""
    models = ["SD 1.5", "SD 1.4", "SD 1.3", "SD 1.2", "SD 1.1"]
    iter_2000 = [0.3594, 0.1992, 0.3438, 0.3906, 0.0352]  # 2000 iterations
    iter_5000 = [0.2656, 0.1523, 0.2305, 0.3008, 0.0078]  # 5000 iterations
    
    x = np.arange(len(models))
    width = 0.35
    
    # NeurIPS-style colors
    colors = {
        '2000_iter': '#E64B35',    # Red
        '5000_iter': '#4DBBD5'     # Blue
    }
    
    # Reversed colors: 2000 iterations red, 5000 iterations blue
    ax.bar(x - width/2, iter_2000, width, label='2000 Iterations', color=colors['2000_iter'])
    ax.bar(x + width/2, iter_5000, width, label='5000 Iterations', color=colors['5000_iter'])
    
    ax.set_ylabel('FPR@95%TPR', labelpad=15)
    ax.set_title('Training Iteration Comparison\n(256 Pixels)', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Set spines to black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

# Set up plotting style
setup_plotting_style()

# Create figure for main FPR plot
fig1 = plt.figure(figsize=(8, 6))  # More compact size
ax1 = fig1.add_subplot(111)
create_plot(ax1, sd_methods, sd_fpr_values)

# Adjust layout and save
plt.tight_layout()
plt.savefig('sd_fpr.png', bbox_inches='tight', dpi=300, transparent=True)
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

# Create figure for iteration comparison
fig4 = plt.figure(figsize=(8, 6))
ax4 = fig4.add_subplot(111)
create_iteration_comparison(ax4)
plt.tight_layout()
plt.savefig('sd_iteration_comparison.png', bbox_inches='tight', dpi=300, transparent=True)
plt.close() 