import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import font_manager

# Add Segoe UI font
plt.rcParams['font.sans-serif'] = ['Segoe UI']
plt.rcParams['font.family'] = 'sans-serif'

# Set style for professional publication
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 24,              # Even larger base font size
    'axes.labelsize': 28,         # Much larger axis label size
    'axes.titlesize': 30,         # Much larger title size
    'legend.fontsize': 22,        # Much larger legend font size
    'xtick.labelsize': 24,        # Much larger tick label size
    'ytick.labelsize': 24,        # Much larger tick label size
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.facecolor': 'none',   
    'axes.facecolor': 'white',    
    'grid.linewidth': 1.0,        # Thicker grid lines
    'axes.linewidth': 2.0,        # Thicker axis lines
    'legend.title_fontsize': 24,  # Larger legend title
    'font.weight': 'normal',      # Normal weight for better Segoe UI rendering
    'axes.labelweight': 'normal', # Normal weight for axis labels
    'axes.titleweight': 'normal'  # Normal weight for title
})

# Data
methods = [
    "1K Train Data, ADA Aug",
    "30K Train Data, ADA Aug",
    "70K Train Data, BCR Aug",
    "70K Train Data, No Aug",
    "Quantization INT8",
    "Quantization INT4",
    "Downsampling to 128",
    "Downsampling to 224"
]

fid_scores = [14.0726, 2.5086, 2.1812, 3.1219, 1.7242, 268.1055, 40.2009, 9.9268]
pixels = [1, 4, 16, 32, 256, 1024]
fpr_values = [
    [0.0314, 0.0007, 0, 0, 0.0537, 0.1598],
    [0.0544, 0.0014, 0, 0, 0.0654, 0.1989],
    [0.1243, 0.0258, 0, 0, 0.1938, 0.3979],
    [0.1229, 0.0298, 0, 0, 0.2223, 0.4445],
    [0.9505, 0.9505, 0.9535, 0.9559, 0.9514, 0.9524],
    [0.0208, 0.0001, 0, 0, 0.0273, 0.0536],
    [0.2875, 0.1264, 0, 0, 0.5423, 0.7475],
    [0.7481, 0.7208, 0.0762, 0.2001, 0.8972, 0.9481]
]

# Create figure with original larger size
fig, ax = plt.subplots(figsize=(16, 8))  # Restored to original larger size

# Define categories and colors
categories = {
    'Data Size': ['1K Train Data, ADA Aug', '30K Train Data, ADA Aug'],
    'Augmentation': ['70K Train Data, BCR Aug', '70K Train Data, No Aug'],
    'Quantization': ['Quantization INT8', 'Quantization INT4'],
    'Downsampling': ['Downsampling to 128', 'Downsampling to 224']
}

# Define a modern, vibrant color palette
colors = {
    'Data Size': '#4361EE',      # Modern vibrant blue
    'Augmentation': '#F72585',    # Electric pink
    'Quantization': '#4CC9F0',    # Bright cyan
    'Downsampling': '#7209B7'     # Deep purple
}

# Plot FPR vs Pixels with alternating markers for each category
for cat_idx, (cat, methods_in_cat) in enumerate(categories.items()):
    for method_idx, method in enumerate(methods_in_cat):
        idx = methods.index(method)
        marker = 'o' if method_idx % 2 == 0 else 's'
        label = f"{method} (FID: {fid_scores[idx]:.2f})"
        ax.plot(range(len(pixels)), fpr_values[idx], 
                marker=marker, label=label, color=colors[cat], 
                alpha=0.9, markersize=14, linestyle='-', linewidth=3.5)  # Larger markers and lines

# Set x-axis ticks and labels
ax.set_xticks(range(len(pixels)))
ax.set_xticklabels(pixels)

# Customize the plot
ax.set_xlabel('Number of Pixels', labelpad=20)  # More padding
ax.set_ylabel('FPR@95%TPR', labelpad=20)       # More padding
ax.set_title('FPR vs Number of Pixels for Different Methods', pad=20)

# Add legend with better spacing and larger elements
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, borderaxespad=0, 
         handletextpad=0.5, handlelength=2.0,  # Larger legend elements
         markerscale=1.5)  # Larger legend markers

# Adjust layout to prevent legend cutoff
plt.tight_layout()

# Save with transparent background
plt.savefig('results_figure.pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.savefig('results_figure.png', bbox_inches='tight', dpi=300, transparent=True)
plt.close() 