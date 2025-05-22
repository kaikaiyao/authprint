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

ffhq_std_values = {
    "1K Train Data, ADA Aug": [0.026, 0.0142, 0.0085, 0.0057, 0.0044, 0.0051],
    "30K Train Data, ADA Aug": [0.027, 0.0098, 0.0083, 0.0049, 0.0042, 0.0048],
    "70K Train Data, BCR Aug": [0.023, 0.0155, 0.0048, 0.0043, 0.0069, 0.0065],
    "70K Train Data, No Aug": [0.041, 0.0185, 0.0122, 0.0061, 0.0098, 0.0092],
    "Quantization INT8": [0.0095, 0.0081, 0.0064, 0.0063, 0.0059, 0.0057],
    "Quantization INT4": [0.0022, 0.0019, 0.0021, 0.0018, 0.0022, 0.0021],
    "Downsampling to 128": [0.019, 0.0072, 0.0042, 0.0038, 0.0049, 0.0043],
    "Downsampling to 224": [0.037, 0.0255, 0.0047, 0.0042, 0.0068, 0.0049]
}

lsun_std_values = {
    "1K Train Data, ADA Aug": [0.089, 0.0142, 0.0071, 0.0045, 0.0135, 0.0119],
    "30K Train Data, ADA Aug": [0.091, 0.0156, 0.0062, 0.0048, 0.0082, 0.0098],
    "100K Train Data, BCR Aug": [0.049, 0.0148, 0.0072, 0.0058, 0.0069, 0.0165],
    "100K Train Data, No Aug": [0.056, 0.0169, 0.0082, 0.0068, 0.0073, 0.0152],
    "Quantization INT8": [0.0092, 0.0079, 0.0065, 0.0055, 0.0051, 0.0048],
    "Quantization INT4": [0.0018, 0.0022, 0.0019, 0.0021, 0.0018, 0.0023],
    "Downsampling to 128": [0.039, 0.0142, 0.0052, 0.0045, 0.0059, 0.0048],
    "Downsampling to 224": [0.052, 0.0285, 0.0155, 0.0108, 0.0112, 0.0082]
}


def create_subplot(ax, methods, fid_scores, fpr_values, dataset_name):
    """Create a subplot for the given dataset."""
    pixels = [1, 4, 16, 32, 256, 1024]
    
    # Define categories and colors
    categories = {
        'Data Size': [methods[0], methods[1]],
        'Augmentation': [methods[2], methods[3]],
        'Quantization': [methods[4], methods[5]],
        'Downsampling': [methods[6], methods[7]]
    }
    
    colors = {
        'Data Size': '#2166AC',      # Strong blue
        'Augmentation': '#B2182B',    # Deep red
        'Quantization': '#35978F',    # Teal
        'Downsampling': '#756BB1'     # Purple
    }

    # Define line styles for each category
    line_styles = {
        'Data Size': '-',           # Solid line
        'Augmentation': '-',        # Solid line
        'Quantization': '--',       # Dashed line
        'Downsampling': '--'        # Dashed line
    }
    
    lines = []  # Store lines for legend
    labels = []  # Store labels for legend
    
    # Get the appropriate std values dictionary based on dataset
    std_values = ffhq_std_values if dataset_name == "FFHQ" else lsun_std_values
    
    # Plot FPR vs Pixels with std
    for cat_idx, (cat, methods_in_cat) in enumerate(categories.items()):
        for method_idx, method in enumerate(methods_in_cat):
            idx = methods.index(method)
            marker = 'o' if method_idx % 2 == 0 else 's'
            label = f"{method} (FID: {fid_scores[idx]:.2f})"
            
            y_values = np.array(fpr_values[idx])
            std_values_for_method = std_values[method]
            
            # Plot shaded std region
            ax.fill_between(range(len(pixels)), 
                          np.maximum(y_values - std_values_for_method, 0.0),
                          np.minimum(y_values + std_values_for_method, 1.0),
                          color=colors[cat], 
                          alpha=0.2)
            
            # Plot main line
            line = ax.plot(range(len(pixels)), y_values,
                    marker=marker, label=label, color=colors[cat],
                    alpha=0.9, markersize=10, linestyle=line_styles[cat], linewidth=2.5)[0]
            
            lines.append(line)
            labels.append(label)
    
    # Customize the subplot
    ax.set_xticks(range(len(pixels)))
    ax.set_xticklabels(pixels)
    ax.set_xlabel('Fingerprint Length (Number of Pixels Selected)', labelpad=15)
    ax.set_ylabel('FPR@95%TPR', labelpad=15)
    ax.set_title(f"Target Model: {dataset_name} 70k Train Data, ADA Aug" if dataset_name == "FFHQ" else f"Target Model: {dataset_name} 100k Train Data, ADA Aug", pad=15)
    ax.set_ylim(-0.05, 1.05)  # Add margins above and below
    
    # Set spines to black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    return lines, labels

# Data for FFHQ dataset
ffhq_methods = [
    "1K Train Data, ADA Aug",
    "30K Train Data, ADA Aug",
    "70K Train Data, BCR Aug",
    "70K Train Data, No Aug",
    "Quantization INT8",
    "Quantization INT4",
    "Downsampling to 128",
    "Downsampling to 224"
]

ffhq_fid_scores = [14.0726, 2.5086, 2.1812, 3.1219, 1.7242, 268.1055, 40.2009, 9.9268]
ffhq_fpr_values = [
    [0.0314, 0.0007, 0, 0, 0.0537, 0.1598],
    [0.0544, 0.0014, 0, 0, 0.0654, 0.1989],
    [0.1243, 0.0258, 0, 0, 0.1938, 0.3979],
    [0.1229, 0.0298, 0, 0, 0.2223, 0.4445],
    [0.9505, 0.9505, 0.9535, 0.9559, 0.9514, 0.9524],
    [0.0208, 0.0001, 0, 0, 0.0273, 0.0536],
    [0.2875, 0.1264, 0, 0, 0.5423, 0.7475],
    [0.7481, 0.7208, 0.0762, 0.2001, 0.8972, 0.9481]
]

# Data for LSUN Cat dataset
lsun_methods = [
    "1K Train Data, ADA Aug",
    "30K Train Data, ADA Aug",
    "100K Train Data, BCR Aug",
    "100K Train Data, No Aug",
    "Quantization INT8",
    "Quantization INT4",
    "Downsampling to 128",
    "Downsampling to 224"
]

lsun_fid_scores = [28.5018, 4.9672, 4.6296, 5.0971, 2.092, 286415.3296, 19.6367, 4.839]
lsun_fpr_values = [
    [0.5651, 0.0026, 0, 0, 0.3674, 0.8167],
    [0.5583, 0.0026, 0.0001, 0.0001, 0.3657, 0.8423],
    [0.2082, 0.002, 0.0001, 0.0001, 0.1831, 0.7713],
    [0.1559, 0.018, 0.0024, 0.0013, 0.1838, 0.7501],
    [0.954, 0.9419, 0.9446, 0.949, 0.9482, 0.9482],
    [0.0045, 0, 0, 0, 0, 0],
    [0.8457, 0.3517, 0.1332, 0.1452, 0.8715, 0.9843],
    [0.9217, 0.7992, 0.6728, 0.7191, 0.966, 0.9781]
]

# Set up plotting style
setup_plotting_style()

# Create a single figure with two subplots
fig = plt.figure(figsize=(18, 7.5))  # Middle ground size
gs = fig.add_gridspec(2, 2, height_ratios=[3.5, 1], width_ratios=[1, 1])

# Create main subplots and share y-axis
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)  # Share y-axis with first subplot

# Create both subplots and get legend handles
lines1, labels1 = create_subplot(ax1, ffhq_methods, ffhq_fid_scores, ffhq_fpr_values, "FFHQ")
lines2, labels2 = create_subplot(ax2, lsun_methods, lsun_fid_scores, lsun_fpr_values, "LSUN-Cat")

# Remove redundant y-axis label from second subplot
ax2.set_ylabel('')

# Create separate legend axes
legend_ax1 = fig.add_subplot(gs[1, 0])
legend_ax2 = fig.add_subplot(gs[1, 1])

# Remove axes for legend subplots
legend_ax1.axis('off')
legend_ax2.axis('off')

# Add legends below each subplot
legend1 = legend_ax1.legend(lines1, labels1, 
                          loc='center',
                          ncol=2, borderaxespad=0,
                          handletextpad=0.3, handlelength=1.5, markerscale=1.2,
                          fontsize=15,  # Middle ground font size
                          edgecolor='black')

legend2 = legend_ax2.legend(lines2, labels2, 
                          loc='center',
                          ncol=2, borderaxespad=0,
                          handletextpad=0.3, handlelength=1.5, markerscale=1.2,
                          fontsize=15,  # Middle ground font size
                          edgecolor='black')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('stylegan2_fpr.png', bbox_inches='tight', dpi=300, transparent=True)
plt.close() 