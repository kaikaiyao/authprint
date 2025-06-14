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
        'font.size': 20,           
        'axes.labelsize': 22,      
        'axes.titlesize': 24,      
        'legend.fontsize': 18,     
        'xtick.labelsize': 20,     
        'ytick.labelsize': 20,     
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
        'grid.color': '#E5E5E5',   
        'figure.edgecolor': 'black'
    })

# Comment out the STD values dictionaries but preserve them for future use
"""
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
"""

def create_subplot(ax, methods, fid_scores, fpr_values, dataset_name, plot_type="training"):
    """Create a subplot for the given dataset."""
    pixels = [1, 4, 16, 32, 256, 1024]
    
    # ML conference standard colors
    aaai_colors = {
        'Data Size': ['#0077BB', '#EE7733'],      # Blue and Orange
        'Augmentation': ['#009988', '#CC3311'],    # Teal and Red
        'Quantization': ['#88CCEE', '#0077BB'],    # Light blue and Dark blue (INT8 light circle, INT4 dark square)
        'Downsampling': ['#88DDBB', '#009988']     # Light green and Dark green (224px light circle, 128px dark square)
    }
    
    if plot_type == "training":
        # Define categories for training methods
        categories = {
            'Data Size': [methods[0], methods[1]],
            'Augmentation': [methods[2], methods[3]]
        }
    else:
        # Define categories for optimization methods
        categories = {
            'Quantization': [methods[4], methods[5]],
            'Downsampling': [methods[6], methods[7]]
        }
    
    # Define line styles for each category
    line_styles = {cat: '-' for cat in categories.keys()}
    
    lines = []  # Store lines for legend
    labels = []  # Store labels for legend
    
    # Get the appropriate std values dictionary based on dataset
    # std_values = ffhq_std_values if dataset_name == "FFHQ" else lsun_std_values
    
    # Plot FPR vs Pixels with std
    for cat_idx, (cat, methods_in_cat) in enumerate(categories.items()):
        for method_idx, method in enumerate(methods_in_cat):
            idx = methods.index(method)
            marker = 'o' if method_idx % 2 == 0 else 's'
            # Restore "FID:" prefix but keep one decimal place
            label = f"{method} (FID: {fid_scores[idx]:.1f})"
            
            y_values = np.array(fpr_values[idx])
            # std_values_for_method = std_values[method]
            
            color = aaai_colors[cat][method_idx]
            
            # Plot shaded std region
            # ax.fill_between(range(len(pixels)), 
            #               np.maximum(y_values - std_values_for_method, 0.0),
            #               np.minimum(y_values + std_values_for_method, 1.0),
            #               color=color, 
            #               alpha=0.2)
            
            # Plot main line
            line = ax.plot(range(len(pixels)), y_values,
                    marker=marker, label=label, color=color,
                    alpha=0.9, markersize=10, linestyle=line_styles[cat], linewidth=2.5,
                    markeredgecolor='black', markeredgewidth=1)[0]
            
            lines.append(line)
            labels.append(label)
    
    # Customize the subplot
    ax.set_xticks(range(len(pixels)))
    ax.set_xticklabels(pixels)
    ax.set_xlabel('Fingerprint Length', labelpad=15)
    ax.set_ylabel('FPR@95%TPR', labelpad=15)
    
    # Split titles into main title and subtitle
    if dataset_name == "FFHQ":
        main_title = "StyleGAN2, FFHQ"
        subtitle = "(Target Model: 70K Train Data, ADA Aug)"
    else:
        main_title = "StyleGAN2, LSUN-Cat"
        subtitle = "(Target Model: 100K Train Data, ADA Aug)"
    
    # Set main title and subtitle with default spacing
    ax.set_title(main_title + '\n', pad=20, fontsize=24)
    ax.text(0.5, 1.05, subtitle, 
            horizontalalignment='center',
            transform=ax.transAxes,
            fontsize=18)  # Default spacing
    
    ax.set_ylim(-0.05, 1.05)  # Add margins above and below
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
    
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
    "Quant INT8",  # Light blue, circle
    "Quant INT4",  # Dark blue, square
    "Downsample 224px",  # Light green, circle
    "Downsample 128px"   # Dark green, square
]

ffhq_fid_scores = [14.0726, 2.5086, 2.1812, 3.1219, 1.7242, 268.1055, 9.9268, 40.2009]
ffhq_fpr_values = [
    [0.0314, 0.0007, 0, 0, 0.0537, 0.1598],
    [0.0544, 0.0014, 0, 0, 0.0654, 0.1989],
    [0.1243, 0.0258, 0, 0, 0.1938, 0.3979],
    [0.1229, 0.0298, 0, 0, 0.2223, 0.4445],
    [0.9505, 0.9505, 0.9535, 0.9559, 0.9514, 0.9524],
    [0.0208, 0.0001, 0, 0, 0.0273, 0.0536],
    [0.7481, 0.7208, 0.0762, 0.2001, 0.8972, 0.9481],
    [0.2875, 0.1264, 0, 0, 0.5423, 0.7475]
]

# Data for LSUN Cat dataset
lsun_methods = [
    "1K Train Data, ADA Aug",
    "30K Train Data, ADA Aug",
    "100K Train Data, BCR Aug",
    "100K Train Data, No Aug",
    "Quant INT8",  # Light blue, circle
    "Quant INT4",  # Dark blue, square
    "Downsample 224px",  # Light green, circle
    "Downsample 128px"   # Dark green, square
]

lsun_fid_scores = [28.5018, 4.9672, 4.6296, 5.0971, 2.092, 286.415, 4.839, 19.6367]
lsun_fpr_values = [
    [0.5651, 0.0026, 0, 0, 0.3674, 0.8167],
    [0.5583, 0.0026, 0.0001, 0.0001, 0.3657, 0.8423],
    [0.2082, 0.002, 0.0001, 0.0001, 0.1831, 0.7713],
    [0.1559, 0.018, 0.0024, 0.0013, 0.1838, 0.7501],
    [0.954, 0.9419, 0.9446, 0.949, 0.9482, 0.9482],
    [0.0045, 0, 0, 0, 0, 0],
    [0.9217, 0.7992, 0.6728, 0.7191, 0.966, 0.9781],
    [0.8457, 0.3517, 0.1332, 0.1452, 0.8715, 0.9843]
]

# Set up plotting style
setup_plotting_style()

# Create two separate figures - one for training methods and one for optimization methods
def create_and_save_plots(plot_type="training"):
    # Set consistent figure size and layout parameters
    fig_width = 14  # Double width to accommodate two plots side by side
    fig_height = 7 if plot_type == "optimization" else 5  # Less height needed for training plots
    
    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    if plot_type == "training":
        # For training plots, use simple subplot layout
        ax_ffhq = plt.subplot(121)
        ax_lsun = plt.subplot(122)
    else:
        # For optimization plots, use gridspec for bottom legends
        gs = fig.add_gridspec(2, 2, height_ratios=[4, 1])
        ax_ffhq = fig.add_subplot(gs[0, 0])
        ax_lsun = fig.add_subplot(gs[0, 1])
    
    # Create FFHQ subplot
    lines1, labels1 = create_subplot(ax_ffhq, ffhq_methods, ffhq_fid_scores, ffhq_fpr_values, "FFHQ", plot_type)
    
    # Create LSUN subplot
    lines2, labels2 = create_subplot(ax_lsun, lsun_methods, lsun_fid_scores, lsun_fpr_values, "LSUN-Cat", plot_type)
    
    if plot_type == "training":
        # Place legends inside plots at top-left for training plots
        legend_ffhq = ax_ffhq.legend(lines1, labels1,
                                loc='upper left',
                                ncol=1,  # Vertical alignment
                                handletextpad=0.3,
                                handlelength=1.5,
                                markerscale=1.2,
                                fontsize=15,
                                edgecolor='black',
                                bbox_to_anchor=(0.02, 0.98))
        
        legend_lsun = ax_lsun.legend(lines2, labels2,
                                loc='upper left',
                                ncol=1,  # Vertical alignment
                                handletextpad=0.3,
                                handlelength=1.5,
                                markerscale=1.2,
                                fontsize=15,
                                edgecolor='black',
                                bbox_to_anchor=(0.02, 0.98))
        
        # Ensure legend backgrounds are opaque
        legend_ffhq.get_frame().set_facecolor('white')
        legend_ffhq.get_frame().set_alpha(0.9)
        legend_lsun.get_frame().set_facecolor('white')
        legend_lsun.get_frame().set_alpha(0.9)
        
        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.3)
    else:
        # Create legend areas below each plot for optimization plots
        legend_ax_ffhq = fig.add_subplot(gs[1, 0])
        legend_ax_lsun = fig.add_subplot(gs[1, 1])
        legend_ax_ffhq.axis('off')
        legend_ax_lsun.axis('off')
        
        # Reorder lines and labels for the desired legend layout
        def reorder_for_legend(lines, labels):
            # Get indices for optimization methods (last 4 items)
            opt_indices = range(len(lines)-4, len(lines))
            # Reorder as: INT8, INT4, 224px, 128px
            new_order = list(range(len(lines)-4))  # Keep training methods in original order
            new_order.extend([opt_indices[0], opt_indices[1], opt_indices[2], opt_indices[3]])
            return [lines[i] for i in new_order], [labels[i] for i in new_order]
        
        lines1_reordered, labels1_reordered = reorder_for_legend(lines1, labels1)
        lines2_reordered, labels2_reordered = reorder_for_legend(lines2, labels2)
        
        # FFHQ Legend with custom marker assignment
        legend_ffhq = legend_ax_ffhq.legend(lines1_reordered, labels1_reordered,
                                loc='center',
                                ncol=2,
                                handletextpad=0.3,
                                handlelength=1.5,
                                markerscale=1.2,
                                fontsize=14,
                                edgecolor='black',
                                columnspacing=1.0,
                                bbox_to_anchor=(0.5, 0.5))
        
        # LSUN Legend with custom marker assignment
        legend_lsun = legend_ax_lsun.legend(lines2_reordered, labels2_reordered,
                                loc='center',
                                ncol=2,
                                handletextpad=0.3,
                                handlelength=1.5,
                                markerscale=1.2,
                                fontsize=14,
                                edgecolor='black',
                                columnspacing=1.0,
                                bbox_to_anchor=(0.5, 0.5))
        
        # Ensure legend backgrounds are opaque
        legend_ffhq.get_frame().set_facecolor('white')
        legend_ffhq.get_frame().set_alpha(1.0)
        legend_lsun.get_frame().set_facecolor('white')
        legend_lsun.get_frame().set_alpha(1.0)
        
        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Save the combined figure
    suffix = "training" if plot_type == "training" else "optimization"
    plt.savefig(f'stylegan2_fpr_combined_{suffix}.png', bbox_inches='tight', dpi=300, transparent=True)
    plt.close()

# Create both plots
create_and_save_plots("training")
create_and_save_plots("optimization") 