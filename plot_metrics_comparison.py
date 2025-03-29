import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Set up the font settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = False

# Data
pixel_numbers = [1, 4, 8, 32, 128, 1024]
x_positions = np.arange(len(pixel_numbers))

# Group cases by category
training_cases = ['1K Train Data', '30K Train Data']
augmentation_cases = ['BCR Augmentation', 'No Augmentation']
quantization_cases = ['Quantization INT8', 'Quantization INT4']
other_cases = ['Original Model', 'Truncation', 'Downsampling']

# Define markers for each group
training_markers = ['o', 's']             # circle, square
augmentation_markers = ['^', 'D']         # triangle, diamond
quantization_markers = ['o', 's']         # circle, square
other_markers = ['o', 'o', 'o']           # all circles for other cases

# Define line styles
solid_line = '-'
dashed_line = '--'

# Updated color palette - bright colors for solid lines, dark colors for dashed
training_color = '#e74c3c'    # Bright red
augmentation_color = '#ff7f0e' # Bright orange
other_colors = ['#000080', '#8c564b', '#1b5e20']  # Navy blue, Dark brown, Dark forest green
quantization_color = '#7d3c98' # Dark purple

# FPR@95%TPR values
fpr_values = {
    'Original Model': [0.292, 0.000, 0.000, 0.000, 0.000, 0.000],
    '1K Train Data': [0.058, 0.000, 0.000, 0.000, 0.000, 0.000],
    '30K Train Data': [0.074, 0.000, 0.000, 0.000, 0.000, 0.000],
    'BCR Augmentation': [0.048, 0.000, 0.000, 0.000, 0.000, 0.000],
    'No Augmentation': [0.058, 0.000, 0.000, 0.000, 0.000, 0.000],
    'Quantization INT8': [0.490, 0.353, 0.288, 0.054, 0.084, 0.390],
    'Quantization INT4': [0.050, 0.000, 0.000, 0.000, 0.000, 0.000],
    'Truncation': [0.086, 0.000, 0.000, 0.000, 0.000, 0.000],
    'Downsampling': [0.336, 0.000, 0.000, 0.000, 0.000, 0.000]
}

# ASR@95%TPR values
asr_values = {
    'Original Model': [0.74, 0.77, 0.40, 0.20, 0.29, 0.63],
    '1K Train Data': [0.75, 0.18, 0.01, 0.00, 0.05, 0.41],
    '30K Train Data': [0.81, 0.32, 0.08, 0.02, 0.12, 0.43],
    'BCR Augmentation': [0.82, 0.48, 0.29, 0.06, 0.14, 0.45],
    'No Augmentation': [0.75, 0.49, 0.20, 0.05, 0.21, 0.57],
    'Quantization INT8': [0.94, 0.96, 0.94, 0.96, 0.92, 0.97],
    'Quantization INT4': [0.38, 0.06, 0.07, 0.00, 0.04, 0.05],
    'Truncation': [0.87, 0.81, 0.85, 0.87, 0.80, 0.82],
    'Downsampling': [0.75, 0.65, 0.46, 0.23, 0.31, 0.75]
}

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Function to plot data on a given axis
def plot_metric(ax, values, title, ylabel):
    # Plot other cases first (to appear at top of legend)
    for case, marker, color in zip(other_cases, other_markers, other_colors):
        ax.plot(x_positions, values[case], marker=marker, linewidth=1.0, 
                markersize=4, label=case, color=color, markerfacecolor=color,
                linestyle=dashed_line, alpha=0.5)

    # Plot training cases
    for case, marker in zip(training_cases, training_markers):
        ax.plot(x_positions, values[case], marker=marker, linewidth=1.0, 
                markersize=4, label=case, color=training_color, markerfacecolor=training_color,
                linestyle=solid_line)

    # Plot augmentation cases
    for case, marker in zip(augmentation_cases, augmentation_markers):
        ax.plot(x_positions, values[case], marker=marker, linewidth=1.0, 
                markersize=4, label=case, color=augmentation_color, markerfacecolor=augmentation_color,
                linestyle=solid_line)

    # Plot quantization cases
    for case, marker in zip(quantization_cases, quantization_markers):
        ax.plot(x_positions, values[case], marker=marker, linewidth=1.0, 
                markersize=4, label=case, color=quantization_color, markerfacecolor=quantization_color,
                linestyle=dashed_line, alpha=0.5)

    # Customize the plot
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.set_xlabel('Number of Pixels', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(pixel_numbers, fontsize=11)
    ax.tick_params(axis='y', labelsize=11)

# Plot FPR
plot_metric(ax1, fpr_values, 'FPR@95%TPR vs. Number of Pixels\n(Key Length = 128)', 'FPR@95%TPR')

# Plot ASR
plot_metric(ax2, asr_values, 'ASR@95%TPR vs. Number of Pixels\n(Key Length = 128)', 'ASR@95%TPR')

# Add legend to the right of both plots
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1.02, 0.5), loc='center left', 
          fontsize=10, handlelength=1.5, columnspacing=1, handletextpad=0.5)

# Adjust layout
plt.tight_layout()

# Save the figure in high resolution
plt.savefig('metrics_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('metrics_comparison.png', format='png', dpi=300, bbox_inches='tight')
plt.close() 