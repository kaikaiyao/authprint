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
pixel_numbers = [1, 8, 32, 128]
x_positions = np.arange(len(pixel_numbers))  # Create evenly spaced positions

# Group cases by category
training_cases = ['1K Train Data', '30K Train Data', 'BCR Augmentation', 'No Augmentation']
quantization_cases = ['Quantization INT8', 'Quantization INT4', 'Quantization INT2']
other_cases = ['Original Model', 'Truncation', 'Downsampling']

# Define markers for each group
training_markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
quantization_markers = ['o', 's', '^']    # circle, square, triangle
other_markers = ['o', 'o', 'o']           # all circles for other cases

# FPR@95%TPR values for each case (in order of pixel_numbers)
values = {
    'Original Model': [0.292, 0.000, 0.000, 0.000],
    '1K Train Data': [0.058, 0.000, 0.000, 0.000],
    '30K Train Data': [0.074, 0.000, 0.000, 0.000],
    'BCR Augmentation': [0.048, 0.000, 0.000, 0.000],
    'No Augmentation': [0.058, 0.000, 0.000, 0.000],
    'Quantization INT8': [0.490, 0.288, 0.054, 0.084],
    'Quantization INT4': [0.050, 0.000, 0.000, 0.000],
    'Quantization INT2': [0.000, 0.000, 0.000, 0.000],
    'Truncation': [0.086, 0.000, 0.000, 0.000],
    'Downsampling': [0.336, 0.000, 0.000, 0.000]
}

# Create figure and axis with conference-appropriate size
fig, ax = plt.subplots(figsize=(5, 4))

# Define colors for each group
training_color = '#2ecc71'    # green
quantization_color = '#e74c3c' # red
other_colors = ['#3498db', '#9b59b6', '#f1c40f']  # blue, purple, yellow

# Plot other cases first (to appear at top of legend)
for case, marker, color in zip(other_cases, other_markers, other_colors):
    plt.plot(x_positions, values[case], marker=marker, linewidth=1.0, 
             markersize=4, label=case, color=color, markerfacecolor=color)

# Plot training cases
for case, marker in zip(training_cases, training_markers):
    plt.plot(x_positions, values[case], marker=marker, linewidth=1.0, 
             markersize=4, label=case, color=training_color, markerfacecolor=training_color)

# Plot quantization cases
for case, marker in zip(quantization_cases, quantization_markers):
    plt.plot(x_positions, values[case], marker=marker, linewidth=1.0, 
             markersize=4, label=case, color=quantization_color, markerfacecolor=quantization_color)

# Customize the plot
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xlabel('Number of Pixels', fontsize=10)
plt.ylabel('FPR@95%TPR', fontsize=10)
plt.title('FPR@95%TPR vs. Number of Pixels\n(Key Length = 128)', fontsize=11, pad=10)

# Customize ticks
plt.xticks(x_positions, labels=pixel_numbers, fontsize=9)
plt.yticks(fontsize=9)

# Add legend inside the plot with smaller font
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., 
          fontsize=8, ncol=1, handlelength=1.5, columnspacing=1,
          handletextpad=0.5)

# Adjust layout
plt.tight_layout()

# Save the figure in high resolution
plt.savefig('fpr_tpr_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('fpr_tpr_plot.png', format='png', dpi=300, bbox_inches='tight')
plt.close() 