import re
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

def parse_log_file(log_file):
    # Initialize lists to store metrics
    key_loss = []
    lpips_loss = []
    total_loss = []
    match_rate = []
    iterations = []
    
    # Regular expression patterns
    patterns = {
        'iteration': r'Iteration \[(\d+)/500000\]',
        'key_loss': r'Key Loss: ([\d.]+)',
        'lpips_loss': r'LPIPS Loss: ([\d.]+)',
        'total_loss': r'Total Loss: ([\d.]+)',
        'match_rate': r'Match Rate: ([\d.]+)%'
    }
    
    print(f"Reading log file: {log_file}")
    
    with open(log_file, 'r') as f:
        for line in f:
            # Skip lines that don't contain metrics
            if 'Iteration [' not in line:
                continue
                
            # Extract iteration
            iter_match = re.search(patterns['iteration'], line)
            if iter_match:
                iterations.append(int(iter_match.group(1)))
            
            # Extract metrics
            for metric, pattern in patterns.items():
                if metric != 'iteration':
                    match = re.search(pattern, line)
                    if match:
                        value = float(match.group(1))
                        if metric == 'key_loss':
                            key_loss.append(value)
                        elif metric == 'lpips_loss':
                            lpips_loss.append(value)
                        elif metric == 'total_loss':
                            total_loss.append(value)
                        elif metric == 'match_rate':
                            match_rate.append(value)
    
    # Print debugging information
    print(f"Number of iterations found: {len(iterations)}")
    print(f"Number of key_loss values: {len(key_loss)}")
    print(f"Number of lpips_loss values: {len(lpips_loss)}")
    print(f"Number of total_loss values: {len(total_loss)}")
    print(f"Number of match_rate values: {len(match_rate)}")
    
    if len(iterations) == 0:
        print("Warning: No iterations found in the log file!")
        print("First few lines of the log file:")
        with open(log_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:  # Print first 5 lines
                    print(line.strip())
                else:
                    break
    
    return iterations, key_loss, lpips_loss, total_loss, match_rate

def average_metrics(iterations, metrics, window_size=100):
    # Convert to numpy arrays for easier manipulation
    iterations = np.array(iterations)
    metrics = np.array(metrics)
    
    # Calculate number of windows
    n_windows = len(iterations) // window_size
    
    print(f"Number of windows for averaging: {n_windows}")
    
    # Initialize arrays for averaged values
    avg_iterations = []
    avg_metrics = []
    
    # Calculate averages for each window
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        avg_iterations.append(np.mean(iterations[start_idx:end_idx]))
        avg_metrics.append(np.mean(metrics[start_idx:end_idx]))
    
    return np.array(avg_iterations), np.array(avg_metrics)

def create_plot(iterations, metrics, title, ylabel, color, save_path):
    if len(iterations) == 0 or len(metrics) == 0:
        print(f"Warning: No data to plot for {title}")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, metrics, color=color, linewidth=2)
    
    # Set style
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Iterations', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Customize ticks
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot training metrics from a log file.')
    parser.add_argument('--log_file', type=str, default='train_rank0.log',
                        help='Path to the log file (default: train_rank0.log)')
    args = parser.parse_args()
    
    # Create plots directory if it doesn't exist
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    # Parse log file
    iterations, key_loss, lpips_loss, total_loss, match_rate = parse_log_file(args.log_file)
    
    if len(iterations) == 0:
        print("Error: No data found in the log file. Please check the file path and format.")
        return
    
    # Average metrics
    avg_iterations, avg_key_loss = average_metrics(iterations, key_loss)
    _, avg_lpips_loss = average_metrics(iterations, lpips_loss)
    _, avg_total_loss = average_metrics(iterations, total_loss)
    _, avg_match_rate = average_metrics(iterations, match_rate)
    
    # Define colors and create plots
    colors = {
        'key_loss': '#FF6B6B',
        'lpips_loss': '#4ECDC4',
        'total_loss': '#45B7D1',
        'match_rate': '#96CEB4'
    }
    
    # Create individual plots
    create_plot(avg_iterations, avg_key_loss, 'Key Loss Over Training', 'Key Loss', colors['key_loss'], plots_dir / 'key_loss.png')
    create_plot(avg_iterations, avg_lpips_loss, 'LPIPS Loss Over Training', 'LPIPS Loss', colors['lpips_loss'], plots_dir / 'lpips_loss.png')
    create_plot(avg_iterations, avg_total_loss, 'Total Loss Over Training', 'Total Loss', colors['total_loss'], plots_dir / 'total_loss.png')
    create_plot(avg_iterations, avg_match_rate, 'Match Rate Over Training', 'Match Rate (%)', colors['match_rate'], plots_dir / 'match_rate.png')
    
    # Create subplot
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Key Loss
    plt.subplot(2, 2, 1)
    plt.plot(avg_iterations, avg_key_loss, color=colors['key_loss'], linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel('Iterations', fontsize=10)
    plt.ylabel('Key Loss', fontsize=10)
    plt.title('Key Loss', fontsize=12, pad=10)
    
    # Plot 2: LPIPS Loss
    plt.subplot(2, 2, 2)
    plt.plot(avg_iterations, avg_lpips_loss, color=colors['lpips_loss'], linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel('Iterations', fontsize=10)
    plt.ylabel('LPIPS Loss', fontsize=10)
    plt.title('LPIPS Loss', fontsize=12, pad=10)
    
    # Plot 3: Total Loss
    plt.subplot(2, 2, 3)
    plt.plot(avg_iterations, avg_total_loss, color=colors['total_loss'], linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel('Iterations', fontsize=10)
    plt.ylabel('Total Loss', fontsize=10)
    plt.title('Total Loss', fontsize=12, pad=10)
    
    # Plot 4: Match Rate
    plt.subplot(2, 2, 4)
    plt.plot(avg_iterations, avg_match_rate, color=colors['match_rate'], linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel('Iterations', fontsize=10)
    plt.ylabel('Match Rate (%)', fontsize=10)
    plt.title('Match Rate', fontsize=12, pad=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(plots_dir / 'all_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to {plots_dir / 'all_metrics.png'}")

if __name__ == '__main__':
    main() 