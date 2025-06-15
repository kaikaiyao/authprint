import re
import matplotlib.pyplot as plt
import numpy as np

# Read the loss file
with open('loss.txt', 'r') as f:
    lines = f.readlines()

# Extract training loss values
iterations = []
train_losses = []

for line in lines:
    # Look for lines containing training loss information
    match = re.search(r'Iteration (\d+)/\d+ .* Train Loss: ([\d.]+)', line)
    if match:
        iteration = int(match.group(1))
        train_loss = float(match.group(2))
        iterations.append(iteration)
        train_losses.append(train_loss)

# Create the plot
plt.figure(figsize=(12, 12))
plt.plot(iterations, train_losses, 'b-', label='Train Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True)
plt.legend()
plt.ylim(bottom=0.0, top=0.1)  # Set y-axis range from 0.0 to 0.25

# Save the plot
plt.savefig('train_loss.png', dpi=300, bbox_inches='tight')
plt.close() 