import numpy as np
import matplotlib.pyplot as plt
import os

def create_matrix_images():
    # Create the figure directory if it doesn't exist
    if not os.path.exists('figure'):
        os.makedirs('figure')
    
    # Generate 5 matrices of size 16x32
    print("Generating 5 matrices of size 16x32...")
    for i in range(5):
        # Generate random matrix with values between 0 and 1, then scale to 0.3-1.0
        matrix_16x32 = np.random.rand(16, 32)
        matrix_16x32 = 0.3 + matrix_16x32 * 0.7  # Scale to range 0.3 to 1.0
        
        # Create and save the image
        plt.figure(figsize=(8, 4))
        plt.imshow(matrix_16x32, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')  # Remove axes, ticks, and labels
        
        # Save the image
        filename = f'figure/matrix_16x32_{i+1}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {filename}")
    
    # Generate 5 matrices of size 4x4
    print("\nGenerating 5 matrices of size 4x4...")
    for i in range(5):
        # Generate random matrix with values between 0 and 1, then scale to 0.3-1.0
        matrix_4x4 = np.random.rand(4, 4)
        matrix_4x4 = 0.3 + matrix_4x4 * 0.7  # Scale to range 0.3 to 1.0
        
        # Create and save the image
        plt.figure(figsize=(4, 4))
        plt.imshow(matrix_4x4, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')  # Remove axes, ticks, and labels
        
        # Save the image
        filename = f'figure/matrix_4x4_{i+1}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {filename}")
    
    print(f"\nAll matrices have been generated and saved in the 'figure' folder!")
    
    # Optional: Print some statistics about the generated matrices
    print("\nMatrix statistics:")
    print("- Values range from 0.3 to 1.0 (70% lighter black to white)")
    print("- 16x32 matrices: 5 images saved")
    print("- 4x4 matrices: 5 images saved")
    print("- Clean images with no decorative elements")

if __name__ == "__main__":
    create_matrix_images() 