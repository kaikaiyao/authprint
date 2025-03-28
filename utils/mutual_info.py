"""
Utilities for mutual information estimation using k-NN entropy estimation.
"""
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import logging
from typing import Tuple


def knn_entropy_estimation(X: np.ndarray, k: int = 3, norm: str = 'chebyshev') -> float:
    """
    Estimate the entropy of a continuous random variable using k-NN method.
    
    Args:
        X (np.ndarray): Data matrix of shape (n_samples, n_features)
        k (int): Number of nearest neighbors to use
        norm (str): Distance metric to use ('chebyshev' recommended for high dimensions)
        
    Returns:
        float: Estimated entropy H(X)
    """
    n_samples, n_dim = X.shape
    
    # Find k-nearest neighbors for each point
    knn = NearestNeighbors(n_neighbors=k+1, metric=norm)  # k+1 because point is its own neighbor
    knn.fit(X)
    distances, _ = knn.kneighbors(X)
    
    # Get the distance to the k-th neighbor (exclude self)
    epsilon = distances[:, k]
    
    # Compute entropy estimate using Kozachenko-Leonenko estimator
    # H(X) ≈ log(n-1) + log(V_d) + d/n * sum(log(epsilon_i)) + euler_constant
    # where V_d is volume of d-dimensional unit ball (constant)
    euler_constant = 0.5772156649
    entropy = np.log(n_samples - 1) + n_dim * np.log(2) + np.mean(np.log(epsilon)) + euler_constant
    
    return float(entropy)


def estimate_mutual_information(
    features: torch.Tensor,
    images: torch.Tensor,
    n_samples: int = 1000,
    k: int = 3,
    device: torch.device = 'cpu'
) -> Tuple[float, float, float, float, float]:
    """
    Estimate mutual information I(features; images) and conditional entropy H(features|images)
    using k-NN entropy estimation.
    Uses the relations:
    - I(X;Y) = H(X) + H(Y) - H(X,Y)
    - H(X|Y) = H(X,Y) - H(Y)
    
    Args:
        features (torch.Tensor): Selected pixel features tensor [n_samples, n_features]
        images (torch.Tensor): Full images tensor [n_samples, channels, height, width]
        n_samples (int): Number of samples to use for estimation
        k (int): Number of nearest neighbors for entropy estimation
        device (torch.device): Device to use for computation
        
    Returns:
        Tuple[float, float, float, float, float]: 
            (mutual_info, h_features, h_images, h_joint, h_features_given_images)
    """
    # Move tensors to CPU for numpy processing
    features = features.detach().cpu().numpy()
    images = images.detach().cpu().numpy()
    
    # Reshape images to 2D
    images_flat = images.reshape(images.shape[0], -1)
    
    # Normalize data to [0,1] range for each dimension
    features = (features - features.min(0)) / (features.max(0) - features.min(0) + 1e-8)
    images_flat = (images_flat - images_flat.min(0)) / (images_flat.max(0) - images_flat.min(0) + 1e-8)
    
    # Stack features and images for joint distribution
    joint = np.hstack([features, images_flat])
    
    # Estimate entropies
    h_features = knn_entropy_estimation(features, k=k)
    h_images = knn_entropy_estimation(images_flat, k=k)
    h_joint = knn_entropy_estimation(joint, k=k)
    
    # Calculate mutual information
    mutual_info = h_features + h_images - h_joint
    
    # Calculate conditional entropy H(features|images)
    h_features_given_images = h_joint - h_images
    
    return mutual_info, h_features, h_images, h_joint, h_features_given_images 