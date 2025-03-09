import math
from scipy.stats import norm

def calculate_auc(mu_positive, std_positive, mu_attacked, std_attacked):
    """
    Calculate the AUC (probability that a sample from the positive distribution 
    scores higher than a sample from the attacked distribution) assuming both distributions 
    are Gaussian.

    Parameters:
        mu_positive (float): Mean of the positive distribution.
        std_positive (float): Standard deviation of the positive distribution.
        mu_attacked (float): Mean of the attacked (or negative) distribution.
        std_attacked (float): Standard deviation of the attacked distribution.

    Returns:
        float: Estimated AUC value.
    """
    # Compute the z-score for the difference in means normalized by the combined variance.
    z = (mu_positive - mu_attacked) / math.sqrt(std_positive**2 + std_attacked**2)
    auc = norm.cdf(z)
    return auc

# Example usage:
mu_positive = 0.9235
std_positive = 0.1092
mu_negative = 0.473
std_negative = 0.330

auc_estimate = calculate_auc(mu_positive, std_positive, mu_negative, std_negative)
print(f"Estimated AUC: {auc_estimate:.3f}")

import math
from scipy.stats import norm

def tpr_at_fpr(mu_positive, std_positive, mu_negative, std_negative, target_fpr=0.01):
    """
    Compute the True Positive Rate (TPR) at a specified False Positive Rate (FPR)
    for two Gaussian distributions: one for positives and one for negatives.
    
    Parameters:
        mu_positive (float): Mean of the positive distribution.
        std_positive (float): Standard deviation of the positive distribution.
        mu_negative (float): Mean of the negative (attacked) distribution.
        std_negative (float): Standard deviation of the negative distribution.
        target_fpr (float): The target false positive rate (default is 0.01, i.e., 1%).
    
    Returns:
        tuple: (tpr, threshold) where tpr is the True Positive Rate at the given FPR,
               and threshold is the decision threshold used.
    """
    # Determine the threshold for negatives such that FPR = target_fpr
    # FPR is defined as P(x > threshold | negatives), so we set:
    # norm.cdf(threshold, loc=mu_negative, scale=std_negative) = 1 - target_fpr.
    threshold = norm.ppf(1 - target_fpr, loc=mu_negative, scale=std_negative)
    
    # Calculate the TPR on the positive distribution:
    # TPR is P(x > threshold | positives)
    tpr = 1 - norm.cdf(threshold, loc=mu_positive, scale=std_positive)
    
    return tpr, threshold

# Example usage:
target_fpr = 0.01   # 1% FPR

tpr, threshold = tpr_at_fpr(mu_positive, std_positive, mu_negative, std_negative, target_fpr)
print(f"Threshold for FPR=1%: {threshold:.3f}")
print(f"TPR at FPR=1%: {tpr:.3f}")
