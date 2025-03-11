import os
import gc
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from models.stylegan2 import is_stylegan2
from models.stylegan2 import load_stylegan2_model
from utils.image_utils import constrain_image, downsample_and_upsample, \
    apply_jpeg_compression, quantize_model_weights, apply_truncation
from utils.file_utils import generate_time_based_string
from key.key import generate_mask_secret_key, mask_image_with_key
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_curve, roc_auc_score
from utils.logging import LogRankFilter

def train_surrogate_decoder(
    attack_type: str,
    surrogate_decoder: nn.Module,
    gan_model: nn.Module,
    watermarked_model: nn.Module,
    max_delta: float,
    latent_dim: int,
    device: torch.device,
    train_size: int,
    epochs: int = 5,
    batch_size: int = 16,
    rank: int = 0,
    world_size: int = 1,
    saving_path: str = "results",
) -> None:
    """
    Train the surrogate decoder using a loss function similar to the real decoder's training objective.
    """
    # Configure logging to filter based on rank
    root_logger = logging.getLogger()
    
    # First remove existing rank filters if any
    for handler in root_logger.handlers:
        for filter in handler.filters[:]:
            if isinstance(filter, LogRankFilter):
                handler.removeFilter(filter)
        
        # Add our rank filter to each handler
        handler.addFilter(LogRankFilter(rank))
        
    time_string = generate_time_based_string()
    if rank == 0:
        logging.info(f"time_string = {time_string}")

    torch.manual_seed(2024 + rank)
    surrogate_decoder.train()
    surrogate_decoder.to(device)
    gan_model.eval()
    gan_model.to(device)
    watermarked_model.eval()
    watermarked_model.to(device)

    optimizer = torch.optim.Adagrad(surrogate_decoder.parameters(), lr=0.0001)
    is_stylegan2_model = is_stylegan2(gan_model)
    num_batches = (train_size + batch_size - 1) // batch_size

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_norm_diff = 0.0
        num_samples = 0

        for batch_idx in range(num_batches):
            with torch.no_grad():
                z = torch.randn(batch_size, latent_dim, device=device)
                if not is_stylegan2_model:
                    z = z.view(batch_size, latent_dim, 1, 1)
                if is_stylegan2_model:
                    x_M = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
                    x_M_hat = watermarked_model(z, None, truncation_psi=1.0, noise_mode="const")
                else:
                    x_M = gan_model(z)
                    x_M_hat = watermarked_model(z)
                x_M_hat = constrain_image(x_M_hat, x_M, max_delta)

            k_M = surrogate_decoder(x_M)
            k_M_hat = surrogate_decoder(x_M_hat)
            d_k_M = torch.norm(k_M, dim=1)
            d_k_M_hat = torch.norm(k_M_hat, dim=1)
            norm_diff = d_k_M - d_k_M_hat
            loss = ((norm_diff).max() + 1) ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size
            epoch_norm_diff += norm_diff.mean().item() * batch_size
            num_samples += batch_size

            if rank == 0 and batch_idx % 10 == 0:
                logging.info(
                    f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{num_batches}, "
                    f"Loss: {loss.item():.4f}, Norm diff: {norm_diff.mean().item():.4f}, "
                    f"d_k_M range: [{d_k_M.min().item():.4f}, {d_k_M.max().item():.4f}], "
                    f"d_k_M_hat range: [{d_k_M_hat.min().item():.4f}, {d_k_M_hat.max().item():.4f}]"
                )

            del z, x_M, x_M_hat, k_M, k_M_hat, d_k_M, d_k_M_hat, norm_diff, loss
            torch.cuda.empty_cache()
            gc.collect()

        if world_size > 1:
            local_avg_loss = epoch_loss / num_samples
            local_avg_loss_tensor = torch.tensor(local_avg_loss, device=device)
            dist.all_reduce(local_avg_loss_tensor, op=dist.ReduceOp.SUM)
            global_avg_loss = local_avg_loss_tensor.item() / world_size

            local_avg_norm_diff = epoch_norm_diff / num_samples
            local_avg_norm_diff_tensor = torch.tensor(local_avg_norm_diff, device=device)
            dist.all_reduce(local_avg_norm_diff_tensor, op=dist.ReduceOp.SUM)
            global_avg_norm_diff = local_avg_norm_diff_tensor.item() / world_size
        else:
            global_avg_loss = epoch_loss / num_samples
            global_avg_norm_diff = epoch_norm_diff / num_samples

        if rank == 0:
            logging.info(
                f"Epoch {epoch + 1}/{epochs} Completed. "
                f"Average Loss: {global_avg_loss:.4f}, Average Norm Difference: {global_avg_norm_diff:.4f}"
            )

        torch.cuda.empty_cache()
        gc.collect()

    if rank == 0:
        model_filename = os.path.join(saving_path, f"surrogate_decoder_{time_string}.pt")
        if world_size > 1:
            torch.save(surrogate_decoder.module.state_dict(), model_filename)
        else:
            torch.save(surrogate_decoder.state_dict(), model_filename)
        logging.info(f"Surrogate Decoder model saved as {model_filename}")

def generate_attack_images(
    gan_model: nn.Module,
    image_attack_size: int,
    latent_dim: int,
    device: torch.device,
    batch_size: int = 100,
    attack_image_type: str = "original_image",
) -> torch.Tensor:
    """
    Generate images for the attack.
    
    Args:
        gan_model: The GAN model to use for generating images
        image_attack_size: Number of images to generate
        latent_dim: Dimension of the latent space
        device: Device to use for computation
        batch_size: Batch size for generation
        attack_image_type: Type of images to generate ("original_image" or "random_image")
    
    Returns:
        torch.Tensor: Generated attack images
    """
    # Input validation
    if attack_image_type not in ["original_image", "random_image"]:
        raise ValueError(f"Invalid attack_image_type: {attack_image_type}. Must be 'original_image' or 'random_image'")
    
    if image_attack_size <= 0:
        raise ValueError(f"image_attack_size must be positive, got {image_attack_size}")
    
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    
    # Ensure batch_size doesn't exceed image_attack_size
    batch_size = min(batch_size, image_attack_size)
    
    image_attack_batches = []
    num_batches = math.ceil(image_attack_size / batch_size)
    gan_model.to(device)

    with torch.no_grad():
        # Get image shape from GAN model by doing a single forward pass
        if is_stylegan2(gan_model):
            z = torch.randn((1, latent_dim), device=device)
            sample_image = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
        else:
            z = torch.randn(1, latent_dim, 1, 1, device=device)
            sample_image = gan_model(z)
        image_shape = sample_image.shape[1:]
        del z, sample_image

        for batch_idx in range(num_batches):
            logging.info(f"Generating attack images, batch {batch_idx+1}/{num_batches}")
            current_batch_size = min(batch_size, image_attack_size - batch_idx * batch_size)
            
            if attack_image_type == "original_image":
                # Generate images using the GAN model
                if is_stylegan2(gan_model):
                    z = torch.randn((current_batch_size, latent_dim), device=device)
                    x_M = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
                else:
                    z = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
                    x_M = gan_model(z)
                
                image_attack_batches.append(x_M.cpu())
                del z, x_M
            else:  # random_image
                # Generate random noise images with same shape as GAN output
                random_images = torch.rand((current_batch_size,) + image_shape, device=device) * 2 - 1  # Scale to [-1, 1]
                image_attack_batches.append(random_images.cpu())
                del random_images
            
            torch.cuda.empty_cache()
            gc.collect()

    image_attack = torch.cat(image_attack_batches).to(device)
    del image_attack_batches
    torch.cuda.empty_cache()
    gc.collect()
    return image_attack

def generate_initial_perturbations(
    surrogate_decoder: nn.Module,
    decoder: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    num_steps: int = 50,  # Reduced steps
    alpha: float = 0.2,  # Increased step size
    max_delta: float = 2.0,
    early_stop_threshold: float = 1e-6
) -> tuple:
    surrogate_decoder.eval()
    decoder.eval()
    images = images.clone().detach().to(device)
    images.requires_grad = True
    original_images = images.clone().detach()
    
    # Get initial k_scores from real decoder
    with torch.no_grad():
        k_orig = decoder(original_images)
        k_scores_orig = torch.norm(k_orig, dim=1)
        initial_direction = F.normalize(k_orig, dim=1)
        initial_norm = k_scores_orig.mean()

    best_perturbation = None
    best_k_scores = None
    min_k_score = float('inf')
    prev_k_score = float('inf')
    plateau_count = 0

    for step in range(num_steps):
        if images.grad is not None:
            images.grad.zero_()
        
        # Get surrogate decoder's output
        k_surrogate = surrogate_decoder(images)
        k_surrogate_norm = torch.norm(k_surrogate, dim=1)
        
        # Compute real decoder output for loss calculation
        with torch.no_grad():
            k_real = decoder(images)
            k_real_norm = torch.norm(k_real, dim=1)
            current_direction = F.normalize(k_real, dim=1)
        
        # Combined loss to maximize norm reduction and direction change
        norm_reduction = F.relu(k_real_norm - k_scores_orig.mean())  # Encourage norm reduction
        direction_change = 1 - F.cosine_similarity(current_direction, initial_direction, dim=1)
        loss = -(norm_reduction.mean() + direction_change.mean())
        
        loss.backward()
        
        with torch.no_grad():
            # Normalize gradients for stable updates
            grad_norm = torch.norm(images.grad.view(images.size(0), -1), dim=1).view(-1, 1, 1, 1).clamp(min=1e-8)
            normalized_grad = images.grad / grad_norm
            
            # Update images with normalized gradient
            images = images - alpha * normalized_grad
            
            # Project back to valid perturbation range
            delta = images - original_images
            delta = torch.clamp(delta, -max_delta, max_delta)
            images = original_images + delta
            images = torch.clamp(images, -1, 1)
            images.requires_grad = True
            
            # Evaluate real decoder's output
            k_pert = decoder(images)
            k_scores_pert = torch.norm(k_pert, dim=1)
            
            # Keep track of best perturbation
            current_k_score = k_scores_pert.mean().item()
            if current_k_score < min_k_score:
                min_k_score = current_k_score
                best_perturbation = images.clone().detach()
                best_k_scores = k_scores_pert.clone().detach()
            
            # Early stopping check
            if abs(current_k_score - prev_k_score) < early_stop_threshold:
                plateau_count += 1
                if plateau_count >= 5:
                    break
            else:
                plateau_count = 0
            prev_k_score = current_k_score
            
            if step % 10 == 0:
                logging.info(
                    f"PGD Step {step}, "
                    f"Original k_scores: {k_scores_orig.mean().item():.4f} "
                    f"[{k_scores_orig.min().item():.4f}, {k_scores_orig.max().item():.4f}], "
                    f"Current k_scores: {k_scores_pert.mean().item():.4f} "
                    f"[{k_scores_pert.min().item():.4f}, {k_scores_pert.max().item():.4f}], "
                    f"Direction Change: {direction_change.mean().item():.4f}, "
                    f"Best k_score: {min_k_score:.4f}"
                )

    return best_perturbation, k_scores_orig, best_k_scores

def fine_tune_surrogate(
    surrogate_decoder: nn.Module,
    decoder: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 16,
    rank: int = 0
) -> nn.Module:
    """Fine-tune surrogate on perturbed images labeled by real decoder to match output patterns."""
    surrogate_decoder.train()
    decoder.eval()
    
    # Stronger regularization and lower learning rate
    optimizer = torch.optim.Adam(surrogate_decoder.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    num_batches = (images.size(0) + batch_size - 1) // batch_size
    best_loss = float('inf')
    best_model_state = None
    
    # Initialize EMA of real decoder norms
    ema_norm = 0.0
    ema_alpha = 0.1

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_samples = 0

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, images.size(0))
            batch = images[start:end].to(device)
            
            # Generate perturbed images with stronger perturbations
            perturbed_batch, k_scores_orig, k_scores_pert = generate_initial_perturbations(
                surrogate_decoder=surrogate_decoder,
                decoder=decoder,
                images=batch,
                device=device,
                num_steps=50,
                alpha=0.2,
                max_delta=2.0
            )
            
            # Get outputs from both decoders
            with torch.no_grad():
                k_real = decoder(perturbed_batch)
                k_real_orig = decoder(batch)
                
                # Update EMA of real decoder norms
                batch_norm = (torch.norm(k_real, dim=1).mean() + torch.norm(k_real_orig, dim=1).mean()) / 2
                ema_norm = ema_norm * (1 - ema_alpha) + batch_norm.item() * ema_alpha
                
                # Normalize real decoder outputs
                k_real_norm = F.normalize(k_real, dim=1)
                k_real_orig_norm = F.normalize(k_real_orig, dim=1)
            
            # Get surrogate outputs
            k_surrogate = surrogate_decoder(perturbed_batch)
            k_surrogate_orig = surrogate_decoder(batch)
            
            # Normalize surrogate outputs
            k_surrogate_norm = F.normalize(k_surrogate, dim=1)
            k_surrogate_orig_norm = F.normalize(k_surrogate_orig, dim=1)
            
            # 1. Direction alignment loss (primary objective)
            direction_loss = (1 - F.cosine_similarity(k_real_norm, k_surrogate_norm, dim=1)).mean() + \
                           (1 - F.cosine_similarity(k_real_orig_norm, k_surrogate_orig_norm, dim=1)).mean()
            
            # 2. Scale matching loss with EMA norm target
            scale_loss = (F.mse_loss(torch.norm(k_surrogate, dim=1), torch.norm(k_real, dim=1)) + \
                         F.mse_loss(torch.norm(k_surrogate_orig, dim=1), torch.norm(k_real_orig, dim=1)))
            
            # 3. Feature matching loss (reduced weight)
            feature_loss = F.mse_loss(k_surrogate / (torch.norm(k_surrogate, dim=1, keepdim=True) + 1e-8),
                                    k_real / (torch.norm(k_real, dim=1, keepdim=True) + 1e-8)) + \
                         F.mse_loss(k_surrogate_orig / (torch.norm(k_surrogate_orig, dim=1, keepdim=True) + 1e-8),
                                   k_real_orig / (torch.norm(k_real_orig, dim=1, keepdim=True) + 1e-8))
            
            # Combined loss with adjusted weights
            loss = direction_loss + 0.1 * scale_loss + 0.01 * feature_loss
            
            # Add L2 regularization to prevent collapse
            l2_reg = sum(torch.norm(p) ** 2 for p in surrogate_decoder.parameters())
            loss = loss + 0.001 * l2_reg
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(surrogate_decoder.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)
            num_samples += batch.size(0)

            if rank == 0 and i % 10 == 0:
                logging.info(
                    f"Fine-tune Epoch {epoch+1}, Batch {i+1}/{num_batches}, "
                    f"Total Loss: {loss.item():.4f}, "
                    f"Direction Loss: {direction_loss.item():.4f}, "
                    f"Scale Loss: {scale_loss.item():.4f}, "
                    f"Feature Loss: {feature_loss.item():.4f}, "
                    f"L2 Reg: {l2_reg.item():.4f}, "
                    f"Real Norm EMA: {ema_norm:.4f}, "
                    f"Surrogate Norm Range: [{torch.norm(k_surrogate, dim=1).min().item():.4f}, "
                    f"{torch.norm(k_surrogate, dim=1).max().item():.4f}], "
                    f"PGD k_scores - Original: {k_scores_orig.mean().item():.4f}, "
                    f"Perturbed: {k_scores_pert.mean().item():.4f}"
                )

            del batch, perturbed_batch, k_real, k_real_orig, k_surrogate, k_surrogate_orig
            torch.cuda.empty_cache()
            gc.collect()

        avg_epoch_loss = epoch_loss / num_samples
        if rank == 0:
            logging.info(
                f"Fine-tune Epoch {epoch + 1}/{epochs} Completed. "
                f"Average Loss: {avg_epoch_loss:.4f}"
            )
        
        scheduler.step(avg_epoch_loss)
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_state = surrogate_decoder.state_dict().copy()

        torch.cuda.empty_cache()
        gc.collect()

    # Restore best model
    if best_model_state is not None:
        surrogate_decoder.load_state_dict(best_model_state)

    return surrogate_decoder

def perform_pgd_attack(
    attack_type: str,
    surrogate_decoders: list,
    decoder: nn.Module,
    image_attack: torch.Tensor,
    max_delta: float,
    device: torch.device,
    num_steps: int,
    alpha_values: list,
    attack_batch_size: int = 10,
    momentum: float = 0.9,
    ensemble_weights: list = None,
    key_type: str = "csprng",
    return_details: bool = False
) -> tuple:
    """
    Perform PGD attack with optional momentum using multiple surrogate decoders.
    
    Args:
        surrogate_decoders: List of surrogate decoder models
        ensemble_weights: List of weights for each surrogate decoder (default: equal weights)
        return_details: Whether to return detailed attack results
        
    Returns:
        If return_details=False: (k_attack_scores_mean, k_attack_scores_std)
        If return_details=True: (k_attack_scores_mean, k_attack_scores_std, attacked_images, attack_scores)
    """
    # Input validation
    if len(surrogate_decoders) == 0:
        raise ValueError("Must provide at least one surrogate decoder")
    
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    
    if not alpha_values:
        raise ValueError("alpha_values cannot be empty")
        
    if attack_batch_size <= 0:
        raise ValueError(f"attack_batch_size must be positive, got {attack_batch_size}")
    
    # Ensure attack_batch_size doesn't exceed image size
    attack_batch_size = min(attack_batch_size, image_attack.size(0))
    
    for surrogate_decoder in surrogate_decoders:
        surrogate_decoder.eval()
        for param in surrogate_decoder.parameters():
            param.requires_grad = False
    
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False

    # Initialize ensemble weights if not provided
    if ensemble_weights is None:
        ensemble_weights = [1.0 / len(surrogate_decoders)] * len(surrogate_decoders)
    else:
        # Validate ensemble weights
        if len(ensemble_weights) != len(surrogate_decoders):
            raise ValueError(f"ensemble_weights length ({len(ensemble_weights)}) must match surrogate_decoders length ({len(surrogate_decoders)})")
        
        # Normalize ensemble weights to sum to 1
        weight_sum = sum(ensemble_weights)
        if weight_sum <= 0:
            raise ValueError(f"Sum of ensemble_weights must be positive, got {weight_sum}")
        
        if abs(weight_sum - 1.0) > 1e-6:
            logging.warning(f"Normalizing ensemble_weights to sum to 1.0 (was {weight_sum})")
            ensemble_weights = [w / weight_sum for w in ensemble_weights]
    
    criterion = nn.BCELoss()
    k_attack_scores_mean = []
    k_attack_scores_std = []
    
    # If we need detailed results, keep track of the best attacked images and scores
    if return_details:
        best_attacked_images = None
        best_attack_scores = None
        best_mean_score = -float('inf')  # We want to maximize score

    # Test different alpha values for PGD attack
    for alpha_idx, alpha in enumerate(alpha_values):
        image_attack_batches = []
        k_attack_scores_batches = []
        
        for batch_idx in range(0, image_attack.size(0), attack_batch_size):
            logging.info(f"Processing batch {batch_idx // attack_batch_size + 1}/{math.ceil(image_attack.size(0) / attack_batch_size)}")
            batch_end = min(batch_idx + attack_batch_size, image_attack.size(0))
            batch_size = batch_end - batch_idx
            batch_attack = image_attack[batch_idx:batch_end].clone().detach().to(device)

            # Apply PGD attack
            best_adv_images, best_adv_scores = None, None
            best_score = -float('inf')

            # Initialize perturbation with zeros or from previous perturbation if available
            perturbation = torch.zeros_like(batch_attack, requires_grad=True)
            
            # Initialize momentum buffer
            if momentum > 0:
                grad_momentum = torch.zeros_like(batch_attack)
                
            # Ensure the decoder is in eval mode
            decoder.eval()
            
            # Perform PGD steps
            for step in range(num_steps):
                perturbation.requires_grad_(True)

                adv_images = batch_attack + perturbation
                
                # Ensure adversarial images are within valid range [-1, 1]
                adv_images = torch.clamp(adv_images, -1, 1)
                
                # Forward pass through each surrogate decoder
                ensemble_score = 0
                for i, surrogate_decoder in enumerate(surrogate_decoders):
                    surrogate_out = surrogate_decoder(adv_images)
                    # Weighted ensemble of surrogate decoders
                    ensemble_score += ensemble_weights[i] * surrogate_out

                # Compute loss - we want to maximize the decoder score to fool the detector
                # BCELoss expects target to be in [0, 1]
                target = torch.ones_like(ensemble_score)
                loss = criterion(ensemble_score, target)
                
                grad = torch.autograd.grad(loss, perturbation)[0]
                
                # Apply momentum if specified
                if momentum > 0:
                    grad_momentum = momentum * grad_momentum + grad
                    update = alpha * grad_momentum.sign()
                else:
                    update = alpha * grad.sign()
                
                # Apply update
                with torch.no_grad():
                    perturbation -= update # right direction
                    
                    # Project perturbation to ensure max_delta constraint
                    perturbation.data = torch.clamp(perturbation.data, -max_delta, max_delta)
                    
                    # Ensure images stay in valid range [-1, 1]
                    adv_images = torch.clamp(batch_attack + perturbation, -1, 1)
                    
                    # Recompute perturbation based on clamped adv_images
                    perturbation.data = adv_images - batch_attack

                # Evaluate current attack with real decoder
                if step % 50 == 0 or step == num_steps - 1:
                    with torch.no_grad():
                        # Forward pass through real decoder
                        if key_type != "none":
                            # Apply masking if enabled
                            k_mask = generate_mask_secret_key(adv_images.shape, 2024, device, key_type=key_type)
                            masked_images = mask_image_with_key(images=adv_images, cnn_key=k_mask)
                            real_scores = decoder(masked_images)
                        else:
                            real_scores = decoder(adv_images)

                        avg_score = real_scores.mean().item()
                        
                        if avg_score > best_score:
                            best_score = avg_score
                            best_adv_images = adv_images.clone().detach().cpu()
                            best_adv_scores = real_scores.clone().detach().cpu()
                            
                            # Log progress for significant improvements
                            if step > 0 and step % 50 == 0:
                                logging.info(f"Step {step}/{num_steps}, Alpha = {alpha}: Current best score = {best_score:.3f}")
            
            # Add batch results to overall results
            image_attack_batches.append(best_adv_images)
            k_attack_scores_batches.append(best_adv_scores)
            
            # Clean up GPU memory
            del perturbation, adv_images
            torch.cuda.empty_cache()
            gc.collect()
        
        # Combine batches for this alpha value
        attacked_images = torch.cat(image_attack_batches).to(device)
        k_attack_scores_alpha = torch.cat(k_attack_scores_batches).to(device)
        
        # Calculate statistics
        mean_score = k_attack_scores_alpha.mean().item()
        std_score = k_attack_scores_alpha.std().item()
        
        k_attack_scores_mean.append(mean_score)
        k_attack_scores_std.append(std_score)
        
        # Update best overall results if needed
        if return_details and mean_score > best_mean_score:
            best_mean_score = mean_score
            best_attacked_images = attacked_images.clone().detach().cpu()
            best_attack_scores = k_attack_scores_alpha.flatten().detach().cpu().numpy().tolist()
        
        # Clean up for next alpha
        del attacked_images, k_attack_scores_alpha
        torch.cuda.empty_cache()
        gc.collect()
        
        logging.info(f"Alpha = {alpha}: k_attack_score mean = {mean_score:.3f}, std = {std_score:.3f}")
    
    if return_details:
        return k_attack_scores_mean, k_attack_scores_std, best_attacked_images, best_attack_scores
    return k_attack_scores_mean, k_attack_scores_std

def attack_label_based(
    attack_type: str,
    gan_model: nn.Module,
    watermarked_model: nn.Module,
    max_delta: float,
    decoder: nn.Module,
    surrogate_decoders: list,
    latent_dim: int,
    device: torch.device,
    train_size: int,
    image_attack_size: int,
    batch_size: int = 16,
    epochs: int = 1,
    attack_batch_size: int = 16,
    num_steps: int = 500,
    alpha_values: list = None,
    train_surrogate: bool = True,
    finetune_surrogate: bool = False,
    rank: int = 0,
    world_size: int = 1,
    momentum: float = 0.9,
    key_type: str = "csprng",
    surrogate_training_only: bool = False,
    saving_path: str = "results",
) -> tuple:
    """
    Performs a label-based attack on a watermarked GAN model with optional surrogate fine-tuning.
    Evaluates attack effectiveness across multiple test cases and reports comprehensive metrics.

    Args:
        attack_type (str): Type of attack to perform
        gan_model (nn.Module): Original GAN model
        watermarked_model (nn.Module): Watermarked GAN model to attack
        max_delta (float): Maximum perturbation delta
        decoder (nn.Module): True watermark decoder
        surrogate_decoders (list): List of surrogate decoders
        latent_dim (int): Latent dimension of the GAN
        device (torch.device): Device to run the attack on
        train_size (int): Number of images for training surrogate decoders
        image_attack_size (int): Number of images to attack
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        attack_batch_size (int): Batch size for the attack
        num_steps (int): Number of PGD steps
        alpha_values (list): List of step sizes for PGD
        train_surrogate (bool): Whether to train surrogate decoders
        finetune_surrogate (bool): Whether to fine-tune surrogate decoders
        rank (int): Process rank for distributed training
        world_size (int): Number of processes for distributed training
        momentum (float): Momentum for PGD attack
        key_type (str): Type of key to use for masking
        surrogate_training_only (bool): If True, only perform surrogate training and skip all other steps
        saving_path (str): Path to save models and results
    """
    # Configure logging to filter based on rank
    root_logger = logging.getLogger()
    
    # First remove existing rank filters if any
    for handler in root_logger.handlers:
        for filter in handler.filters[:]:
            if isinstance(filter, LogRankFilter):
                handler.removeFilter(filter)
        
        # Add our rank filter to each handler
        handler.addFilter(LogRankFilter(rank))

    if rank == 0:
        if surrogate_training_only:
            logging.info("="*80)
            logging.info("Starting surrogate decoder training only mode")
            logging.info("="*80)
            logging.info(f"Attack type: {attack_type}")
            logging.info(f"Key type: {key_type}")
            logging.info(f"Number of surrogate decoders: {len(surrogate_decoders)}")
            logging.info("-"*80)
        else:
            logging.info("="*80)
            logging.info("Starting comprehensive attack evaluation on multiple test cases")
            logging.info("="*80)
            logging.info(f"Attack type: {attack_type}")
            logging.info(f"Key type: {key_type}")
            logging.info(f"Number of surrogate decoders: {len(surrogate_decoders)}")
            logging.info("-"*80)

    # Set models to evaluation mode
    gan_model.eval()
    watermarked_model.eval()
    for param in gan_model.parameters():
        param.requires_grad = False
    for param in watermarked_model.parameters():
        param.requires_grad = False

    # If surrogate_training_only mode is enabled, skip to Step 4 (surrogate training)
    if surrogate_training_only:
        if rank == 0:
            logging.info("Surrogate training only mode: skipping to Step 4 (surrogate training)")
            logging.info("Step 4: Training surrogate decoders...")
        
        if train_surrogate:
            for i, surrogate_decoder in enumerate(surrogate_decoders):
                train_surrogate_decoder(
                    attack_type=attack_type,
                    surrogate_decoder=surrogate_decoder,
                    gan_model=gan_model,
                    watermarked_model=watermarked_model,
                    max_delta=max_delta,
                    latent_dim=latent_dim,
                    device=device,
                    train_size=train_size,
                    epochs=epochs,
                    batch_size=batch_size,
                    rank=rank,
                    world_size=world_size,
                    saving_path=saving_path,
                )
                if rank == 0:
                    logging.info(f"  Surrogate decoder {i + 1} training completed")
        else:
            if rank == 0:
                logging.info("  Using pre-trained surrogate decoders")
        
        if rank == 0:
            logging.info("="*80)
            logging.info("Surrogate training completed. Skipping remaining attack steps.")
            logging.info("="*80)
        
        return None, None, None, None
    
    # Generate original watermarked images for reference
    with torch.no_grad():
        if rank == 0:
            logging.info("Step 1: Generating reference watermarked images for threshold calculation (10,000 images)...")
            
        # Use fixed 10,000 images for threshold calculation
        threshold_image_count = 10000
        
        # Generate a set of original watermarked images to calculate threshold
        watermarked_batches = []
        original_batches = []
        num_threshold_batches = math.ceil(threshold_image_count / 100) 
        
        for batch_idx in range(num_threshold_batches):
            current_batch_size = min(100, threshold_image_count - batch_idx * 100)
            if is_stylegan2(gan_model):
                z = torch.randn((current_batch_size, latent_dim), device=device)
                x_original = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
                x_watermarked = watermarked_model(z, None, truncation_psi=1.0, noise_mode="const")
            else:
                z = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
                x_original = gan_model(z)
                x_watermarked = watermarked_model(z)
                
            # Apply delta constraint
            x_watermarked = constrain_image(x_watermarked, x_original, max_delta)
            
            original_batches.append(x_original.cpu())
            watermarked_batches.append(x_watermarked.cpu())
        
        original_images = torch.cat(original_batches).to(device)
        watermarked_images = torch.cat(watermarked_batches).to(device)
        
        # Get watermarking scores for reference
        if rank == 0:
            logging.info("Step 2: Calculating detection scores for reference images...")
            
        original_scores = []
        watermarked_scores = []
        
        for i in range(0, original_images.size(0), attack_batch_size):
            batch_original = original_images[i:i+attack_batch_size]
            batch_watermarked = watermarked_images[i:i+attack_batch_size]
            
            # Apply masking if using key
            if key_type != "none":
                k_mask = generate_mask_secret_key(batch_original.shape, 2024, device, key_type=key_type)
                batch_original = mask_image_with_key(images=batch_original, cnn_key=k_mask)
                
                k_mask = generate_mask_secret_key(batch_watermarked.shape, 2024, device, key_type=key_type)
                batch_watermarked = mask_image_with_key(images=batch_watermarked, cnn_key=k_mask)
            
            # Get scores
            original_output = decoder(batch_original)
            watermarked_output = decoder(batch_watermarked)
            
            original_scores.extend(original_output.cpu().numpy().flatten().tolist())
            watermarked_scores.extend(watermarked_output.cpu().numpy().flatten().tolist())
            
        # Calculate threshold at 0.1% FPR (changed from 1%)
        original_scores_np = np.array(original_scores)
        watermarked_scores_np = np.array(watermarked_scores)
        
        # Check if we have valid scores before proceeding
        if len(original_scores_np) == 0 or len(watermarked_scores_np) == 0:
            raise ValueError("Empty original or watermarked scores arrays")
            
        labels = np.array([0] * len(original_scores_np) + [1] * len(watermarked_scores_np))
        combined_scores = np.concatenate([original_scores_np, watermarked_scores_np])
        
        fpr, tpr, thresholds = roc_curve(labels, combined_scores)
        
        # Find threshold at 0.1% FPR (changed from 1%)
        target_fpr = 0.001
        threshold_candidates = [(fpr[i], thresholds[i]) for i in range(len(fpr)) if fpr[i] <= target_fpr]
        if threshold_candidates:
            threshold = min(threshold_candidates, key=lambda x: x[1])[1]  # Use minimum valid threshold
        else:
            # If no threshold gives exactly 0.1% FPR, take the one closest to it
            threshold_idx = np.argmin(np.abs(fpr - target_fpr))
            threshold = thresholds[threshold_idx]
            
        # Check if the threshold is valid
        if not np.isfinite(threshold):
            # Fallback to a manual threshold calculation if ROC curve fails
            logging.warning("Invalid threshold from ROC curve, using manual calculation")
            orig_max = np.max(original_scores_np)
            wm_min = np.min(watermarked_scores_np)
            # If distributions are well-separated, set threshold between them
            if orig_max < wm_min:
                threshold = (orig_max + wm_min) / 2
            else:
                # Otherwise use a percentile-based approach
                threshold = np.percentile(original_scores_np, 99.9)  # 99.9th percentile for 0.1% FPR
            
        if rank == 0:
            original_auc = roc_auc_score(labels, combined_scores)
            logging.info(f"Reference ROC-AUC: {original_auc:.4f}")
            logging.info(f"Selected threshold at 0.1% FPR: {threshold:.4f}")
            logging.info(f"Mean original score: {original_scores_np.mean():.4f} ± {original_scores_np.std():.4f}")
            logging.info(f"Mean watermarked score: {watermarked_scores_np.mean():.4f} ± {watermarked_scores_np.std():.4f}")
            logging.info("-"*80)

    # Generate different types of attack images using image_attack_size
    if rank == 0:
        logging.info(f"Step 3: Generating test cases for attack using image_attack_size={image_attack_size}...")
    
    # Define num_batches based on image_attack_size for attack test cases
    num_batches = math.ceil(image_attack_size / 100)
    
    test_cases = {
        'original': generate_attack_images(
            gan_model, image_attack_size, latent_dim, device, batch_size=100,
            attack_image_type="original_image"
        ),
        'random': generate_attack_images(
            gan_model, image_attack_size, latent_dim, device, batch_size=100,
            attack_image_type="random_image"
        )
    }
    
    # Generate additional test cases using utility functions
    if is_stylegan2(gan_model):
        if rank == 0:
            logging.info("  Generating truncated images...")
            
        # Case: Truncated images using apply_truncation
        truncated_batches = []
        with torch.no_grad():
            for batch_idx in range(num_batches):
                current_batch_size = min(100, image_attack_size - batch_idx * 100)
                z = torch.randn((current_batch_size, latent_dim), device=device)
                # Use the utility function from image_utils
                x_truncated = apply_truncation(gan_model, z, truncation_psi=0.5)
                truncated_batches.append(x_truncated.cpu())
        test_cases['truncated'] = torch.cat(truncated_batches).to(device)
        
        if rank == 0:
            logging.info("  Generating quantized images...")
            
        # Case: Quantized images using quantize_model_weights
        quantized_batches = []
        with torch.no_grad():
            for batch_idx in range(num_batches):
                current_batch_size = min(100, image_attack_size - batch_idx * 100)
                z = torch.randn((current_batch_size, latent_dim), device=device)
                # Use the utility function from image_utils
                quantized_model = quantize_model_weights(gan_model, precision='int8')
                x_quantized = quantized_model(z, None, truncation_psi=1.0, noise_mode="const")
                quantized_batches.append(x_quantized.cpu())
                del quantized_model
        test_cases['quantized'] = torch.cat(quantized_batches).to(device)
        
        if rank == 0:
            logging.info("  Generating downsampled images...")
            
        # Case: Downsampled images using downsample_and_upsample
        downsampled_batches = []
        with torch.no_grad():
            for batch_idx in range(num_batches):
                current_batch_size = min(100, image_attack_size - batch_idx * 100)
                batch_images = test_cases['original'][batch_idx*100:batch_idx*100+current_batch_size]
                # Use the utility function from image_utils
                batch_downsampled = downsample_and_upsample(batch_images, downsample_size=128)
                downsampled_batches.append(batch_downsampled.cpu())
        test_cases['downsampled'] = torch.cat(downsampled_batches).to(device)
        
        if rank == 0:
            logging.info("  Generating compressed images...")
            
        # Case: Compressed images using apply_jpeg_compression
        compressed_batches = []
        with torch.no_grad():
            for batch_idx in range(num_batches):
                current_batch_size = min(100, image_attack_size - batch_idx * 100)
                batch_images = test_cases['original'][batch_idx*100:batch_idx*100+current_batch_size]
                # Use the utility function from image_utils
                batch_compressed = apply_jpeg_compression(batch_images, quality=55)
                compressed_batches.append(batch_compressed.cpu())
        test_cases['compressed'] = torch.cat(compressed_batches).to(device)
        
        # Case: Alternative models
        model_configs = {
            'ffhq1k': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq1k-paper256-ada.pkl",
                      "ffhq1k-paper256-ada.pkl"),
            'ffhq30k': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq30k-paper256-ada.pkl",
                       "ffhq30k-paper256-ada.pkl"),
            'ffhq70k-bcr': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada-bcr.pkl",
                           "ffhq70k-paper256-ada-bcr.pkl"),
            'ffhq70k-noaug': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-noaug.pkl",
                             "ffhq70k-paper256-noaug.pkl")
        }
        
        # Only rank 0 downloads models
        if rank == 0:
            for model_name, (url, local_path) in model_configs.items():
                try:
                    logging.info(f"  Loading alternative model: {model_name}...")
                    alt_model = load_stylegan2_model(url=url, local_path=local_path, device=device)
                    
                    alt_batches = []
                    for batch_idx in range(num_batches):
                        current_batch_size = min(100, image_attack_size - batch_idx * 100)
                        z = torch.randn((current_batch_size, latent_dim), device=device)
                        alt_images = alt_model(z, None, truncation_psi=1.0, noise_mode="const")
                        alt_batches.append(alt_images.cpu())
                    
                    test_cases[f'alt_{model_name}'] = torch.cat(alt_batches).to(device)
                except Exception as e:
                    logging.warning(f"  Failed to load {model_name}: {str(e)}")
                    continue
        
        # If using distributed training, wait for rank 0 to finish downloading
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    
    if rank == 0:
        logging.info(f"  Generated {len(test_cases)} attack test cases")
        logging.info("-"*80)

    # Train surrogate decoders if specified
    if rank == 0:
        logging.info("Step 4: Training/preparing surrogate decoders...")
        
    if train_surrogate:
        for i, surrogate_decoder in enumerate(surrogate_decoders):
            train_surrogate_decoder(
                attack_type=attack_type,
                surrogate_decoder=surrogate_decoder,
                gan_model=gan_model,
                watermarked_model=watermarked_model,
                max_delta=max_delta,
                latent_dim=latent_dim,
                device=device,
                train_size=train_size,
                epochs=epochs,
                batch_size=batch_size,
                rank=rank,
                world_size=world_size,
                saving_path=saving_path,
            )
            if rank == 0:
                logging.info(f"  Surrogate decoder {i + 1} training completed")
    else:
        if rank == 0:
            logging.info("  Using pre-trained surrogate decoders")

    # Fine-tune surrogate decoders if specified
    if finetune_surrogate and rank == 0:
        logging.info("Step 5: Fine-tuning surrogate decoders...")
        for i, surrogate_decoder in enumerate(surrogate_decoders):
            # Fine-tune on original images for better results
            surrogate_decoder = fine_tune_surrogate(
                surrogate_decoder=surrogate_decoder,
                decoder=decoder,
                images=test_cases['original'],
                device=device,
                epochs=3,  # Adjustable
                batch_size=batch_size,
                rank=rank
            )
            logging.info(f"  Surrogate decoder {i + 1} fine-tuned with real decoder outputs")
    elif rank == 0:
        logging.info("Step 5: Skipping fine-tuning")
        logging.info("-"*80)

    # Perform PGD attack on all test cases
    attack_results = {}
    if rank == 0:
        logging.info("Step 6: Performing PGD attack on all test cases...")
        
        # Collect results for all test cases
        for i, (case_name, case_images) in enumerate(test_cases.items()):
            logging.info(f"  Attacking {case_name} images ({i+1}/{len(test_cases)})...")
            attack_scores_mean, attack_scores_std, attack_images, attack_scores = perform_pgd_attack(
                attack_type=attack_type,
                surrogate_decoders=surrogate_decoders,
                decoder=decoder,
                image_attack=case_images,
                max_delta=max_delta,
                device=device,
                num_steps=num_steps,
                alpha_values=alpha_values,
                attack_batch_size=attack_batch_size,
                momentum=momentum,
                key_type=key_type,
                return_details=True
            )
            
            attack_results[case_name] = {
                'mean': attack_scores_mean,
                'std': attack_scores_std,
                'images': attack_images,
                'scores': attack_scores
            }
        
        logging.info("-"*80)
        logging.info("Step 7: Calculating attack metrics...")
        
        # Calculate combined metrics for overall ROC-AUC
        combined_attack_scores = []
        combined_attack_labels = []
        
        for case_name, result in attack_results.items():
            attack_scores = result['scores']
            combined_attack_scores.extend(attack_scores)
            combined_attack_labels.extend([0] * len(attack_scores))  # All attack images are negatives
        
        # Add watermarked images as positives
        combined_attack_scores.extend(watermarked_scores)
        combined_attack_labels.extend([1] * len(watermarked_scores))
        
        # Calculate overall ROC-AUC
        overall_attack_auc = roc_auc_score(combined_attack_labels, combined_attack_scores)
        
        # Calculate per-case ROC-AUC and ASR@1%FPR
        attack_success_rates = {}
        per_case_auc = {}
        
        for case_name, result in attack_results.items():
            attack_scores = np.array(result['scores'])
            
            # Calculate ASR@1%FPR - percentage of attack images above threshold
            asr = np.mean(attack_scores >= threshold) * 100  # Convert to percentage
            attack_success_rates[case_name] = asr
            
            # Calculate per-case ROC-AUC - compare this attack case vs watermarked
            # Ensure equal number of samples from each class for balanced evaluation
            n_samples = min(len(attack_scores), len(watermarked_scores))
            
            if n_samples > 0:
                # Randomly sample if needed to ensure balance
                if len(attack_scores) > n_samples:
                    np.random.seed(2024)  # For reproducibility
                    attack_idx = np.random.choice(len(attack_scores), n_samples, replace=False)
                    case_attack_scores = attack_scores[attack_idx]
                else:
                    case_attack_scores = attack_scores
                
                if len(watermarked_scores) > n_samples:
                    np.random.seed(2024)  # For reproducibility
                    wm_idx = np.random.choice(len(watermarked_scores), n_samples, replace=False)
                    case_wm_scores = np.array(watermarked_scores)[wm_idx]
                else:
                    case_wm_scores = np.array(watermarked_scores)
                
                # Create labels and combined scores
                case_labels = np.array([0] * len(case_attack_scores) + [1] * len(case_wm_scores))
                case_scores = np.concatenate([case_attack_scores, case_wm_scores])
                
                # Calculate ROC-AUC for this case
                case_auc = roc_auc_score(case_labels, case_scores)
                per_case_auc[case_name] = case_auc
            else:
                per_case_auc[case_name] = float('nan')
        
        # Print results in a clean, readable format
        logging.info("\n")
        logging.info("="*80)
        logging.info(f"{'ATTACK EVALUATION RESULTS':^80}")
        logging.info("="*80)
        logging.info(f"Reference threshold at 1% FPR: {threshold:.4f}")
        logging.info(f"Overall Attack ROC-AUC: {overall_attack_auc:.4f}")
        logging.info("-"*80)
        logging.info(f"{'Attack Success Rates (ASR@1%FPR) and Per-Case ROC-AUC':^80}")
        logging.info("-"*80)
        logging.info(f"{'Attack Case':<25} | {'ASR (%)':<10} | {'ROC-AUC':<10} | {'Mean Score':<15} | {'Std Dev':<10}")
        logging.info("-"*80)
        
        for case_name in attack_success_rates.keys():
            asr = attack_success_rates[case_name]
            auc = per_case_auc[case_name]
            mean_score = attack_results[case_name]['mean'][0]  # Get the mean for the first alpha value
            std_score = attack_results[case_name]['std'][0]    # Get the std for the first alpha value
            logging.info(f"{case_name:<25} | {asr:9.2f}% | {auc:10.4f} | {mean_score:14.4f} | {std_score:9.4f}")
        
        logging.info("="*80)
        logging.info(f"Higher ASR means more successful attack (more images fooling the detector)")
        logging.info(f"Lower ROC-AUC means less separation between authentic and attacked images")
        logging.info("="*80)
        
        return overall_attack_auc, attack_success_rates, attack_results, per_case_auc  # Return per-case AUC as well
    
    return None, None, None, None  # Return None for non-rank-0 processes