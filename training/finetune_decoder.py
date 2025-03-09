import torch
import torch.nn as nn
import logging
import os
import gc
import math

from key.key import generate_mask_secret_key, mask_image_with_key
from utils.image_utils import constrain_image
from models.stylegan2 import is_stylegan2
from evaluation.evaluate_model import evaluate_model

def finetune_decoder(
    time_string,
    gan_model,
    watermarked_model,
    decoder,
    latent_dim,
    batch_size,
    device,
    num_epochs,
    learning_rate,
    max_delta,
    saving_path,
    mask_switch_on,
    seed_key,
    rank=0,
    world_size=1,
    key_type="csprng",
    pgd_steps=200,
    pgd_alpha=0.01,
    save_interval=1
):
    """
    Finetune an existing decoder to be more robust against PGD attacks by training it on:
    1. Original images (label 1)
    2. Watermarked images (post-constrained) (label 0)
    3. Adversarial samples from PGD attack (label 0)
    
    Args:
        time_string: String based on current time for file naming
        gan_model: Original generator model
        watermarked_model: Watermarked model
        decoder: Decoder model to be finetuned
        latent_dim: Dimension of the latent space
        batch_size: Batch size for training
        device: Device to use for training
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        max_delta: Maximum allowed perturbation for both watermarking and attacks
        saving_path: Path to save the finetuned model
        mask_switch_on: Whether to use masking
        seed_key: Seed for random number generation
        rank: Process rank for distributed training
        world_size: Total number of processes for distributed training
        key_type: Type of key generation method
        pgd_steps: Number of PGD steps for generating adversarial examples
        pgd_alpha: Step size for PGD attack
        save_interval: How often to save the model (in epochs)
    """
    if rank == 0:
        logging.info(f"Starting decoder finetuning for {num_epochs} epochs")
        logging.info(f"World size: {world_size}")
        logging.info(f"max_delta = {max_delta}")
        logging.info(f"time_string = {time_string}")
        logging.info(f"PGD steps = {pgd_steps}, PGD alpha = {pgd_alpha}")
        logging.info("Decoder structure:\n%s", decoder.module if isinstance(decoder, nn.parallel.DistributedDataParallel) else decoder)
        logging.info(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters())}")

    gan_model.eval()
    watermarked_model.eval()
    decoder.train()
    
    # Create optimizer for decoder only
    optimizer_D = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    
    # BCE loss for binary classification
    criterion = nn.BCELoss()
    
    # Track metrics
    avg_losses = []
    
    # Training loop
    total_batches = math.ceil(num_epochs * 1000 / batch_size)  # Assuming 1000 images per epoch
    
    # Generate a mask for the entire training if using masking
    if mask_switch_on and key_type != "none":
        # Create a dummy batch to get the shape
        dummy_batch = torch.zeros((batch_size, 3, 256, 256), device=device)
        k_mask = generate_mask_secret_key(
            image_shape=dummy_batch.shape,
            seed=seed_key,
            device=device,
            key_type=key_type
        )
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # We'll process approximately 1000 images per epoch
        for i in range(0, 1000, batch_size):
            batch_size_actual = min(batch_size, 1000 - i)
            if batch_size_actual <= 0:
                break
                
            torch.cuda.empty_cache()
            gc.collect()
            
            # Generate random latent vectors (different on each device)
            torch.manual_seed(seed_key + epoch * 1000 + i * world_size + rank)
            z = torch.randn((batch_size_actual, latent_dim), device=device)
            
            # Step 1: Generate original images (label 1)
            with torch.no_grad():
                if is_stylegan2(gan_model):
                    original_images = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
                else:
                    original_images = gan_model(z)
            
            # Step 2: Generate watermarked images (label 0)
            with torch.no_grad():
                if is_stylegan2(watermarked_model):
                    watermarked_images = watermarked_model(z, None, truncation_psi=1.0, noise_mode="const")
                else:
                    watermarked_images = watermarked_model(z)
                
                # Apply constraint to ensure max_delta
                watermarked_images = constrain_image(watermarked_images, original_images, max_delta)
            
            # Step 3: Generate adversarial samples using PGD attack (label 0)
            adversarial_images = generate_adversarial_examples(
                decoder=decoder,
                original_images=original_images.clone(),
                device=device,
                num_steps=pgd_steps,
                alpha=pgd_alpha,
                max_delta=max_delta,
                key_type=key_type if mask_switch_on else "none",
                k_mask=k_mask if mask_switch_on and key_type != "none" else None
            )
            
            # Apply masking if enabled
            if mask_switch_on and key_type != "none":
                masked_original = mask_image_with_key(images=original_images, cnn_key=k_mask)
                masked_watermarked = mask_image_with_key(images=watermarked_images, cnn_key=k_mask)
                masked_adversarial = mask_image_with_key(images=adversarial_images, cnn_key=k_mask)
                
                train_images = torch.cat([masked_original, masked_watermarked, masked_adversarial], dim=0)
            else:
                train_images = torch.cat([original_images, watermarked_images, adversarial_images], dim=0)
            
            # Create labels:
            # 0 for original and adversarial (first batch_size_actual and last batch_size_actual)
            # 1 for watermarked (middle batch_size_actual)
            labels = torch.zeros(train_images.size(0), 1, device=device)
            labels[batch_size_actual:2*batch_size_actual] = 1.0  # Watermarked images have label 1
            
            # Train the decoder
            optimizer_D.zero_grad()
            
            # Forward pass
            outputs = decoder(train_images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Update weights
            optimizer_D.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            
            if rank == 0 and (i == 0 or (i + batch_size) >= 1000):
                # Calculate and log accuracy for each type
                with torch.no_grad():
                    original_scores = outputs[:batch_size_actual]
                    watermarked_scores = outputs[batch_size_actual:2*batch_size_actual]
                    adversarial_scores = outputs[2*batch_size_actual:]
                    
                    # Calculate accuracy
                    orig_acc = (original_scores < 0.5).float().mean().item()
                    water_acc = (watermarked_scores > 0.5).float().mean().item()
                    adv_acc = (adversarial_scores < 0.5).float().mean().item()
                    
                    # Calculate mean and std of confidence scores
                    orig_mean = original_scores.mean().item()
                    orig_std = original_scores.std().item()
                    water_mean = watermarked_scores.mean().item()
                    water_std = watermarked_scores.std().item()
                    adv_mean = adversarial_scores.mean().item()
                    adv_std = adversarial_scores.std().item()
                    
                    logging.info(
                        f"Epoch {epoch+1}/{num_epochs}, Batch {i//batch_size + 1}/{1000//batch_size}, "
                        f"Loss: {loss.item():.4f}"
                    )
                    logging.info(
                        f"Accuracy: Original {orig_acc:.4f}, Watermarked {water_acc:.4f}, Adversarial {adv_acc:.4f}"
                    )
                    logging.info(
                        f"Scores: Original {orig_mean:.4f}±{orig_std:.4f}, "
                        f"Watermarked {water_mean:.4f}±{water_std:.4f}, "
                        f"Adversarial {adv_mean:.4f}±{adv_std:.4f}"
                    )
        
        # Average loss for the epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_losses.append(avg_loss)
        
        if rank == 0:
            logging.info(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
            
            # Save intermediate models
            if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
                os.makedirs(saving_path, exist_ok=True)
                decoder_path = os.path.join(saving_path, f'finetuned_decoder_{time_string}_epoch{epoch+1}.pth')
                
                # Save the model state dict (handle distributed/non-distributed case)
                if isinstance(decoder, nn.parallel.DistributedDataParallel):
                    torch.save(decoder.module.state_dict(), decoder_path)
                else:
                    torch.save(decoder.state_dict(), decoder_path)
                    
                logging.info(f"Saved finetuned decoder at epoch {epoch+1}: {decoder_path}")
                
                # Run evaluation on the finetuned decoder
                decoder.eval()
                with torch.no_grad():
                    eval_results = evaluate_model(
                        num_images=100,  # Small sample for quick evaluation
                        gan_model=gan_model,
                        watermarked_model=watermarked_model,
                        decoder=decoder,
                        device=device,
                        plotting=False,
                        latent_dim=latent_dim,
                        max_delta=max_delta,
                        mask_switch_on=mask_switch_on,
                        seed_key=seed_key,
                        flip_key_type="none",
                        compute_fid=False,
                        key_type=key_type,
                        rank=rank,
                        batch_size=batch_size
                    )
                    
                    # Log the key metrics
                    auc = eval_results['auc']
                    tpr_at_1_fpr = eval_results['watermarked']['tpr_at_1_fpr']
                    logging.info(f"Evaluation Results at Epoch {epoch+1}:")
                    logging.info(f"AUC score: {auc:.4f}")
                    logging.info(f"TPR@1%FPR (watermarked): {tpr_at_1_fpr:.4f}")
                decoder.train()
    
    # Final save
    if rank == 0:
        os.makedirs(saving_path, exist_ok=True)
        final_decoder_path = os.path.join(saving_path, f'finetuned_decoder_final_{time_string}.pth')
        
        # Save the model state dict
        if isinstance(decoder, nn.parallel.DistributedDataParallel):
            torch.save(decoder.module.state_dict(), final_decoder_path)
        else:
            torch.save(decoder.state_dict(), final_decoder_path)
            
        logging.info(f"Saved final finetuned decoder: {final_decoder_path}")
        
        # Final evaluation
        decoder.eval()
        with torch.no_grad():
            eval_results = evaluate_model(
                num_images=500,  # Larger sample for final evaluation
                gan_model=gan_model,
                watermarked_model=watermarked_model,
                decoder=decoder,
                device=device,
                plotting=False,
                latent_dim=latent_dim,
                max_delta=max_delta,
                mask_switch_on=mask_switch_on,
                seed_key=seed_key,
                flip_key_type="none",
                compute_fid=False,
                key_type=key_type,
                rank=rank,
                batch_size=batch_size
            )
            
            # Log the final metrics
            auc = eval_results['auc']
            tpr_at_1_fpr = eval_results['watermarked']['tpr_at_1_fpr']
            mean_score_watermarked = eval_results['watermarked']['mean_score']
            std_score_watermarked = eval_results['watermarked']['std_score']
            
            logging.info("Final Evaluation Results:")
            logging.info(f"AUC score: {auc:.4f}")
            logging.info(f"TPR@1%FPR (watermarked): {tpr_at_1_fpr:.4f}")
            logging.info(f"Mean score (watermarked): {mean_score_watermarked:.4f}")
            logging.info(f"Std score (watermarked): {std_score_watermarked:.4f}")
        
        logging.info("Decoder finetuning completed.")
        
    return decoder


def generate_adversarial_examples(
    decoder: nn.Module,
    original_images: torch.Tensor,
    device: torch.device,
    num_steps: int = 200,
    alpha: float = 0.01,
    max_delta: float = 0.05,
    momentum: float = 0.9,
    key_type: str = "none",
    k_mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Generate adversarial examples for the decoder by performing PGD attack.
    Unlike the regular attack function, this uses the actual decoder for the attack.
    
    Args:
        decoder: The decoder model to attack
        original_images: The original images to perturb
        device: Device to perform the attack on
        num_steps: Number of PGD steps
        alpha: Step size for PGD
        max_delta: Maximum perturbation allowed
        momentum: Momentum factor for PGD attack
        key_type: Type of key generation method
        k_mask: Pre-generated mask for efficiency (only used if key_type != "none")
        
    Returns:
        Adversarial examples
    """
    # Ensure decoder is in eval mode during attack generation
    decoder.eval()
    
    # Clone the original images to avoid modifying them
    images = original_images.clone().detach()
    
    # Initialize perturbation
    perturbation = torch.zeros_like(images, requires_grad=True)
    
    # Initialize momentum buffer if using momentum
    if momentum > 0:
        grad_momentum = torch.zeros_like(images)
    
    # BCE loss with target 1 (making the decoder classify adversarial examples as watermarked)
    criterion = nn.BCELoss()
    
    # Perform PGD attack
    for step in range(num_steps):
        perturbation.requires_grad_(True)
        
        # Create adversarial examples
        adv_images = torch.clamp(images + perturbation, -1, 1)
        
        # Apply masking if enabled
        if key_type != "none":
            if k_mask is None:
                # Generate mask if not provided
                k_mask = generate_mask_secret_key(adv_images.shape, 2024, device, key_type=key_type)
            masked_images = mask_image_with_key(images=adv_images, cnn_key=k_mask)
            decoder_input = masked_images
        else:
            decoder_input = adv_images
        
        # Forward pass through decoder
        scores = decoder(decoder_input)
        
        # Target is 1 to fool the detector into classifying adversarial examples as watermarked
        target = torch.ones_like(scores)
        loss = criterion(scores, target)
        
        # Compute gradients
        grad = torch.autograd.grad(loss, perturbation)[0]
        
        # Apply momentum if used
        if momentum > 0:
            grad_momentum = momentum * grad_momentum + grad
            update = alpha * grad_momentum.sign()
        else:
            update = alpha * grad.sign()
        
        # Update perturbation (gradient descent to minimize BCE loss)
        with torch.no_grad():
            perturbation -= update  # Use gradient descent to minimize BCE loss
            
            # Project perturbation to ensure max_delta constraint
            perturbation.data = torch.clamp(perturbation.data, -max_delta, max_delta)
            
            # Ensure resulting images are in valid range [-1, 1]
            adv_images = torch.clamp(images + perturbation, -1, 1)
            
            # Recompute perturbation based on clamped images
            perturbation.data = adv_images - images
    
    # Return final adversarial examples
    return torch.clamp(images + perturbation.detach(), -1, 1) 