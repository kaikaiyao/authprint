import torch
import torch.nn as nn
import logging
import os
import gc
import math
import torch.nn.functional as F

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
    pgd_steps=100,
    pgd_alpha=0.01,
    save_interval=1
):
    """
    Finetune an existing decoder to be more robust against PGD attacks by training it on:
    1. Original images (label 0)
    2. Watermarked images (post-constrained) (label 1)
    3. Adversarial samples from PGD attack (label 0)
    
    This function makes the entire decoder trainable without freezing any layers.
    
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
    
    # Get the actual decoder model (handle DDP case)
    decoder_module = decoder.module if isinstance(decoder, nn.parallel.DistributedDataParallel) else decoder

    # Make all decoder layers trainable instead of freezing some
    trainable_count = 0
    
    # Make all parameters trainable
    for name, param in decoder_module.named_parameters():
        param.requires_grad = True
        trainable_count += param.numel()
    
    if rank == 0:
        logging.info(f"Total model layers are trainable: {trainable_count} parameters")
    
    decoder.train()
    
    # Create optimizer for decoder only - add momentum and weight decay (L2 regularization)
    optimizer_D = torch.optim.SGD(
        filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D, mode='min', factor=0.5, patience=2, verbose=True if rank == 0 else False
    )
    
    # Track metrics
    avg_losses = []
    
    # Training loop
    total_batches = math.ceil(num_epochs * 1000 / batch_size)  # Assuming 1000 images per epoch
    
    # For logits output, use BCEWithLogitsLoss; for sigmoid output, use BCELoss
    # First, let's check what kind of output the model produces
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 256, 256, device=device)
        dummy_output = decoder(dummy_input)
        is_sigmoid_output = torch.all((dummy_output >= 0) & (dummy_output <= 1))
    
    if is_sigmoid_output:
        criterion = nn.BCELoss()
        if rank == 0:
            logging.info("Using BCELoss for sigmoid output")
    else:
        criterion = nn.BCEWithLogitsLoss()
        if rank == 0:
            logging.info("Using BCEWithLogitsLoss for logit output")
    
    # Save the initial model state for reset if needed
    initial_state = {k: v.clone() for k, v in decoder_module.state_dict().items()}
    
    # Generate a mask for the entire training if using masking
    if mask_switch_on and key_type != "none":
        # Create a dummy batch to get the shape
        dummy_batch = torch.zeros((batch_size, 3, 256, 256), device=device)
        # For CryptoCNN, we only need to generate it once
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
            
            # Step 1: Generate original images (label 0)
            with torch.no_grad():
                if is_stylegan2(gan_model):
                    original_images = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
                else:
                    original_images = gan_model(z)
            
            # Step 2: Generate watermarked images (label 1)
            with torch.no_grad():
                if is_stylegan2(watermarked_model):
                    watermarked_images = watermarked_model(z, None, truncation_psi=1.0, noise_mode="const")
                else:
                    watermarked_images = watermarked_model(z)
                
                # Apply constraint to ensure max_delta
                watermarked_images = constrain_image(watermarked_images, original_images, max_delta)
            
            # Step 3: Generate adversarial samples using PGD attack (label 0)
            # Use a step-wise progression with plateaus instead of a smooth curve
            current_progress = (epoch * 1000 + i) / (num_epochs * 1000)
            
            # Get the current step level (0-10) based on progress
            # This creates 10 distinct difficulty levels
            # We'll use a custom step function to create plateaus
            def get_difficulty_level(progress, num_levels=10):
                """
                Create a step-wise progression with plateaus.
                Starts at very low strength and increases in steps.
                
                Plateau periods:
                - 2% strength for first 10% of training
                - 5% strength from 10-20% of training
                - 10% strength from 20-30% of training
                - 20% strength from 30-40% of training
                - 30% strength from 40-50% of training
                - 45% strength from 50-60% of training
                - 60% strength from 60-70% of training
                - 75% strength from 70-80% of training
                - 90% strength from 80-90% of training
                - Full strength for last 10% of training
                """
                if progress < 0.10:
                    return 0.05  # Start at 2% strength
                elif progress < 0.20:
                    return 0.10  # 5% strength
                elif progress < 0.30:
                    return 0.10  # 10% strength
                elif progress < 0.40:
                    return 0.20  # 20% strength
                elif progress < 0.50:
                    return 0.30  # 30% strength
                elif progress < 0.60:
                    return 0.45  # 45% strength
                elif progress < 0.70:
                    return 0.60  # 60% strength
                elif progress < 0.80:
                    return 0.75  # 75% strength
                elif progress < 0.90:
                    return 0.90  # 90% strength
                else:
                    return 1.0  # Full strength
            
            # Get current difficulty level (0.2-1.0)
            difficulty = get_difficulty_level(current_progress)
            
            # Calculate PGD steps and alpha based on difficulty level
            # Steps range from 2% to 100% of pgd_steps
            # Alpha ranges from 2% to 100% of pgd_alpha
            current_pgd_steps = max(3, int(pgd_steps * difficulty))
            current_alpha = pgd_alpha * difficulty
            
            # Log PGD difficulty adjustment at regular intervals
            if rank == 0 and (i == 0 or (i + batch_size) >= 1000 or i % 5 == 0):
                steps_pct = (current_pgd_steps / pgd_steps) * 100
                alpha_pct = (current_alpha / pgd_alpha) * 100
                training_pct = current_progress * 100
                logging.info(
                    f"PGD Difficulty: Training Progress {training_pct:.1f}%, "
                    f"Level: {difficulty:.1f}, Steps {current_pgd_steps}/{pgd_steps} ({steps_pct:.1f}%), "
                    f"Alpha {current_alpha:.5f}/{pgd_alpha:.5f} ({alpha_pct:.1f}%)"
                )
            
            adversarial_images = generate_adversarial_examples(
                decoder=decoder,
                original_images=original_images.clone(),
                device=device,
                num_steps=current_pgd_steps,
                alpha=current_alpha,
                max_delta=max_delta,
                key_type=key_type if mask_switch_on else "none",
                k_mask=k_mask if mask_switch_on and key_type != "none" else None,
                rank=rank
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
            
            # Apply sigmoid if using BCELoss and outputs are logits
            probs = outputs
            if not is_sigmoid_output:
                probs = torch.sigmoid(outputs)
            
            # Split the outputs for different types of images
            original_outputs = outputs[:batch_size_actual]
            watermarked_outputs = outputs[batch_size_actual:2*batch_size_actual]
            adversarial_outputs = outputs[2*batch_size_actual:]
            
            # Binary cross entropy loss for each type separately
            # This gives us more control over weighting
            original_labels = torch.zeros_like(original_outputs)
            watermarked_labels = torch.ones_like(watermarked_outputs)
            adversarial_labels = torch.zeros_like(adversarial_outputs)  # Should be classified as 0 (non-watermarked)
            
            # Compute separate losses for each type
            if isinstance(criterion, nn.BCELoss):
                orig_loss = criterion(original_outputs, original_labels)
                water_loss = criterion(watermarked_outputs, watermarked_labels)
                adv_loss = criterion(adversarial_outputs, adversarial_labels)
            else:
                orig_loss = F.binary_cross_entropy_with_logits(original_outputs, original_labels)
                water_loss = F.binary_cross_entropy_with_logits(watermarked_outputs, watermarked_labels)
                adv_loss = F.binary_cross_entropy_with_logits(adversarial_outputs, adversarial_labels)
            
            # Weighted loss - gradually increase the weight of adversarial loss
            # Start with equal weights and linearly increase adv_weight
            progress = (epoch * 1000 + i) / (num_epochs * 1000)
            adv_weight = 1.0 + 4.0 * progress  # Increase from 1.0 to 5.0 over training
            
            # Combined classification loss with weighting
            classification_loss = orig_loss + water_loss + adv_weight * adv_loss
            
            # Diversity loss - cosine similarity between predictions within the same class
            # This encourages the network to make diverse predictions even within a class
            original_preds = probs[:batch_size_actual]
            watermarked_preds = probs[batch_size_actual:2*batch_size_actual]
            adversarial_preds = probs[2*batch_size_actual:]
            
            # Calculate pair-wise cosine similarities (if batch_size > 1)
            diversity_loss = 0.0
            if batch_size_actual > 1:
                orig_flat = original_preds.view(-1, 1)
                water_flat = watermarked_preds.view(-1, 1)
                adv_flat = adversarial_preds.view(-1, 1)
                
                # Compute cosine similarity (dot product for normalized vectors)
                orig_sim = torch.mm(orig_flat, orig_flat.t()).mean()
                water_sim = torch.mm(water_flat, water_flat.t()).mean()
                adv_sim = torch.mm(adv_flat, adv_flat.t()).mean()
                
                # Penalize high similarity (encourage diversity)
                diversity_loss = (orig_sim + adv_sim + water_sim) / 3.0
            
            # Combined loss
            loss = classification_loss + 0.01 * diversity_loss
            
            # Add L2 regularization to prevent extreme weights
            l2_reg = 0.0
            for param in decoder.parameters():
                if param.requires_grad:
                    l2_reg += torch.norm(param)
            
            loss += 0.001 * l2_reg
            
            loss.backward()
            
            # Gradient clipping to prevent extreme updates
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer_D.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            
            # Check if model is collapsing and reset if necessary
            with torch.no_grad():
                pred_std = probs.std().item()
                if pred_std < 0.01:  # If predictions have very small std deviation
                    if rank == 0:
                        logging.warning(f"Model collapse detected (std={pred_std:.6f})! Resetting model weights.")
                    
                    # Reset model weights to initial state with small random perturbation
                    for name, param in decoder_module.named_parameters():
                        if param.requires_grad:
                            init_weight = initial_state[name]
                            # Add small random noise to break symmetry
                            param.data = init_weight + 0.01 * torch.randn_like(init_weight)
            
            # Log more frequently - every 5 batches or at beginning/end
            if rank == 0 and (i == 0 or (i + batch_size) >= 1000 or i % 5 == 0):
                # Calculate and log scores
                with torch.no_grad():
                    original_scores = original_preds
                    watermarked_scores = watermarked_preds
                    adversarial_scores = adversarial_preds
                    
                    # Calculate mean and std of confidence scores
                    orig_mean = original_scores.mean().item()
                    orig_std = original_scores.std().item()
                    water_mean = watermarked_scores.mean().item()
                    water_std = watermarked_scores.std().item()
                    adv_mean = adversarial_scores.mean().item()
                    adv_std = adversarial_scores.std().item()
                    all_std = probs.std().item()
                    
                    # Calculate success rates
                    orig_success = (original_scores < 0.5).float().mean().item() * 100  # Should be classified as 0
                    water_success = (watermarked_scores > 0.5).float().mean().item() * 100  # Should be classified as 1
                    adv_success_before = (adversarial_scores > 0.5).float().mean().item() * 100  # Was initially misclassified as 1
                    adv_success_after = (adversarial_scores < 0.5).float().mean().item() * 100  # Now correctly classified as 0
                    
                    # Calculate strength of adversarial attack
                    adv_score_high = (adversarial_scores > 0.9).float().mean().item() * 100  # Very high confidence wrong predictions
                    adv_score_medium = (adversarial_scores > 0.7).float().mean().item() * 100  # Medium confidence wrong predictions
                    adv_score_low = (adversarial_scores > 0.5).float().mean().item() * 100  # Low confidence wrong predictions
                    
                    if rank == 0:
                        logging.info(
                            f"Epoch {epoch+1}/{num_epochs}, Batch {i//batch_size + 1}/{1000//batch_size}, "
                            f"Loss: {loss.item():.4f}, Class Loss: {classification_loss.item():.4f}, "
                            f"Div Loss: {diversity_loss:.4f}, L2: {l2_reg.item():.4f}, All StdDev: {all_std:.4f}"
                        )
                        logging.info(
                            f"Separate Losses - Original: {orig_loss.item():.4f}, Watermarked: {water_loss.item():.4f}, "
                            f"Adversarial: {adv_loss.item():.4f} (weight: {adv_weight:.2f}), "
                            f"PGD Steps: {current_pgd_steps}, PGD Alpha: {current_alpha:.5f}"
                        )
                        logging.info(
                            f"Scores: Original {orig_mean:.4f}±{orig_std:.4f}, "
                            f"Watermarked {water_mean:.4f}±{water_std:.4f}, "
                            f"Adversarial {adv_mean:.4f}±{adv_std:.4f}"
                        )
                        logging.info(
                            f"Success Rates: Original {orig_success:.1f}%, Watermarked {water_success:.1f}%, "
                            f"Adversarial (misclassified as watermarked): {adv_success_before:.1f}% "
                            f"(high: {adv_score_high:.1f}%, med: {adv_score_medium:.1f}%, low: {adv_score_low:.1f}%), "
                            f"Adversarial (correctly classified): {adv_success_after:.1f}%"
                        )
        
        # Average loss for the epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_losses.append(avg_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_loss)
        
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
        
        if rank == 0:
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
    k_mask: torch.Tensor = None,  # Note: k_mask can be a Tensor or a CryptoCNN object
    rank: int = 0  # Add rank parameter for controlling logging
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
        rank: Process rank for controlling logging
        
    Returns:
        Adversarial examples
    """
    # Ensure decoder is in eval mode during attack generation
    was_training = decoder.training
    decoder.eval()
    
    # Determine output type (sigmoid or logits)
    with torch.no_grad():
        dummy_input = torch.zeros((1, 3, 256, 256), device=device)  # Smaller dummy input
        dummy_output = decoder(dummy_input)
        is_sigmoid_output = torch.all((dummy_output >= 0) & (dummy_output <= 1))
    
    # Get batch size and determine sub-batch size to reduce memory usage
    full_batch_size = original_images.size(0)
    # Use smaller sub-batches for memory efficiency in DDP
    sub_batch_size = min(8, full_batch_size)
    
    # Initialize the result tensor to store all adversarial examples
    with torch.no_grad():
        all_adv_images = torch.zeros_like(original_images)
    
    # Process each sub-batch separately
    for start_idx in range(0, full_batch_size, sub_batch_size):
        end_idx = min(start_idx + sub_batch_size, full_batch_size)
        current_batch_size = end_idx - start_idx
        
        # Get the current sub-batch
        images = original_images[start_idx:end_idx].clone().detach()
        
        # Initialize perturbation with small random noise to avoid bad local minima
        perturbation = (torch.rand_like(images) * 2 - 1) * max_delta * 0.1
        perturbation.requires_grad_(True)
        
        # Target is 1.0 (watermarked) to fool the decoder
        target = torch.ones(current_batch_size, 1, device=device)
        
        # Initialize tracking variables
        best_adv_images = images.clone()  # Default to original images
        best_loss = float('inf')
        initial_score = None
        best_score = 0.0  # Initialize to a low value
        
        # Momentum buffer
        if momentum > 0:
            grad_momentum = torch.zeros_like(images)
        
        # Stagnation tracking
        stagnation_counter = 0
        last_loss = float('inf')
        
        # Perform PGD attack
        for step in range(num_steps):
            # Free up memory
            if step > 0:
                torch.cuda.empty_cache()
            
            # Make sure perturbation requires gradients for this step
            perturbation.requires_grad_(True)
            
            # Create adversarial examples
            adv_images = torch.clamp(images + perturbation, -1, 1)
            
            # Apply masking if enabled
            if key_type != "none":
                if k_mask is None:
                    # Generate mask if not provided
                    mask_shape = (current_batch_size,) + images.shape[1:]
                    k_mask_batch = generate_mask_secret_key(mask_shape, 2024, device, key_type=key_type)
                    decoder_input = mask_image_with_key(images=adv_images, cnn_key=k_mask_batch)
                else:
                    # Use the provided mask - handle both tensor and CryptoCNN cases
                    # CryptoCNN objects should be used directly without slicing
                    if hasattr(k_mask, 'size') and k_mask.size(0) > 1:
                        # It's a tensor with batch dimension, slice it
                        mask_slice = k_mask[start_idx:end_idx]
                        decoder_input = mask_image_with_key(images=adv_images, cnn_key=mask_slice)
                    else:
                        # It's either a single-batch tensor or a CryptoCNN object, use as is
                        decoder_input = mask_image_with_key(images=adv_images, cnn_key=k_mask)
            else:
                decoder_input = adv_images
            
            # Forward pass through decoder
            outputs = decoder(decoder_input)
            probs = outputs
            if not is_sigmoid_output:
                probs = torch.sigmoid(outputs)
            
            # Compute loss - we want to maximize the probability that this is classified as watermarked
            if is_sigmoid_output:
                loss = F.binary_cross_entropy(probs, target)
            else:
                loss = F.binary_cross_entropy_with_logits(outputs, target)
            
            # Check for stagnation - if loss doesn't improve for several steps
            if abs(loss.item() - last_loss) < 1e-5:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            last_loss = loss.item()
            
            # If stagnation detected, add extra randomness to escape local minimum
            if stagnation_counter >= 5:
                with torch.no_grad():
                    random_noise = (torch.rand_like(perturbation) * 2 - 1) * max_delta * 0.1
                    perturbation.data += random_noise
                    stagnation_counter = 0
                    if rank == 0:
                        logging.debug(f"Added random noise at step {step+1} to escape local minimum")
            
            # Keep track of best adversarial examples so far
            with torch.no_grad():
                current_score = probs.mean().item()
                
                # Record initial score
                if step == 0:
                    initial_score = current_score
                    best_score = current_score
                    best_adv_images = adv_images.clone()
                
                # Track best examples - LOWER loss is better for fooling the model
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_score = current_score
                    best_adv_images = adv_images.clone()
                
                # Also track by direct score comparison - higher score is better
                if current_score > best_score:
                    best_score = current_score
                    best_adv_images = adv_images.clone()
                    
                # Log progress at specific intervals
                log_steps = [0, num_steps//4, num_steps//2, 3*num_steps//4, num_steps-1]  # Log at these steps
                if step in log_steps and rank == 0:
                    perturbation_norm = torch.norm(perturbation, p=float('inf')).item()
                    logging.info(
                        f"PGD Attack Step {step+1}/{num_steps}: "
                        f"Loss={loss.item():.6f} (lower is better), "
                        f"Current Score={current_score:.4f}, Best Score Found={best_score:.4f}, "
                        f"Max Perturbation={perturbation_norm:.4f}"
                    )
            
            # Compute gradients
            grad = torch.autograd.grad(loss, perturbation, create_graph=False, retain_graph=False)[0]
            
            # Free up memory explicitly
            outputs = None
            probs = None
            loss = None
            
            # Apply momentum if used
            if momentum > 0:
                grad_momentum = momentum * grad_momentum + grad
                update = alpha * grad_momentum.sign()
            else:
                update = alpha * grad.sign()
            
            # Free up memory explicitly
            grad = None
            torch.cuda.empty_cache()
            
            # Update perturbation - gradient descent to minimize loss
            with torch.no_grad():
                perturbation -= update
                
                # Add small random noise occasionally to escape local minima
                if step % 10 == 0:
                    noise = (torch.rand_like(perturbation) * 2 - 1) * alpha * 0.1
                    perturbation.data += noise
                
                # Project perturbation to ensure max_delta constraint
                perturbation.data = torch.clamp(perturbation.data, -max_delta, max_delta)
                
                # Ensure resulting images are in valid range [-1, 1]
                adv_images = torch.clamp(images + perturbation, -1, 1)
                
                # Recompute perturbation based on clamped images
                perturbation.data = adv_images - images
            
            # Free up memory
            update = None
            adv_images = None
            torch.cuda.empty_cache()
        
        # Log final attack results
        if initial_score is not None and best_score is not None and rank == 0:
            improvement = best_score - initial_score
            pct_improvement = (improvement / max(initial_score, 0.0001)) * 100
            logging.info(
                f"PGD Attack Complete: "
                f"Initial Score={initial_score:.4f}, Best Score Found={best_score:.4f}, "
                f"Improvement={improvement:.4f} ({pct_improvement:.1f}%)"
            )
        
        # Store the best adversarial examples for this sub-batch
        with torch.no_grad():
            all_adv_images[start_idx:end_idx] = best_adv_images
        
        # Free memory
        images = None
        best_adv_images = None
        perturbation = None
        if momentum > 0:
            grad_momentum = None
        torch.cuda.empty_cache()
    
    # Restore original training state
    decoder.train(was_training)
    
    # Return final adversarial examples
    return all_adv_images 