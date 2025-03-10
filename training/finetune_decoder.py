import torch
import torch.nn as nn
import logging
import os
import gc
import math
import torch.nn.functional as F
import time

from key.key import generate_mask_secret_key, mask_image_with_key
from utils.image_utils import constrain_image
from models.stylegan2 import is_stylegan2
from evaluation.evaluate_model import evaluate_model

# Custom logging filter that only allows messages from rank 0
class RankFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        
    def filter(self, record):
        # Only allow log messages from rank 0
        return self.rank == 0

def process_in_subbatches(func, tensor, sub_batch_size=8, **kwargs):
    """
    Process a tensor in smaller sub-batches to reduce memory usage.
    
    Args:
        func: Function to apply to each sub-batch
        tensor: Input tensor to process
        sub_batch_size: Size of each sub-batch
        **kwargs: Additional arguments to pass to func
    
    Returns:
        Processed tensor with same shape as input
    """
    result = []
    num_subbatches = math.ceil(tensor.size(0) / sub_batch_size)
    
    for i in range(num_subbatches):
        start_idx = i * sub_batch_size
        end_idx = min((i + 1) * sub_batch_size, tensor.size(0))
        
        # Process this sub-batch
        sub_tensor = tensor[start_idx:end_idx]
        processed = func(sub_tensor, **kwargs)
        result.append(processed)
        
        # Free memory
        torch.cuda.empty_cache()
    
    # Combine results
    return torch.cat(result, dim=0)

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
    # Configure logging to filter based on rank
    # Get the root logger and apply the rank filter to all handlers
    root_logger = logging.getLogger()
    
    # First remove existing rank filters if any
    for handler in root_logger.handlers:
        for filter in handler.filters[:]:
            if isinstance(filter, RankFilter):
                handler.removeFilter(filter)
        
        # Add our rank filter to each handler
        handler.addFilter(RankFilter(rank))
    
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
    
    # Create optimizer for decoder
    if isinstance(decoder, (nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        decoder_module = decoder.module
    else:
        decoder_module = decoder
    
    # Keep the original optimizer settings
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=learning_rate * 0.1,  # Reduce initial learning rate
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Add learning rate scheduler with more conservative settings
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True if rank == 0 else False,
        min_lr=1e-6  # Add minimum learning rate
    )
    
    # Add accumulation steps for gradient accumulation
    accumulation_steps = 2  # Adjust based on memory requirements
    
    # Track metrics
    avg_losses = []
    
    # Adjust batch size for training
    effective_batch_size = batch_size
    
    # With full model training, reduce batch size to conserve memory if using DDP
    if world_size > 1:
        # Scale down batch size based on world size and reduce further due to full model training
        effective_batch_size = max(1, batch_size // (world_size * 2))
        if rank == 0:
            logging.info(f"Reduced batch size for DDP: {batch_size} -> {effective_batch_size} (full model training)")
    
    # Training loop
    total_batches = math.ceil(num_epochs * 1000 / effective_batch_size)  # Assuming 1000 images per epoch
    
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
        dummy_batch = torch.zeros((effective_batch_size, 3, 256, 256), device=device)
        # For CryptoCNN, we only need to generate it once
        k_mask = generate_mask_secret_key(
            image_shape=dummy_batch.shape,
            seed=seed_key,
            device=device,
            key_type=key_type
        )
    
    for epoch in range(num_epochs):
        # Start time for this epoch
        epoch_start_time = time.time()
        
        # Set models to training mode
        decoder.train()
        
        running_loss = 0.0
        running_classification_loss = 0.0
        running_diversity_loss = 0.0
        running_l2_reg = 0.0
        epoch_losses = []  # Initialize epoch_losses list
        
        # Shuffled indices for accessing both sets
        loader_indices = torch.randperm(1000)
        
        # Process in batches of size batch_size
        for idx in range(0, 1000, effective_batch_size):
            # Get indices for this batch
            indices = loader_indices[idx:min(idx + effective_batch_size, 1000)]
            i = indices[0].item()  # Use first index for logging
            current_batch_size = min(effective_batch_size, 1000 - idx)

            # Zero gradients only at the beginning of accumulation cycle
            if idx % (accumulation_steps * effective_batch_size) == 0:
                optimizer.zero_grad()
                
            torch.cuda.empty_cache()
            gc.collect()
            
            # Generate random latent vectors (different on each device)
            torch.manual_seed(seed_key + epoch * 1000 + i * world_size + rank)
            z = torch.randn((current_batch_size, latent_dim), device=device)
            
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
            if rank == 0 and (i == 0 or (i + effective_batch_size) >= 1000 or i % 5 == 0):
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
                # Process images in smaller batches to reduce memory usage
                sub_batch_size = max(1, current_batch_size // 4)  # Use smaller sub-batches
                
                masked_original = process_in_subbatches(
                    mask_image_with_key, 
                    original_images, 
                    sub_batch_size=sub_batch_size, 
                    cnn_key=k_mask
                )
                
                masked_watermarked = process_in_subbatches(
                    mask_image_with_key, 
                    watermarked_images, 
                    sub_batch_size=sub_batch_size, 
                    cnn_key=k_mask
                )
                
                masked_adversarial = process_in_subbatches(
                    mask_image_with_key, 
                    adversarial_images, 
                    sub_batch_size=sub_batch_size, 
                    cnn_key=k_mask
                )
                
                train_images = torch.cat([masked_original, masked_watermarked, masked_adversarial], dim=0)
            else:
                train_images = torch.cat([original_images, watermarked_images, adversarial_images], dim=0)
            
            # Create labels:
            # 0 for original and adversarial (first current_batch_size and last current_batch_size)
            # 1 for watermarked (middle current_batch_size)
            labels = torch.zeros(train_images.size(0), 1, device=device)
            labels[current_batch_size:2*current_batch_size] = 1.0  # Watermarked images have label 1
            
            # Train the decoder
            optimizer.zero_grad()
            
            # Forward pass
            outputs = decoder(train_images)
            
            # Apply sigmoid if using BCELoss and outputs are logits
            probs = outputs
            if not is_sigmoid_output:
                probs = torch.sigmoid(outputs)
            
            # Split the outputs for different types of images
            original_outputs = outputs[:current_batch_size]
            watermarked_outputs = outputs[current_batch_size:2*current_batch_size]
            adversarial_outputs = outputs[2*current_batch_size:]
            
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
            adv_weight = 1.0 + 2.0 * progress  # Reduce from 4.0 to 2.0 to make it less aggressive
            
            # Combined classification loss with weighting
            classification_loss = orig_loss + water_loss + adv_weight * adv_loss
            
            # Diversity loss - cosine similarity between predictions within the same class
            # This encourages the network to make diverse predictions even within a class
            original_preds = probs[:current_batch_size]
            watermarked_preds = probs[current_batch_size:2*current_batch_size]
            adversarial_preds = probs[2*current_batch_size:]
            
            # Calculate pair-wise cosine similarities (if current_batch_size > 1)
            diversity_loss = 0.0
            if current_batch_size > 1:
                orig_flat = original_preds.view(-1, 1)
                water_flat = watermarked_preds.view(-1, 1)
                adv_flat = adversarial_preds.view(-1, 1)
                
                # Compute cosine similarity (dot product for normalized vectors)
                orig_sim = torch.mm(orig_flat, orig_flat.t()).mean()
                water_sim = torch.mm(water_flat, water_flat.t()).mean()
                adv_sim = torch.mm(adv_flat, adv_flat.t()).mean()
                
                # Penalize high similarity (encourage diversity)
                diversity_loss = (orig_sim + adv_sim + water_sim) / 3.0
            
            # Combined loss with reduced diversity loss weight
            loss = classification_loss + 0.005 * diversity_loss  # Reduce from 0.01 to 0.005
            
            # Add L2 regularization to prevent extreme weights
            l2_reg = 0.0
            for param in decoder.parameters():
                if param.requires_grad:
                    l2_reg += torch.norm(param)
            
            loss += 0.0005 * l2_reg  # Reduce from 0.001 to 0.0005
            
            # Scale the loss according to accumulation steps
            loss = loss / accumulation_steps
            
            # Backward pass with memory optimization
            with torch.amp.autocast('cuda', enabled=False):
                loss.backward()
            
            # Store scores for logging before freeing memory
            # Make copies of the scores to preserve them for logging
            with torch.no_grad():
                score_original = original_preds.detach().clone() if i % 5 == 0 or i == 0 or (i + effective_batch_size) >= 1000 else None
                score_watermarked = watermarked_preds.detach().clone() if i % 5 == 0 or i == 0 or (i + effective_batch_size) >= 1000 else None
                score_adversarial = adversarial_preds.detach().clone() if i % 5 == 0 or i == 0 or (i + effective_batch_size) >= 1000 else None
                score_probs = probs.detach().clone() if i % 5 == 0 or i == 0 or (i + effective_batch_size) >= 1000 else None
            
            # Free up memory
            original_outputs = None
            watermarked_outputs = None
            adversarial_outputs = None
            original_preds = None
            watermarked_preds = None
            adversarial_preds = None
            train_images = None
            torch.cuda.empty_cache()
            
            # Update weights only after accumulating gradients
            if (idx + effective_batch_size) % (accumulation_steps * effective_batch_size) == 0 or (idx + effective_batch_size) >= 1000:
                # Clip gradients to prevent exploding gradients - reduce max_norm
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=0.5)  # Reduce from 1.0 to 0.5
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            
            # Accumulate loss statistics
            running_loss += loss.item() * accumulation_steps
            running_classification_loss += classification_loss.item()
            running_diversity_loss += diversity_loss.item() * 0.01
            running_l2_reg += l2_reg.item()
            epoch_losses.append(loss.item() * accumulation_steps)  # Track for epoch average
            
            # Free up remaining tensors
            classification_loss = None
            diversity_loss = None
            l2_reg = None
            loss = None
            torch.cuda.empty_cache()
            
            # Log more frequently - every 5 batches or at beginning/end
            if rank == 0 and (i == 0 or (i + effective_batch_size) >= 1000 or i % 5 == 0):
                # Calculate and log scores - use stored scores from earlier
                with torch.no_grad():
                    if score_original is not None and score_watermarked is not None and score_adversarial is not None and score_probs is not None:
                        original_scores = score_original
                        watermarked_scores = score_watermarked
                        adversarial_scores = score_adversarial
                        probs_for_logging = score_probs
                        
                        # Calculate mean scores
                        orig_mean = original_scores.mean().item()
                        water_mean = watermarked_scores.mean().item()
                        adv_mean = adversarial_scores.mean().item()
                        
                        # Calculate success rates
                        orig_success = (original_scores < 0.5).float().mean().item() * 100
                        water_success = (watermarked_scores > 0.5).float().mean().item() * 100
                        adv_success = (adversarial_scores < 0.5).float().mean().item() * 100
                        
                        # Log all information in a single line
                        logging.info(
                            f"Epoch {epoch+1}/{num_epochs} | Batch {i//effective_batch_size + 1}/{1000//effective_batch_size} | "
                            f"Loss: {running_loss:.4f} | "
                            f"Scores: O={orig_mean:.4f}({orig_success:.1f}%) W={water_mean:.4f}({water_success:.1f}%) A={adv_mean:.4f}({adv_success:.1f}%) | "
                            f"PGD: steps={current_pgd_steps} alpha={current_alpha:.5f}"
                        )
                        
                        # Free memory again
                        original_scores = None
                        watermarked_scores = None
                        adversarial_scores = None
                        probs_for_logging = None
                        score_original = None
                        score_watermarked = None
                        score_adversarial = None
                        score_probs = None
                        torch.cuda.empty_cache()
                    else:
                        logging.info(f"Skipping detailed logging for batch {i//effective_batch_size + 1} - scores not available")
        
        # Average loss for the epoch
        if len(epoch_losses) > 0:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_losses.append(avg_loss)
        else:
            avg_loss = running_loss / (1000 // effective_batch_size)
            avg_losses.append(avg_loss)
        
        # Update learning rate scheduler
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
                        batch_size=effective_batch_size
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
                batch_size=effective_batch_size
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
    Generate adversarial examples using PGD attack with reduced memory footprint
    """
    # Save computation graph memory by setting no_grad for setup
    with torch.no_grad():
        # Clone the original images to avoid modifying them
        images = original_images.clone()
        current_batch_size = images.size(0)
        
        # Process in smaller sub-batches to reduce memory usage 
        max_sub_batch_size = 2  # Process very small batches at a time
        num_sub_batches = math.ceil(current_batch_size / max_sub_batch_size)
        all_best_adv_images = torch.zeros_like(images)
        
        # Determine if model uses sigmoid in the final layer
        # Check first output to see if values are between 0 and 1
        test_output = decoder(images[0:1])
        is_sigmoid_output = (test_output >= 0).all() and (test_output <= 1).all()
    
    # Process each sub-batch separately to save memory
    for sub_batch_idx in range(num_sub_batches):
        start_idx = sub_batch_idx * max_sub_batch_size
        end_idx = min((sub_batch_idx + 1) * max_sub_batch_size, current_batch_size)
        sub_batch_size = end_idx - start_idx
        
        with torch.no_grad():
            # Get sub-batch of images
            sub_images = images[start_idx:end_idx].clone()
            
            # Initialize perturbation randomly within constraints
            perturbation = torch.zeros_like(sub_images, requires_grad=False)
            perturbation.uniform_(-max_delta, max_delta)
            
            # Make sure initial perturbation is valid
            adv_images = torch.clamp(sub_images + perturbation, -1, 1)
            perturbation.data = adv_images - sub_images
            
            # For momentum-based PGD
            grad_momentum = torch.zeros_like(perturbation)
            
            # For PGD optimization
            best_adv_images = None
            best_loss = float('inf')
            best_score = 0
            initial_score = None
            last_loss = 0
            stagnation_counter = 0
            
            # Create target tensor - PGD tries to maximize the probability of being watermarked
            target = torch.ones((sub_batch_size, 1), device=device)
        
        # Perform PGD attack with memory optimization
        for step in range(num_steps):
            # Free up memory
            torch.cuda.empty_cache()
            
            # Make sure perturbation requires gradients for this step
            perturbation.requires_grad_(True)
            
            # Create adversarial examples
            adv_images = torch.clamp(sub_images + perturbation, -1, 1)
            
            # Apply masking if enabled - process in small batches to save memory
            if key_type != "none":
                if k_mask is None:
                    # Generate mask if not provided
                    mask_shape = (sub_batch_size,) + sub_images.shape[1:]
                    k_mask_batch = generate_mask_secret_key(mask_shape, 2024, device, key_type=key_type)
                    
                    # Use smaller sub-batches for masking
                    decoder_input = mask_image_with_key(adv_images, k_mask_batch)
                else:
                    # Use the provided mask - handle both tensor and CryptoCNN cases
                    # CryptoCNN objects should be used directly without slicing
                    if hasattr(k_mask, 'size') and k_mask.size(0) > 1:
                        # It's a tensor with batch dimension
                        mask_slice = k_mask[start_idx:end_idx]
                        decoder_input = mask_image_with_key(adv_images, mask_slice)
                    else:
                        # It's either a single-batch tensor or a CryptoCNN object
                        decoder_input = mask_image_with_key(adv_images, k_mask)
            else:
                decoder_input = adv_images
            
            # Forward pass through decoder
            outputs = decoder(decoder_input)
            
            # Apply sigmoid if needed
            if is_sigmoid_output:
                probs = outputs
            else:
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
            
            # Compute gradients
            grad = torch.autograd.grad(loss, perturbation, create_graph=False, retain_graph=False)[0]
            
            # Free up memory explicitly
            outputs = None
            probs = None
            loss = None
            decoder_input = None
            
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
                adv_images = torch.clamp(sub_images + perturbation, -1, 1)
                
                # Recompute perturbation based on clamped images
                perturbation.data = adv_images - sub_images
            
            # Free up memory
            update = None
            adv_images = None
            torch.cuda.empty_cache()
        
        # Log sub-batch attack results only at rank 0
        if initial_score is not None and best_score is not None and rank == 0 and sub_batch_idx == 0:
            improvement = best_score - initial_score
            pct_improvement = (improvement / max(initial_score, 0.0001)) * 100
            # logging.info(
            #     f"\nPGD Attack Summary (Sub-batch {sub_batch_idx+1}/{num_sub_batches}):"
            #     f"\n{'='*30}"
            #     f"\nInitial Score: {initial_score:.4f}"
            #     f"\nBest Score: {best_score:.4f}"
            #     f"\nImprovement: {improvement:.4f} ({pct_improvement:.1f}%)"
            #     f"\n{'='*30}"
            # )
        
        # Store the best adversarial examples for this sub-batch
        with torch.no_grad():
            all_best_adv_images[start_idx:end_idx] = best_adv_images.clone()
        
        # Free up memory
        sub_images = None
        perturbation = None
        grad_momentum = None
        best_adv_images = None
        torch.cuda.empty_cache()
    
    return all_best_adv_images 