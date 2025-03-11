import torch
import os
import gc
import numpy as np
import logging
from torch.nn.parallel import DistributedDataParallel as DDP

from models.model_utils import save_finetuned_model
from evaluation.evaluate_model import evaluate_model
from models.stylegan2 import is_stylegan2
from utils.image_utils import constrain_image
from key.key import generate_mask_secret_key, mask_image_with_key
from utils.logging import LogRankFilter

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def train_model(
    time_string,
    gan_model,
    watermarked_model,
    decoder,
    n_iterations,
    latent_dim,
    batch_size,
    device,
    run_eval,
    num_images,
    plotting,
    max_delta,
    saving_path,
    mask_switch_on,
    seed_key,
    optimizer_M_hat,
    optimizer_D,
    start_iter=0,
    initial_loss_history=None,
    rank=0,
    world_size=1,
    key_type="csprng",
    compute_fid=False,
    flip_key_type="none",
    random_smooth=False,
    random_smooth_type="original",
    random_smooth_std=0.01,
):
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
        logging.info(f"World size: {world_size}")
        logging.info(f"max_delta = {max_delta}")
        logging.info(f"time_string = {time_string}")
        logging.info(f"mask_switch_on = {mask_switch_on}")
        logging.info(f"key_type = {key_type}")
        logging.info(f"flip_key_type = {flip_key_type}")
        if random_smooth:
            logging.info(f"Random smoothing enabled with type '{random_smooth_type}' and std {random_smooth_std}")
        else:
            logging.info(f"Random smoothing disabled")
        logging.info("Decoder structure:\n%s", decoder.module if isinstance(decoder, DDP) else decoder)
        logging.info(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters())}")

    gan_model.eval()
    watermarked_model.train()
    decoder.train()

    loss_history = initial_loss_history if initial_loss_history is not None else []

    for i in range(start_iter, n_iterations):
        torch.cuda.empty_cache()
        gc.collect()

        # Generate different noise on each device
        torch.manual_seed(seed_key + i * world_size + rank)
        z = torch.randn((batch_size, latent_dim), device=device)

        with torch.no_grad():
            if is_stylegan2(gan_model):
                x_M = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
            else:
                x_M = gan_model(z)

        if is_stylegan2(gan_model):
            x_M_hat = watermarked_model(z, None, truncation_psi=1.0, noise_mode="const")
        else:
            x_M_hat = watermarked_model(z)
        
        x_M_hat_constrained = constrain_image(x_M_hat, x_M, max_delta)

        # Apply randomized smoothing if enabled, regardless of mask_switch_on
        if random_smooth:
            if random_smooth_type == "original" or random_smooth_type == "both":
                # Add Gaussian noise to original image
                x_M = x_M + torch.randn_like(x_M) * random_smooth_std
            
            if random_smooth_type == "both":
                # Add Gaussian noise to watermarked image
                x_M_hat_constrained = x_M_hat_constrained + torch.randn_like(x_M_hat_constrained) * random_smooth_std

        if mask_switch_on:
            if i == 0 or start_iter != 0:
                k_mask = generate_mask_secret_key(
                    image_shape=x_M_hat_constrained.shape,
                    seed=seed_key,
                    device=device,
                    flip_key_type=flip_key_type,
                    key_type=key_type
                )

            x_M_original = x_M.clone().detach()
            x_M_hat_constrained_original = x_M_hat_constrained.clone().detach()

            x_M = mask_image_with_key(images=x_M, cnn_key=k_mask)
            x_M_hat_constrained = mask_image_with_key(images=x_M_hat_constrained, cnn_key=k_mask)

            if plotting and rank == 0 and (i == 0 or i == 99999):
                os.makedirs(saving_path, exist_ok=True)
                with PdfPages(os.path.join(saving_path, 'first_iteration_images.pdf')) as pdf:
                    fig, axes = plt.subplots(4, 8, figsize=(20, 12))
                    fig.suptitle("First Iteration Images", fontsize=16)

                    for idx in range(8):
                        def normalize_image(img):
                            img = img.cpu().detach().numpy()
                            img_min = img.min()
                            img_max = img.max()
                            img = (img - img_min) / (img_max - img_min) * 255.0
                            return img.astype('uint8')

                        axes[0, idx].imshow(normalize_image(x_M_original[idx].permute(1, 2, 0)))
                        axes[0, idx].set_title(f"Original x_M {idx+1}")
                        axes[0, idx].axis('off')

                        axes[1, idx].imshow(normalize_image(x_M[idx].permute(1, 2, 0)))
                        axes[1, idx].set_title(f"Modified x_M {idx+1}")
                        axes[1, idx].axis('off')

                        axes[2, idx].imshow(normalize_image(x_M_hat_constrained_original[idx].permute(1, 2, 0)))
                        axes[2, idx].set_title(f"Original x_M_hat {idx+1}")
                        axes[2, idx].axis('off')

                        axes[3, idx].imshow(normalize_image(x_M_hat_constrained[idx].permute(1, 2, 0)))
                        axes[3, idx].set_title(f"Modified x_M_hat {idx+1}")
                        axes[3, idx].axis('off')

                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

        x_M = x_M.detach()

        k_M = decoder(x_M)
        k_M_hat = decoder(x_M_hat_constrained)

        del x_M, x_M_hat, x_M_hat_constrained
        torch.cuda.empty_cache()

        d_k_M_hat = torch.norm(k_M_hat, dim=1) 
        d_k_M = torch.norm(k_M, dim=1) 

        del k_M, k_M_hat
        torch.cuda.empty_cache()

        # Sigmoid func in the final layer of the decoder is used to make the output range from 0 to 1, so d ranges from 0 to 1,
        # so the loss is to maximize the difference between the distance of the prediction on the original image and the watermarked image
        # Note:
        # 1. the loss is squared to make it more sensitive to the difference
        # 2. the max is applied to get the maximum difference in all samples in one iteration (batch)
        loss = ((d_k_M - d_k_M_hat).max() + 1) ** 2 

        optimizer_M_hat.zero_grad(set_to_none=True)
        optimizer_D.zero_grad(set_to_none=True)

        loss.backward()

        optimizer_M_hat.step()
        optimizer_D.step()

        loss_history.append(loss.item())

        if rank == 0:
            logging.info(
                f"Train Iteration {i + 1}: "
                f"loss: {loss.item():.4f}, "
                f"d_k_M range: [{d_k_M.min().item():.4f}, {d_k_M.max().item():.4f}], "
                f"d_k_M_hat range: [{d_k_M_hat.min().item():.4f}, {d_k_M_hat.max().item():.4f}]"
            )

        del loss, d_k_M_hat, d_k_M
        torch.cuda.empty_cache()

    # Save final models
    if rank == 0:
        checkpoint = {
            'watermarked_model': watermarked_model.module.state_dict(),
            'decoder': decoder.module.state_dict(),
            'optimizer_M_hat': optimizer_M_hat.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'iteration': n_iterations - 1,
            'loss_history': loss_history,
        }
        checkpoint_path = os.path.join(saving_path, f'checkpoint_final_{time_string}.pt')
        torch.save(checkpoint, checkpoint_path)

        save_finetuned_model(
            model=watermarked_model.module,
            path=saving_path,
            filename=f'watermarked_model_final_{time_string}.pkl'
        )
        torch.save(decoder.module.state_dict(), os.path.join(saving_path, f'decoder_model_final_{time_string}.pth'))
        logging.info(f"Final models saved at iteration {n_iterations}, time_string = {time_string}")

    # Final evaluation and logging
    if rank == 0:
        logging.info("Training completed.")
        
        if run_eval:
            watermarked_model.eval()
            decoder.eval()
            
            with torch.no_grad():
                eval_results = evaluate_model(
                    num_images=num_images,
                    gan_model=gan_model,
                    watermarked_model=watermarked_model.module,
                    decoder=decoder.module,
                    device=device,
                    plotting=plotting,
                    latent_dim=latent_dim,
                    max_delta=max_delta,
                    mask_switch_on=mask_switch_on,
                    seed_key=seed_key,
                    flip_key_type=flip_key_type,
                    compute_fid=compute_fid,
                    key_type=key_type,
                    rank=rank,
                    batch_size=batch_size
                )
            
            # Extract metrics from the results dictionary (only on rank 0)
            auc = eval_results['auc']
            tpr_at_1_fpr = eval_results['watermarked']['tpr_at_1_fpr']
            fid_score = eval_results['fid_score']
            mean_score_watermarked = eval_results['watermarked']['mean_score']
            std_score_watermarked = eval_results['watermarked']['std_score']

            # Log the comprehensive results
            logging.info(f"Final evaluation after {n_iterations} iterations:")
            logging.info(f"AUC score: {auc:.4f}")
            logging.info(f"TPR@1%FPR (watermarked): {tpr_at_1_fpr:.4f}")
            if fid_score is not None:
                logging.info(f"FID score: {fid_score:.4f}")
            logging.info(f"Mean score (watermarked): {mean_score_watermarked:.4f}")
            logging.info(f"Std score (watermarked): {std_score_watermarked:.4f}")

            # Log TNR@1%FPR for other cases
            for case_name, case_data in eval_results.items():
                if case_name not in ['auc', 'fid_score', 'watermarked']:
                    if 'tnr_at_1_fpr' in case_data:
                        logging.info(f"TNR@1%FPR ({case_name}): {case_data['tnr_at_1_fpr']:.4f}")

    # Cleanup
    if rank == 0:
        logging.info("Training completed successfully.")