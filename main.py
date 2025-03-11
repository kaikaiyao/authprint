import argparse 
import os
import torch
import torch.distributed as dist
import pprint
import logging  # Added to use logging.info
import time
import csv

from models.stylegan2 import load_stylegan2_model
from models.gan import load_gan_model
from models.decoder import FlexibleDecoder
from models.attack_combined_model import CombinedModel
from models.model_utils import clone_model, load_finetuned_model
from utils.gpu import get_gpu_info, initialize_cuda
from utils.file_utils import generate_time_based_string
from utils.logging import setup_logging, LogRankFilter

from training.train_model import train_model
from training.finetune_decoder import finetune_decoder
from evaluation.evaluate_model import evaluate_model
from attack.attacks import attack_label_based

def main():
    parser = argparse.ArgumentParser(description="Run training or evaluation for the model.")
    parser.add_argument("mode", choices=["train", "eval", "attack", "finetune"], help="Mode to run the script in")
    
    # Common arguments
    parser.add_argument("--seed_key", type=int, default=2024, help="Seed for the random authentication key")
    parser.add_argument("--stylegan2_url", type=str, default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl", help="URL to load the StyleGAN2 model from")
    parser.add_argument("--self_trained", type=bool, default=False, help="Use a self-trained GAN model")
    parser.add_argument("--self_trained_model_path", type=str, default="generator.pth", help="Path to the self-trained GAN model")
    parser.add_argument("--self_trained_latent_dim", type=int, default=128, help="Latent dim for self-trained GAN")
    parser.add_argument("--saving_path", type=str, default="results", help="Path to save all related results.")

    # Decoder arguments
    parser.add_argument("--num_conv_layers", type=int, default=5, help="Total number of convolutional layers in the model")
    parser.add_argument("--num_pool_layers", type=int, default=5, help="Total number of pooling layers in the model")
    parser.add_argument("--initial_channels", type=int, default=64, help="Initial number of channels for the first convolutional layer")
    parser.add_argument("--num_conv_layers_surr", type=int, default=5, help="Total number of convolutional layers in the model")
    parser.add_argument("--num_pool_layers_surr", type=int, default=5, help="Total number of pooling layers in the model")
    parser.add_argument("--initial_channels_surr", type=int, default=64, help="Initial number of channels for the first convolutional layer")

    # Training arguments
    parser.add_argument("--n_iterations", type=int, default=20000, help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr_M_hat", type=float, default=1e-4, help="Learning rate for the watermarked model")
    parser.add_argument("--lr_D", type=float, default=1e-4, help="Learning rate for the decoder")
    parser.add_argument("--max_delta", type=float, default=0.01, help="Maximum allowed change per pixel (infinite norm constraint)")
    parser.add_argument("--run_eval", type=bool, default=True, help="Run evaluation function during training")
    parser.add_argument("--mask_switch_on", action="store_true", help="Enable the new masking pipeline")
    parser.set_defaults(mask_switch_on=False)
    parser.add_argument("--mask_switch_off", dest="mask_switch_on", action="store_false", help="Disable the new masking pipeline")
    parser.add_argument("--random_smooth", action="store_true", help="Enable randomized smoothing by adding Gaussian noise to images before masking")
    parser.set_defaults(random_smooth=False)
    parser.add_argument("--random_smooth_type", type=str, default="original", choices=["original", "both"], 
                        help="Where to apply noise: 'original' for x_M only, 'both' for both x_M and x_M_hat_constrained")
    parser.add_argument("--random_smooth_std", type=float, default=0.01, help="Standard deviation of Gaussian noise for randomized smoothing")
    parser.add_argument("--resume_checkpoint", type=str, help="Path to a checkpoint file to resume training")

    # Evaluation arguments
    parser.add_argument("--num_eval_samples", type=int, default=100, help="Number of images to evaluate")
    parser.add_argument("--watermarked_model_path", type=str, default="watermarked_model.pkl", help="Path to the finetuned watermarked model")
    parser.add_argument("--decoder_model_path", type=str, default="decoder_model.pth", help="Path to the decoder model state dictionary")
    parser.add_argument("--plotting", type=bool, default=False, help="To plot the results of the evaluation")
    parser.add_argument("--compute_fid", action="store_true", help="Whether to compute FID score during evaluation")
    parser.add_argument("--flip_key_type", type=str, default="none", choices=["none", "1", "10", "random"], help="Whether and how to flip the encryption key")
    parser.add_argument("--key_type", type=str, default="csprng", choices=["none", "encryption", "csprng"], help="Type of key generation method (encryption or csprng, or none representing the baseline pipeline)")

    # Decoder finetuning arguments
    parser.add_argument("--finetune_decoder", action="store_true", help="Enable decoder finetuning with adversarial examples")
    parser.add_argument("--finetune_epochs", type=int, default=10, help="Number of epochs for decoder finetuning")
    parser.add_argument("--finetune_lr", type=float, default=1e-4, help="Learning rate for decoder finetuning")
    parser.add_argument("--finetune_batch_size", type=int, default=16, help="Batch size for decoder finetuning")
    parser.add_argument("--finetune_pgd_steps", type=int, default=200, help="Number of PGD steps for generating adversarial examples during finetuning")
    parser.add_argument("--finetune_pgd_alpha", type=float, default=0.01, help="Step size for PGD attack during finetuning")

    # Attack arguments
    parser.add_argument("--attack_type", type=str, default="base_baseline", choices=["base_baseline", "base_secure", "combined_secure", "fixed_secure"], help="Attack type")
    parser.add_argument("--train_size", type=int, default=100000, help="training set size for training surrogate decoder")
    parser.add_argument("--image_attack_size", type=int, default=1000, help="size of attack image set")
    parser.add_argument("--surrogate_decoder_folder", type=str, help="Path to a folder containing surrogate decoder models (files starting with 'surrogate_decoder_')")
    parser.add_argument("--batch_size_surr", type=int, default=16, help="Batch size for training the surrogate decoder")
    parser.add_argument("--num_steps_pgd", type=int, default=200, help="Number of steps during the attack")
    parser.add_argument("--alpha_values_pgd", type=str, default="0.01", help="Alpha values for the attack (comma-separated list of floats)")
    parser.add_argument("--attack_batch_size_pgd", type=int, default=10, help="Batch size for the attack")
    parser.add_argument("--momentum_pgd", type=float, default=0.9, help="Momentum factor for PGD attack")
    parser.add_argument("--finetune_surrogate", action="store_true", help="Fine-tune surrogate with real decoder outputs")
    parser.add_argument("--surrogate_training_only", action="store_true", help="Only perform surrogate training (Step 4) and skip all other attack steps")

    # DDP arguments
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")

    args = parser.parse_args()

    # Distributed training setup
    if args.mode == "train" or (args.mode == "finetune" and args.finetune_decoder):
        dist.init_process_group(backend='nccl', init_method='env://')
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
    else:
        device = initialize_cuda()
        # If not running in distributed mode, set rank to 0 for logging purposes.
        args.rank = 0

    time_string = generate_time_based_string()

    os.makedirs(args.saving_path, exist_ok=True)

    # Set up logging with rank-aware configuration
    log_file = os.path.join(args.saving_path, f'training_log_{time_string}.txt')
    setup_logging(log_file, args.rank)
    
    # Only rank 0 should log anything from this point on
    if args.rank == 0:
        logging.info("===== Input Parameters =====")
        logging.info(pprint.pformat(vars(args)))
        logging.info("============================\n")
        get_gpu_info()

    if args.mode == "train":
        if args.self_trained:
            latent_dim = args.self_trained_latent_dim
            gan_model = load_gan_model(args.self_trained_model_path, latent_dim).to(device)
            watermarked_model = clone_model(gan_model).to(device)
            watermarked_model.train()
            decoder = FlexibleDecoder(
                args.num_conv_layers,
                args.num_pool_layers,
                args.initial_channels,
            ).to(device)
        else:
            local_path = args.stylegan2_url.split('/')[-1]
            gan_model = load_stylegan2_model(url=args.stylegan2_url, local_path=local_path, device=device)
            watermarked_model = clone_model(gan_model).to(device)
            watermarked_model.train()
            decoder = FlexibleDecoder(
                args.num_conv_layers,
                args.num_pool_layers,
                args.initial_channels,
            ).to(device)
            latent_dim = gan_model.z_dim

        # Wrap models with DDP
        watermarked_model = torch.nn.parallel.DistributedDataParallel(
            watermarked_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
        decoder = torch.nn.parallel.DistributedDataParallel(
            decoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )

        # Optimizers
        optimizer_D = torch.optim.Adagrad(decoder.parameters(), lr=args.lr_D)
        optimizer_M_hat = torch.optim.Adagrad(watermarked_model.parameters(), lr=args.lr_M_hat)

        # Load checkpoint
        start_iter = 0
        initial_loss_history = []
        if args.resume_checkpoint:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
            checkpoint = torch.load(args.resume_checkpoint, map_location=map_location)
            
            watermarked_model.module.load_state_dict(checkpoint['watermarked_model'])
            decoder.module.load_state_dict(checkpoint['decoder'])
            optimizer_M_hat.load_state_dict(checkpoint['optimizer_M_hat'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            
            start_iter = checkpoint['iteration'] + 1
            initial_loss_history = checkpoint['loss_history']
            
            args.n_iterations += start_iter
            if args.rank == 0:
                logging.info(f"Resuming training from iteration {start_iter}")

        train_model(
            time_string=time_string,
            gan_model=gan_model,
            watermarked_model=watermarked_model,
            decoder=decoder,
            n_iterations=args.n_iterations,
            latent_dim=latent_dim,
            batch_size=args.batch_size,
            device=device,
            run_eval=args.run_eval,
            num_images=args.num_eval_samples,
            plotting=args.plotting,
            max_delta=args.max_delta,
            saving_path=args.saving_path,
            mask_switch_on=args.mask_switch_on,
            seed_key=args.seed_key,
            optimizer_M_hat=optimizer_M_hat,
            optimizer_D=optimizer_D,
            start_iter=start_iter,
            initial_loss_history=initial_loss_history,
            rank=args.rank,
            world_size=args.world_size,
            key_type=args.key_type,
            compute_fid=args.compute_fid,
            flip_key_type=args.flip_key_type,
            random_smooth=args.random_smooth,
            random_smooth_type=args.random_smooth_type,
            random_smooth_std=args.random_smooth_std,
        )

    elif args.mode == "eval":
        local_path = args.stylegan2_url.split('/')[-1]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.self_trained:
            latent_dim = args.self_trained_latent_dim
            gan_model = load_gan_model(args.self_trained_model_path, latent_dim).to(device)
        else:
            gan_model = load_stylegan2_model(url=args.stylegan2_url, local_path=local_path, device=device)
            latent_dim = gan_model.z_dim

        watermarked_model = load_finetuned_model(args.watermarked_model_path)
        watermarked_model.to(device)
        watermarked_model.eval()

        decoder = FlexibleDecoder(
            args.num_conv_layers,
            args.num_pool_layers,
            args.initial_channels,
        ).to(device)
        decoder.load_state_dict(torch.load(args.decoder_model_path))
        decoder = decoder.to(device)

        logging.info(f"Plotting: {args.plotting}")
        
        eval_results = evaluate_model(
            num_images=args.num_eval_samples,
            gan_model=gan_model,
            watermarked_model=watermarked_model,
            decoder=decoder,
            device=device,
            plotting=args.plotting,
            latent_dim=latent_dim,
            max_delta=args.max_delta,
            mask_switch_on=args.mask_switch_on,
            seed_key=args.seed_key,
            flip_key_type=args.flip_key_type,
            compute_fid=args.compute_fid,
            key_type=args.key_type,
            rank=args.rank,
            batch_size=args.batch_size
        )

        # Only process results on rank 0
        if args.rank == 0:
            # Extract metrics from the results dictionary
            auc = eval_results['auc']
            tpr_at_1_fpr = eval_results['watermarked']['tpr_at_1_fpr']
            fid_score = eval_results['fid_score']
            mean_score_watermarked = eval_results['watermarked']['mean_score']
            std_score_watermarked = eval_results['watermarked']['std_score']
            mean_score_original = eval_results['original']['mean_score']
            std_score_original = eval_results['original']['std_score']

            # Log the comprehensive results
            logging.info("Evaluation Results:")
            logging.info(f"AUC score: {auc:.10f}")
            logging.info(f"TPR@1%FPR (watermarked): {tpr_at_1_fpr:.10f}")
            if fid_score is not None:
                logging.info(f"FID score: {fid_score:.4f}")
            logging.info(f"Mean score (watermarked): {mean_score_watermarked:.10f}")
            logging.info(f"Std score (watermarked): {std_score_watermarked:.10f}")
            logging.info(f"Mean score (original): {mean_score_original:.10f}")
            logging.info(f"Std score (original): {std_score_original:.10f}")

            # Log TNR@1%FPR for other cases
            for case_name, case_data in eval_results.items():
                if case_name not in ['auc', 'fid_score', 'watermarked']:
                    if 'tnr_at_1_fpr' in case_data:
                        logging.info(f"TNR@1%FPR ({case_name}): {case_data['tnr_at_1_fpr']:.10f}")

    elif args.mode == "finetune":
        logging.info("Starting decoder finetuning...")
        
        # Generate a time string for file naming
        time_string = generate_time_based_string()
        
        # Load the GAN model
        local_path = args.stylegan2_url.split('/')[-1]
        
        if args.self_trained:
            latent_dim = args.self_trained_latent_dim
            gan_model = load_gan_model(args.self_trained_model_path, latent_dim).to(device)
        else:
            gan_model = load_stylegan2_model(url=args.stylegan2_url, local_path=local_path, device=device)
            latent_dim = gan_model.z_dim

        # Load the watermarked model
        watermarked_model = load_finetuned_model(args.watermarked_model_path)
        watermarked_model.to(device)
        watermarked_model.eval()

        # Load the decoder
        decoder = FlexibleDecoder(
            args.num_conv_layers,
            args.num_pool_layers,
            args.initial_channels,
        ).to(device)
        decoder.load_state_dict(torch.load(args.decoder_model_path))
        decoder = decoder.to(device)
        
        # Wrap decoder with DDP
        if torch.distributed.is_initialized():
            decoder = torch.nn.parallel.DistributedDataParallel(
                decoder,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )
        
        # Finetune the decoder
        finetuned_decoder = finetune_decoder(
            time_string=time_string,
            gan_model=gan_model,
            watermarked_model=watermarked_model,
            decoder=decoder,
            latent_dim=latent_dim,
            batch_size=args.finetune_batch_size,
            device=device,
            num_epochs=args.finetune_epochs,
            learning_rate=args.finetune_lr,
            max_delta=args.max_delta,
            saving_path=args.saving_path,
            mask_switch_on=args.mask_switch_on,
            seed_key=args.seed_key,
            rank=args.rank,
            world_size=args.world_size if torch.distributed.is_initialized() else 1,
            key_type=args.key_type,
            pgd_steps=args.finetune_pgd_steps,
            pgd_alpha=args.finetune_pgd_alpha
        )
        
        if args.rank == 0:
            logging.info("Decoder finetuning completed successfully.")

    elif args.mode == "attack":
        local_path = args.stylegan2_url.split('/')[-1]

        # Check if pre-trained surrogate decoder paths are provided
        if args.surrogate_decoder_folder is not None:
            train_surrogate = False
            # Check if folder exists
            if not os.path.exists(args.surrogate_decoder_folder):
                raise ValueError(f"Surrogate decoder folder {args.surrogate_decoder_folder} does not exist")
            
            # Get all files starting with 'surrogate_decoder_'
            surrogate_paths = [f for f in os.listdir(args.surrogate_decoder_folder) if f.startswith('surrogate_decoder_')]
            
            if not surrogate_paths:
                logging.warning(f"No files starting with 'surrogate_decoder_' found in {args.surrogate_decoder_folder}")
                train_surrogate = True
                surrogate_paths = [None]  # Will create one surrogate decoder
        else:
            train_surrogate = True
            surrogate_paths = [None]  # Will create one surrogate decoder

        # Convert alpha_values string to a vector of values
        args.alpha_values_pgd = [float(x) for x in args.alpha_values_pgd.split(',')]

        # Initialize device and DDP if training surrogate decoder
        if train_surrogate:
            dist.init_process_group(backend='nccl', init_method='env://')
            args.local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(args.local_rank)
            device = torch.device('cuda', args.local_rank)
            args.world_size = dist.get_world_size()
            args.rank = dist.get_rank()
        else:
            device = initialize_cuda()
            args.rank = 0

        if args.self_trained:
            latent_dim = args.self_trained_latent_dim
            gan_model = load_gan_model(args.self_trained_model_path, latent_dim).to(device)
        else:
            gan_model = load_stylegan2_model(url=args.stylegan2_url, local_path=local_path, device=device)
            latent_dim = gan_model.z_dim

        watermarked_model = load_finetuned_model(args.watermarked_model_path)
        watermarked_model.to(device)
        watermarked_model.eval()

        decoder = FlexibleDecoder(
            args.num_conv_layers,
            args.num_pool_layers,
            args.initial_channels,
        ).to(device)
        decoder.load_state_dict(torch.load(args.decoder_model_path))
        decoder = decoder.to(device)

        # Initialize list to store surrogate decoders
        surrogate_decoders = []

        # Create or load surrogate decoders
        for surrogate_path in surrogate_paths:
            if args.attack_type in ["base_baseline", "base_secure"]:
                surrogate_decoder = FlexibleDecoder(
                    args.num_conv_layers_surr,
                    args.num_pool_layers_surr,
                    args.initial_channels_surr,
                ).to(device)
            elif args.attack_type in ["combined_secure"]:
                surrogate_decoder = CombinedModel(
                    input_channels=3,
                    decoder_total_conv_layers=args.num_conv_layers_surr,
                    decoder_total_pool_layers=args.num_pool_layers_surr,
                    decoder_initial_channels=args.initial_channels_surr,
                    cnn_instance=None,
                    cnn_mode="fresh",
                ).to(device)
            elif args.attack_type in ["fixed_secure"]:
                from key.key import generate_mask_secret_key
                mask_cnn = generate_mask_secret_key(
                    image_shape=(1, 3, 256, 256),
                    seed=2024,
                    device=device,
                    key_type=args.key_type,
                )
                surrogate_decoder = CombinedModel(
                    input_channels=3,
                    decoder_total_conv_layers=args.num_conv_layers_surr,
                    decoder_total_pool_layers=args.num_pool_layers_surr,
                    decoder_initial_channels=args.initial_channels_surr,
                    cnn_instance=mask_cnn,
                    cnn_mode="fixed",
                ).to(device)

            # Wrap surrogate_decoder with DDP if training
            if train_surrogate:
                surrogate_decoder = torch.nn.parallel.DistributedDataParallel(
                    surrogate_decoder,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank
                )

            # Load pre-trained weights if provided
            if surrogate_path is not None:
                full_path = os.path.join(args.surrogate_decoder_folder, surrogate_path)
                logging.info(f"Loading pre-trained surrogate decoder from {full_path}")
                state_dict = torch.load(full_path, map_location=device)
                # If DDP is active, load into the module
                if train_surrogate:
                    # Remove 'module.' prefix from state dict keys if present
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        if k.startswith('module.'):
                            new_state_dict[k[7:]] = v  # Remove 'module.' prefix (7 characters)
                        else:
                            new_state_dict[k] = v
                    surrogate_decoder.module.load_state_dict(new_state_dict)
                else:
                    # Remove 'module.' prefix from state dict keys if present
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        if k.startswith('module.'):
                            new_state_dict[k[7:]] = v  # Remove 'module.' prefix (7 characters)
                        else:
                            new_state_dict[k] = v
                    surrogate_decoder.load_state_dict(new_state_dict)

            surrogate_decoders.append(surrogate_decoder)

        if train_surrogate:
            logging.info(f"Training {len(surrogate_decoders)} surrogate decoder(s) from scratch.")
        else:
            logging.info(f"Using {len(surrogate_decoders)} pre-trained surrogate decoder(s).")

        attack_auc, attack_success_rates, attack_results, per_case_auc = attack_label_based(
            attack_type=args.attack_type,
            gan_model=gan_model,
            watermarked_model=watermarked_model,
            max_delta=args.max_delta,
            decoder=decoder,
            surrogate_decoders=surrogate_decoders,
            latent_dim=latent_dim,
            device=device,
            train_size=args.train_size,
            image_attack_size=args.image_attack_size,
            batch_size=args.batch_size_surr,
            epochs=1,
            attack_batch_size=args.attack_batch_size_pgd,
            num_steps=args.num_steps_pgd,
            alpha_values=args.alpha_values_pgd,
            train_surrogate=train_surrogate,
            finetune_surrogate=args.finetune_surrogate,
            rank=args.rank,
            world_size=args.world_size if train_surrogate else 1,
            momentum=args.momentum_pgd,
            key_type=args.key_type,
            surrogate_training_only=args.surrogate_training_only,
            saving_path=args.saving_path,
        )
        
        if args.rank == 0 and attack_auc is not None:
            # Log the per-case AUC values to a CSV file for later analysis
            os.makedirs('results', exist_ok=True)
            time_str = time.strftime("%Y%m%d-%H%M%S")
            csv_path = f"results/attack_metrics_{time_str}.csv"
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Attack Case', 'ASR@1%FPR', 'ROC-AUC', 'Mean Score', 'Std Dev'])
                
                for case_name in attack_success_rates.keys():
                    asr = attack_success_rates[case_name]
                    auc = per_case_auc[case_name]
                    mean_score = attack_results[case_name]['mean'][0]
                    std_score = attack_results[case_name]['std'][0]
                    writer.writerow([case_name, f"{asr:.2f}%", f"{auc:.4f}", f"{mean_score:.4f}", f"{std_score:.4f}"])
                
                # Add overall metrics
                writer.writerow(['Overall', '', f"{attack_auc:.4f}", '', ''])
            
            logging.info(f"Attack metrics saved to {csv_path}")


if __name__ == "__main__":
    main()
