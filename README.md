# StyleGAN Watermarking

This project implements a watermarking technique for StyleGAN2 models. It trains a model to embed imperceptible watermarks in generated images and a decoder to extract watermark keys.

## Project Structure

```
.
├── config/                 # Configuration management
├── models/                 # Model definitions
├── trainers/               # Training logic
├── utils/                  # Utility functions
├── scripts/                # Training and evaluation scripts
├── train.py                # Main entry point
├── requirements.txt        # Dependencies
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NVIDIA GPU with CUDA support
- StyleGAN2-ADA-PyTorch (clone from https://github.com/NVlabs/stylegan2-ada-pytorch.git)

## Installation

1. Clone the StyleGAN2-ADA repository:
```bash
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a Watermarked Model

To train a watermarked StyleGAN2 model, use the main entry point:

```bash
python train.py --batch_size 16 --total_iterations 100000 --key_length 4 --lambda_lpips 1.0 --output_dir results
```

Or run the script directly:

```bash
python scripts/train.py --batch_size 16 --total_iterations 100000 --key_length 4 --lambda_lpips 1.0 --output_dir results
```

### Distributed Training

For distributed training on multiple GPUs:

```bash
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12345 scripts/train.py --batch_size 16 --total_iterations 100000
```

## Configuration

The training can be customized with various parameters:

- `--stylegan2_url`: URL for pretrained StyleGAN2 model
- `--stylegan2_local_path`: Local path to store/load StyleGAN2 model
- `--img_size`: Image resolution
- `--batch_size`: Batch size for training
- `--total_iterations`: Total number of training iterations
- `--lr`: Learning rate
- `--lambda_lpips`: Weight for LPIPS loss
- `--key_length`: Length of the binary key
- `--selected_indices`: Indices to select for latent partial vector
- `--output_dir`: Directory for saving logs and checkpoints
- `--log_interval`: Interval for logging progress
- `--checkpoint_interval`: Interval for saving checkpoints
- `--seed`: Random seed for reproducibility

## Key Components

1. **StyleGAN2 Model**: Base generator used for image synthesis
2. **Watermarked Model**: A fine-tuned version of StyleGAN2 that embeds watermarks
3. **KeyMapper**: Maps latent partial vectors to binary keys
4. **Decoder**: Predicts binary keys from watermarked images

## Method

The watermarking process works as follows:

1. Sample latent vectors and generate images using the watermarked model
2. Extract partial information from the latent space
3. Map this partial information to binary keys using a secret mapping
4. Optimize the watermarked model so that a decoder can extract these keys from the generated images
5. Constrain image modifications using LPIPS loss to keep watermarks imperceptible

## License

This project is available under [insert license here]. 