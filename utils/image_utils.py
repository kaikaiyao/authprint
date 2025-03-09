import torch
import matplotlib.pyplot as plt
import PIL.Image
import io
import torchvision.transforms as transforms
import torch.nn.functional as F
import copy


def constrain_image(x_M_hat, x_M, max_delta):
    # Calculate the difference (delta) between predicted and original images
    delta = x_M_hat - x_M
    
    # Calculate mean of delta and subtract it to remove mean bias
    delta_mean = delta.mean(dim=(2, 3), keepdim=True)
    delta_adjusted = delta - delta_mean

    # Find original min and max range of delta
    delta_min, delta_max = delta.amin(dim=(2, 3), keepdim=True), delta.amax(dim=(2, 3), keepdim=True)
    delta_range = delta_max - delta_min
    
    # Adjusted range for delta_adjusted
    delta_adjusted_min, delta_adjusted_max = delta_adjusted.amin(dim=(2, 3), keepdim=True), delta_adjusted.amax(dim=(2, 3), keepdim=True)
    delta_adjusted_range = delta_adjusted_max - delta_adjusted_min

    # Scale delta_adjusted to match the original range of delta
    delta_adjusted_scaled = delta_adjusted * (delta_range / delta_adjusted_range)

    # Clip the scaled delta to the specified max_delta
    delta_clipped = torch.clamp(delta_adjusted_scaled, min=-max_delta, max=max_delta)
    
    # Compute constrained output
    x_M_hat_constrained = x_M + delta_clipped

    del delta, delta_mean, delta_adjusted, delta_adjusted_scaled, delta_clipped

    return x_M_hat_constrained


def plot_variables(z, x_M, x_M_hat, k_M, k_M_hat):
    x_M_diff = x_M[0].squeeze() - x_M_hat[0].squeeze()
    tensors = {
        "z": z[0].squeeze(),
        "x_M": x_M[0].squeeze(),
        "x_M_hat": x_M_hat[0].squeeze(),
        "x_M_diff": x_M_diff,
        "k_M": k_M[0].squeeze(),
        "k_M_hat": k_M_hat[0].squeeze(),
    }
    for title, tensor in tensors.items():
        if tensor.dim() == 2:
            plt.figure(figsize=(6, 1))
            plt.imshow(tensor.cpu().numpy(), cmap="viridis", interpolation='none')
        elif tensor.dim() == 3 and tensor.size(0) == 3:
            if title == "x_M_diff":
                diff_tensor = tensor
                diff_tensor = diff_tensor.detach().cpu()
                max_val = torch.max(torch.abs(diff_tensor))
                if max_val > 0:
                    diff_tensor = diff_tensor / max_val
                diff_tensor = diff_tensor.permute(1, 2, 0)
                diff_array = diff_tensor.numpy()
                plt.figure(figsize=(6, 6))
                img = plt.imshow(diff_array, cmap='RdBu', vmin=-1, vmax=1)
                plt.colorbar(img, extend='both', orientation='vertical')
                plt.title(title)
                plt.axis("off")
                plt.savefig(f"{title}.png")
                plt.close()
                continue
            else:
                tensor = (
                    (tensor.permute(1, 2, 0) * 127.5 + 128)
                    .clamp(0, 255)
                    .to(torch.uint8)
                )
                PIL.Image.fromarray(tensor.detach().cpu().numpy(), "RGB").save(
                    f"{title}.png"
                )
                continue
        else:
            plt.figure(figsize=(6, 1))
            plt.imshow(tensor.detach().cpu().numpy().reshape(1, -1), cmap="viridis", interpolation='none')
        plt.title(title)
        plt.colorbar()
        plt.axis("off")
        plt.savefig(f"{title}.png")
        plt.close()


def apply_truncation(model, z, truncation_psi=0.5):
    """Generate images with specified truncation level.
    
    Args:
        model: StyleGAN2 model
        z: Input latent vectors
        truncation_psi: Truncation level (lower = more average looking, less variation)
    
    Returns:
        Generated images with specified truncation
    """
    with torch.no_grad():
        return model(z, None, truncation_psi=truncation_psi, noise_mode="const")


def quantize_model_weights(model, precision='int8'):
    """Quantize model weights to specified precision.
    
    Args:
        model: The model to quantize
        precision: Quantization precision ('int8')
    
    Returns:
        Quantized model copy
    """
    quantized_model = copy.deepcopy(model)
    
    def quantize_tensor(tensor):
        # Scale to int8 range
        scale = 127.0 / torch.max(torch.abs(tensor))
        quantized = torch.round(tensor * scale)
        quantized = torch.clamp(quantized, -127, 127)
        # Scale back to original range
        dequantized = quantized / scale
        return dequantized
    
    # Quantize all parameters
    with torch.no_grad():
        for param in quantized_model.parameters():
            param.copy_(quantize_tensor(param))
    
    return quantized_model


def downsample_and_upsample(images, downsample_size=128):
    """Downsample images and then upsample back to original size.
    
    Args:
        images: Input images tensor (B, C, H, W)
        downsample_size: Size to downsample to
    
    Returns:
        Downsampled then upsampled images
    """
    original_size = images.shape[-1]
    # Downsample
    downsampled = F.interpolate(images, size=downsample_size, mode='bilinear', align_corners=False)
    # Upsample back to original size
    upsampled = F.interpolate(downsampled, size=original_size, mode='bilinear', align_corners=False)
    return upsampled


def apply_jpeg_compression(images, quality=55):
    """Apply JPEG compression to images.
    
    Args:
        images: Input images tensor (B, C, H, W) in range [-1, 1]
        quality: JPEG quality (0-100)
    
    Returns:
        Compressed images tensor
    """
    device = images.device
    compressed_batch = []
    
    # Convert to PIL, compress, and back to tensor
    for img in images:
        # Convert to PIL Image (rescale to 0-255 range)
        img_np = ((img.permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
        img_pil = PIL.Image.fromarray(img_np)
        
        # Apply JPEG compression
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = PIL.Image.open(buffer)
        
        # Convert back to tensor and rescale to [-1, 1]
        compressed_tensor = transforms.ToTensor()(compressed_img) * 2 - 1
        compressed_batch.append(compressed_tensor)
    
    # Stack back into batch
    return torch.stack(compressed_batch).to(device)
