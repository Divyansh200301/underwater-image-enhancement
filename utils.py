"""
Utility functions for underwater image enhancement
Includes image transforms, save/load helpers, and metrics (PSNR, SSIM)
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms as transforms


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_image(image_path, size=(256, 256)):
    """
    Load image from path and resize
    
    Args:
        image_path: Path to image file
        size: Tuple (height, width) for resizing
        
    Returns:
        PIL Image
    """
    img = Image.open(image_path).convert('RGB')
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    return img


def save_image(tensor, path):
    """
    Save tensor as image
    
    Args:
        tensor: PyTorch tensor (C, H, W) or (B, C, H, W)
        path: Output file path
    """
    ensure_dir(os.path.dirname(path))
    
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image from batch
    
    # Convert to numpy and denormalize
    img = tensor.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))  # C, H, W -> H, W, C
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def tensor_to_numpy(tensor):
    """
    Convert PyTorch tensor to numpy array for visualization
    
    Args:
        tensor: PyTorch tensor (C, H, W) or (B, C, H, W)
        
    Returns:
        Numpy array (H, W, C) in range [0, 255]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    img = tensor.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def numpy_to_tensor(img):
    """
    Convert numpy array to PyTorch tensor
    
    Args:
        img: Numpy array (H, W, C) in range [0, 255]
        
    Returns:
        PyTorch tensor (1, C, H, W) in range [0, 1]
    """
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # H, W, C -> C, H, W
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor


def get_image_transform(size=(256, 256), normalize=True):
    """
    Get standard image transformation pipeline
    
    Args:
        size: Target size (height, width)
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        torchvision transforms composition
    """
    transform_list = [
        transforms.Resize((size[0], size[1])),
        transforms.ToTensor(),
    ]
    
    if not normalize:
        # If already normalized by ToTensor, no additional normalization
        pass
    
    return transforms.Compose(transform_list)


def calculate_psnr(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio between two images
    
    Args:
        img1, img2: Numpy arrays or PyTorch tensors
        
    Returns:
        PSNR value in dB
    """
    if isinstance(img1, torch.Tensor):
        img1 = tensor_to_numpy(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor_to_numpy(img2)
    
    return psnr(img1, img2, data_range=255)


def calculate_ssim(img1, img2):
    """
    Calculate Structural Similarity Index between two images
    
    Args:
        img1, img2: Numpy arrays or PyTorch tensors
        
    Returns:
        SSIM value between 0 and 1
    """
    if isinstance(img1, torch.Tensor):
        img1 = tensor_to_numpy(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor_to_numpy(img2)
    
    # Convert to grayscale for SSIM calculation
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    return ssim(img1_gray, img2_gray, data_range=255)


def merge_images(dehazed, clarity_enhanced, alpha=0.5):
    """
    Merge dehazed and clarity enhanced images
    
    Args:
        dehazed: Dehazed image tensor
        clarity_enhanced: Clarity enhanced image tensor
        alpha: Blending weight (0 to 1)
        
    Returns:
        Merged image tensor
    """
    merged = alpha * dehazed + (1 - alpha) * clarity_enhanced
    return torch.clamp(merged, 0, 1)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        path: Save path
    """
    ensure_dir(os.path.dirname(path))
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (can be None)
        path: Checkpoint path
        
    Returns:
        epoch, loss
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0)
    
    print(f"Checkpoint loaded from {path} (epoch {epoch})")
    return epoch, loss


def get_image_files(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """
    Get all image files from a directory
    
    Args:
        directory: Directory path
        extensions: Tuple of valid extensions
        
    Returns:
        List of image file paths
    """
    image_files = []
    if not os.path.exists(directory):
        return image_files
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(root, file))
    
    return sorted(image_files)


class AverageMeter:
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
