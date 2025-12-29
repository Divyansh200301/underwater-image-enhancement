"""
DataLoader for underwater image enhancement
Supports paired and unpaired training data
"""

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class UnderwaterDataset(Dataset):
    """
    Dataset class for underwater images
    
    Supports:
    - Paired training (input + reference)
    - Unpaired training (input only)
    """
    
    def __init__(self, raw_dir, reference_dir=None, challenging_dir=None, 
                 size=(256, 256), mode='train'):
        """
        Args:
            raw_dir: Directory with raw underwater images
            reference_dir: Directory with reference/ground truth images (optional)
            challenging_dir: Directory with challenging underwater images (optional)
            size: Resize dimensions (height, width)
            mode: 'train' or 'test'
        """
        self.raw_dir = raw_dir
        self.reference_dir = reference_dir
        self.challenging_dir = challenging_dir
        self.size = size
        self.mode = mode
        
        # Get image files
        self.raw_images = self._get_image_files(raw_dir)
        
        # Add challenging images if provided
        if challenging_dir and os.path.exists(challenging_dir):
            challenging_images = self._get_image_files(challenging_dir)
            self.raw_images.extend(challenging_images)
        
        # Check if we have paired references
        self.paired = False
        self.reference_images = []
        
        if reference_dir and os.path.exists(reference_dir):
            self.reference_images = self._get_image_files(reference_dir)
            # Simple pairing by matching filenames
            if len(self.reference_images) > 0:
                self.paired = True
                # Try to match files by name
                self._match_pairs()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((size[0], size[1])),
            transforms.ToTensor(),
        ])
        
        # Data augmentation for training
        if mode == 'train':
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        else:
            self.augment = None
        
        print(f"Loaded {len(self.raw_images)} images from {raw_dir}")
        if self.paired:
            print(f"Paired mode: {len(self.pairs)} matched pairs")
        else:
            print("Unpaired mode")
    
    def _get_image_files(self, directory):
        """Get all image files from directory"""
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist")
            return []
        
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')
        image_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(extensions):
                    image_files.append(os.path.join(root, file))
        
        return sorted(image_files)
    
    def _match_pairs(self):
        """Match raw images with reference images by filename"""
        self.pairs = []
        
        # Create a mapping of basenames to reference images
        ref_map = {}
        for ref_path in self.reference_images:
            basename = os.path.splitext(os.path.basename(ref_path))[0]
            ref_map[basename] = ref_path
        
        # Try to match raw images
        for raw_path in self.raw_images:
            basename = os.path.splitext(os.path.basename(raw_path))[0]
            
            # Try exact match first
            if basename in ref_map:
                self.pairs.append((raw_path, ref_map[basename]))
            # Try with common suffixes removed
            elif basename.replace('_raw', '') in ref_map:
                self.pairs.append((raw_path, ref_map[basename.replace('_raw', '')]))
            elif basename.replace('_input', '') in ref_map:
                self.pairs.append((raw_path, ref_map[basename.replace('_input', '')]))
        
        # If no pairs found, use unpaired mode
        if len(self.pairs) == 0:
            print("Warning: No matching pairs found, using unpaired mode")
            self.paired = False
    
    def __len__(self):
        if self.paired:
            return len(self.pairs)
        return len(self.raw_images)
    
    def __getitem__(self, idx):
        """
        Returns:
            If paired: (input_tensor, reference_tensor)
            If unpaired: (input_tensor, input_tensor) - reference is same as input
        """
        if self.paired:
            raw_path, ref_path = self.pairs[idx]
            raw_img = Image.open(raw_path).convert('RGB')
            ref_img = Image.open(ref_path).convert('RGB')
        else:
            raw_path = self.raw_images[idx]
            raw_img = Image.open(raw_path).convert('RGB')
            ref_img = raw_img.copy()  # Use same image as reference for unpaired
        
        # Apply augmentation if training
        if self.augment is not None:
            # Apply same augmentation to both images
            seed = random.randint(0, 2**32)
            
            random.seed(seed)
            torch.manual_seed(seed)
            raw_img = self.augment(raw_img)
            
            random.seed(seed)
            torch.manual_seed(seed)
            ref_img = self.augment(ref_img)
        
        # Convert to tensor
        raw_tensor = self.transform(raw_img)
        ref_tensor = self.transform(ref_img)
        
        return raw_tensor, ref_tensor


def create_dataloader(raw_dir, reference_dir=None, challenging_dir=None,
                      batch_size=8, size=(256, 256), mode='train', 
                      num_workers=4, shuffle=True):
    """
    Create a DataLoader for underwater images
    
    Args:
        raw_dir: Directory with raw images
        reference_dir: Directory with reference images (optional)
        challenging_dir: Directory with challenging images (optional)
        batch_size: Batch size
        size: Image size (height, width)
        mode: 'train' or 'test'
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader object
    """
    dataset = UnderwaterDataset(
        raw_dir=raw_dir,
        reference_dir=reference_dir,
        challenging_dir=challenging_dir,
        size=size,
        mode=mode
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and mode == 'train',
        num_workers=0,  # Set to 0 to avoid Windows multiprocessing issues
        pin_memory=True
    )
    
    return dataloader


if __name__ == '__main__':
    # Test the dataloader
    dataloader = create_dataloader(
        raw_dir='raw-890/raw-890',
        reference_dir='reference-890/reference-890',
        challenging_dir='challenging-60/challenging-60',
        batch_size=4,
        mode='train'
    )
    
    print(f"\nDataLoader created with {len(dataloader)} batches")
    
    # Get one batch
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx}: Input shape {inputs.shape}, Target shape {targets.shape}")
        break
