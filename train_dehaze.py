"""
Training script for dehazing model
"""

import os
import argparse
import time
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from data.dataloader import create_dataloader
from models.dehaze_model import DehazeModel
from utils import AverageMeter, ensure_dir, save_image


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Dehazing Model')
    
    # Data arguments
    parser.add_argument('--raw_dir', type=str, default='raw-890/raw-890',
                        help='Directory with raw images')
    parser.add_argument('--reference_dir', type=str, default='reference-890/reference-890',
                        help='Directory with reference images')
    parser.add_argument('--challenging_dir', type=str, default='challenging-60/challenging-60',
                        help='Directory with challenging images')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base number of channels in U-Net')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training')
    
    # Checkpoint arguments
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                        help='Directory to save models')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save model every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Logging arguments
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Log frequency (batches)')
    parser.add_argument('--sample_dir', type=str, default='./output/train_samples',
                        help='Directory to save sample outputs')
    
    return parser.parse_args()


def train_epoch(model, dataloader, epoch, args):
    """Train for one epoch"""
    model.model.train()
    
    losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}')
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        # Training step
        loss = model.train_step(inputs, targets)
        losses.update(loss, inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
        
        # Log periodically
        if batch_idx % args.log_freq == 0:
            print(f'Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(dataloader)}] '
                  f'Loss: {losses.avg:.4f}')
    
    return losses.avg


def save_sample_outputs(model, dataloader, epoch, save_dir):
    """Save sample outputs for visualization"""
    ensure_dir(save_dir)
    
    model.model.eval()
    
    # Get one batch
    inputs, targets = next(iter(dataloader))
    
    # Run inference
    with torch.no_grad():
        outputs = model.infer(inputs)
    
    # Save first 4 images
    num_samples = min(4, inputs.size(0))
    
    for i in range(num_samples):
        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Input
        input_img = inputs[i].cpu().numpy().transpose(1, 2, 0)
        axes[0].imshow(input_img)
        axes[0].set_title('Input')
        axes[0].axis('off')
        
        # Target
        target_img = targets[i].cpu().numpy().transpose(1, 2, 0)
        axes[1].imshow(target_img)
        axes[1].set_title('Target')
        axes[1].axis('off')
        
        # Output
        output_img = outputs[i].cpu().numpy().transpose(1, 2, 0)
        axes[2].imshow(output_img)
        axes[2].set_title('Dehazed')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_sample_{i}.png'))
        plt.close()
    
    print(f"Sample outputs saved to {save_dir}")


def plot_loss_curve(losses, save_path):
    """Plot and save training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Dehazing Model Training Loss')
    plt.legend()
    plt.grid(True)
    
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def main():
    args = parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loader
    print("\nLoading data...")
    dataloader = create_dataloader(
        raw_dir=args.raw_dir,
        reference_dir=args.reference_dir,
        challenging_dir=args.challenging_dir,
        batch_size=args.batch_size,
        size=(args.image_size, args.image_size),
        mode='train',
        num_workers=args.num_workers,
        shuffle=True
    )
    
    # Create model
    print("\nCreating model...")
    model = DehazeModel(device=device)
    model.create_model(base_channels=args.base_channels)
    model.setup_training(learning_rate=args.learning_rate)
    
    # Resume from checkpoint if provided
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        model.load(args.resume)
        # Extract epoch from filename if possible
        try:
            start_epoch = int(args.resume.split('_epoch_')[-1].split('.')[0]) + 1
        except:
            start_epoch = 1
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Model will be saved to: {args.save_dir}\n")
    
    loss_history = []
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        
        # Train one epoch
        avg_loss = train_epoch(model, dataloader, epoch, args)
        loss_history.append(avg_loss)
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s - Loss: {avg_loss:.4f}\n")
        
        # Save sample outputs
        if epoch % args.save_freq == 0 or epoch == 1:
            save_sample_outputs(model, dataloader, epoch, args.sample_dir)
        
        # Save model checkpoint
        if epoch % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f'dehaze_model_epoch_{epoch}.pth')
            model.save(save_path)
    
    # Save final model
    final_path = os.path.join(args.save_dir, 'dehaze_model_final.pth')
    model.save(final_path)
    
    # Plot and save loss curve
    plot_loss_curve(loss_history, os.path.join(args.save_dir, 'dehaze_loss_curve.png'))
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Final model saved to: {final_path}")
    print("="*50)


if __name__ == '__main__':
    main()
