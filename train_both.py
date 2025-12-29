"""
Training script for joint dehazing and clarity enhancement
Train both models together in a pipeline
"""

import os
import argparse
import time
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from data.dataloader import create_dataloader
from models.dehaze_model import DehazeModel
from models.clarity_model import ClarityModel
from utils import AverageMeter, ensure_dir


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Both Models Together')
    
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
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training')
    
    # Training strategy
    parser.add_argument('--mode', type=str, default='sequential', 
                        choices=['sequential', 'joint'],
                        help='Training mode: sequential (dehaze->clarity) or joint (end-to-end)')
    
    # Checkpoint arguments
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                        help='Directory to save models')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save model every N epochs')
    parser.add_argument('--dehaze_checkpoint', type=str, default=None,
                        help='Pre-trained dehaze model checkpoint')
    parser.add_argument('--clarity_checkpoint', type=str, default=None,
                        help='Pre-trained clarity model checkpoint')
    
    # Logging arguments
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Log frequency (batches)')
    parser.add_argument('--sample_dir', type=str, default='./output/train_samples_both',
                        help='Directory to save sample outputs')
    
    return parser.parse_args()


def train_epoch_sequential(dehaze_model, clarity_model, dataloader, epoch, args):
    """
    Train both models sequentially:
    1. Train dehaze model
    2. Apply dehazing to get intermediate results
    3. Train clarity model on dehazed images
    """
    dehaze_model.model.train()
    clarity_model.model.train()
    
    dehaze_losses = AverageMeter()
    clarity_losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}')
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        # Step 1: Train dehazing model
        dehaze_loss = dehaze_model.train_step(inputs, targets)
        dehaze_losses.update(dehaze_loss, inputs.size(0))
        
        # Step 2: Get dehazed outputs (detached to avoid backprop through dehaze)
        with torch.no_grad():
            dehazed = dehaze_model.infer(inputs)
        
        # Step 3: Train clarity model on dehazed images
        clarity_loss = clarity_model.train_step(dehazed, targets)
        clarity_losses.update(clarity_loss, inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'dehaze_loss': f'{dehaze_losses.avg:.4f}',
            'clarity_loss': f'{clarity_losses.avg:.4f}'
        })
        
        # Log periodically
        if batch_idx % args.log_freq == 0:
            print(f'Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(dataloader)}] '
                  f'Dehaze Loss: {dehaze_losses.avg:.4f} Clarity Loss: {clarity_losses.avg:.4f}')
    
    return dehaze_losses.avg, clarity_losses.avg


def train_epoch_joint(dehaze_model, clarity_model, dataloader, epoch, args):
    """
    Train both models jointly end-to-end
    Backpropagate through the entire pipeline
    """
    dehaze_model.model.train()
    clarity_model.model.train()
    
    total_losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}')
    
    # Combine optimizers
    combined_params = list(dehaze_model.model.parameters()) + list(clarity_model.model.parameters())
    combined_optimizer = torch.optim.Adam(combined_params, lr=args.learning_rate)
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(dehaze_model.device)
        targets = targets.to(dehaze_model.device)
        
        # Forward pass through both models
        dehazed = dehaze_model.model(inputs)
        enhanced = clarity_model.model(dehazed)
        
        # Calculate combined loss
        dehaze_loss = dehaze_model.criterion(dehazed, targets)
        clarity_loss = clarity_model.criterion(enhanced, targets)
        total_loss = dehaze_loss + clarity_loss
        
        # Backward pass through entire pipeline
        combined_optimizer.zero_grad()
        total_loss.backward()
        combined_optimizer.step()
        
        total_losses.update(total_loss.item(), inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({'total_loss': f'{total_losses.avg:.4f}'})
        
        # Log periodically
        if batch_idx % args.log_freq == 0:
            print(f'Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(dataloader)}] '
                  f'Total Loss: {total_losses.avg:.4f}')
    
    return total_losses.avg


def save_sample_outputs(dehaze_model, clarity_model, dataloader, epoch, save_dir):
    """Save sample outputs showing the complete pipeline"""
    ensure_dir(save_dir)
    
    dehaze_model.model.eval()
    clarity_model.model.eval()
    
    # Get one batch
    inputs, targets = next(iter(dataloader))
    
    # Run inference through pipeline
    with torch.no_grad():
        dehazed = dehaze_model.infer(inputs)
        enhanced = clarity_model.infer(dehazed)
    
    # Save first 4 images
    num_samples = min(4, inputs.size(0))
    
    for i in range(num_samples):
        # Create comparison figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Input
        input_img = inputs[i].cpu().numpy().transpose(1, 2, 0)
        axes[0].imshow(input_img)
        axes[0].set_title('Input')
        axes[0].axis('off')
        
        # Dehazed
        dehazed_img = dehazed[i].cpu().numpy().transpose(1, 2, 0)
        axes[1].imshow(dehazed_img)
        axes[1].set_title('Dehazed')
        axes[1].axis('off')
        
        # Enhanced
        enhanced_img = enhanced[i].cpu().numpy().transpose(1, 2, 0)
        axes[2].imshow(enhanced_img)
        axes[2].set_title('Enhanced')
        axes[2].axis('off')
        
        # Target
        target_img = targets[i].cpu().numpy().transpose(1, 2, 0)
        axes[3].imshow(target_img)
        axes[3].set_title('Target')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_sample_{i}.png'))
        plt.close()
    
    print(f"Sample outputs saved to {save_dir}")


def plot_loss_curves(dehaze_losses, clarity_losses, save_path):
    """Plot and save training loss curves"""
    plt.figure(figsize=(12, 6))
    
    epochs = range(1, len(dehaze_losses) + 1)
    plt.plot(epochs, dehaze_losses, label='Dehaze Loss', marker='o')
    plt.plot(epochs, clarity_losses, label='Clarity Loss', marker='s')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Joint Training Loss Curves')
    plt.legend()
    plt.grid(True)
    
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curves saved to {save_path}")


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
    
    # Create models
    print("\nCreating models...")
    dehaze_model = DehazeModel(device=device)
    dehaze_model.create_model()
    dehaze_model.setup_training(learning_rate=args.learning_rate)
    
    clarity_model = ClarityModel(device=device)
    clarity_model.create_model()
    clarity_model.setup_training(learning_rate=args.learning_rate)
    
    # Load pre-trained checkpoints if provided
    start_epoch = 1
    if args.dehaze_checkpoint and os.path.exists(args.dehaze_checkpoint):
        print(f"Loading pre-trained dehaze model: {args.dehaze_checkpoint}")
        dehaze_model.load(args.dehaze_checkpoint)
        # Extract epoch number from checkpoint filename
        try:
            epoch_str = args.dehaze_checkpoint.split('_epoch_')[-1].split('.')[0]
            start_epoch = int(epoch_str) + 1
            print(f"Resuming from epoch {start_epoch}")
        except:
            print("Could not extract epoch number, starting from epoch 1")
    
    if args.clarity_checkpoint and os.path.exists(args.clarity_checkpoint):
        print(f"Loading pre-trained clarity model: {args.clarity_checkpoint}")
        clarity_model.load(args.clarity_checkpoint)
    
    # Training loop
    print(f"\nStarting {args.mode} training for {args.epochs} epochs...")
    print(f"Starting from epoch: {start_epoch}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Models will be saved to: {args.save_dir}\n")
    
    dehaze_loss_history = []
    clarity_loss_history = []
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        
        # Train one epoch based on mode
        if args.mode == 'sequential':
            dehaze_loss, clarity_loss = train_epoch_sequential(
                dehaze_model, clarity_model, dataloader, epoch, args
            )
            dehaze_loss_history.append(dehaze_loss)
            clarity_loss_history.append(clarity_loss)
            
            print(f"Epoch {epoch} - Dehaze Loss: {dehaze_loss:.4f}, Clarity Loss: {clarity_loss:.4f}")
        
        else:  # joint
            total_loss = train_epoch_joint(
                dehaze_model, clarity_model, dataloader, epoch, args
            )
            dehaze_loss_history.append(total_loss)
            clarity_loss_history.append(total_loss)
            
            print(f"Epoch {epoch} - Total Loss: {total_loss:.4f}")
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s\n")
        
        # Save sample outputs
        if epoch % args.save_freq == 0 or epoch == 1:
            save_sample_outputs(dehaze_model, clarity_model, dataloader, epoch, args.sample_dir)
        
        # Save model checkpoints
        if epoch % args.save_freq == 0:
            dehaze_path = os.path.join(args.save_dir, f'dehaze_model_joint_epoch_{epoch}.pth')
            clarity_path = os.path.join(args.save_dir, f'clarity_model_joint_epoch_{epoch}.pth')
            
            dehaze_model.save(dehaze_path)
            clarity_model.save(clarity_path)
    
    # Save final models
    dehaze_final = os.path.join(args.save_dir, 'dehaze_model_final.pth')
    clarity_final = os.path.join(args.save_dir, 'clarity_model_final.pth')
    
    dehaze_model.save(dehaze_final)
    clarity_model.save(clarity_final)
    
    # Plot and save loss curves
    plot_loss_curves(
        dehaze_loss_history, 
        clarity_loss_history,
        os.path.join(args.save_dir, 'joint_training_loss_curves.png')
    )
    
    print("\n" + "="*50)
    print("Joint training completed!")
    print(f"Dehaze model saved to: {dehaze_final}")
    print(f"Clarity model saved to: {clarity_final}")
    print("="*50)



if __name__ == '__main__':
    main()
