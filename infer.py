"""
Inference script for underwater image enhancement
Processes single images or entire folders through the pipeline
"""

import os
import argparse
import time
from PIL import Image
import torch
import cv2
import numpy as np

from models.dehaze_model import DehazeModel
from models.clarity_model import ClarityModel
from utils import ensure_dir, get_image_files, numpy_to_tensor, tensor_to_numpy


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Inference for Underwater Image Enhancement')
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                        help='Input image file or directory')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory')
    
    # Model paths
    parser.add_argument('--dehaze_model', type=str, 
                        default='./saved_models/dehaze_model_final.pth',
                        help='Path to dehaze model')
    parser.add_argument('--clarity_model', type=str,
                        default='./saved_models/clarity_model_final.pth',
                        help='Path to clarity model')
    
    # Processing options
    parser.add_argument('--merge_alpha', type=float, default=0.5,
                        help='Alpha for merging dehazed and clarity enhanced (0-1)')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Save intermediate results (dehazed, clarity enhanced)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for inference')
    
    return parser.parse_args()


def load_models(dehaze_path, clarity_path, device):
    """
    Load both models
    
    Args:
        dehaze_path: Path to dehaze model
        clarity_path: Path to clarity model
        device: Device to use
        
    Returns:
        dehaze_model, clarity_model
    """
    print("Loading models...")
    
    # Load dehaze model
    dehaze_model = DehazeModel(device=device)
    dehaze_model.create_model()
    if os.path.exists(dehaze_path):
        dehaze_model.load(dehaze_path)
    else:
        print(f"Warning: Dehaze model not found at {dehaze_path}")
        print("Using untrained model")
    
    # Load clarity model
    clarity_model = ClarityModel(device=device)
    clarity_model.create_model()
    if os.path.exists(clarity_path):
        clarity_model.load(clarity_path)
    else:
        print(f"Warning: Clarity model not found at {clarity_path}")
        print("Using untrained model")
    
    return dehaze_model, clarity_model


def process_image(image_path, dehaze_model, clarity_model, args):
    """
    Process a single image through the enhancement pipeline
    
    Args:
        image_path: Path to input image
        dehaze_model: Dehaze model
        clarity_model: Clarity model
        args: Command line arguments
        
    Returns:
        original, dehazed, clarity_enhanced, merged
    """
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    img_tensor = numpy_to_tensor(img).to(dehaze_model.device)
    
    # Run through pipeline
    with torch.no_grad():
        # Dehazing
        dehazed = dehaze_model.infer(img_tensor)
        
        # Clarity enhancement
        clarity_enhanced = clarity_model.infer(dehazed)
        
        # Merge results
        alpha = args.merge_alpha
        merged = alpha * dehazed + (1 - alpha) * clarity_enhanced
        merged = torch.clamp(merged, 0, 1)
    
    # Convert back to numpy
    original = img
    dehazed_np = tensor_to_numpy(dehazed)
    clarity_np = tensor_to_numpy(clarity_enhanced)
    merged_np = tensor_to_numpy(merged)
    
    return original, dehazed_np, clarity_np, merged_np


def save_results(image_name, original, dehazed, clarity_enhanced, merged, 
                output_dir, save_intermediate=False):
    """
    Save all results
    
    Args:
        image_name: Name of the image
        original: Original image
        dehazed: Dehazed image
        clarity_enhanced: Clarity enhanced image
        merged: Merged final result
        output_dir: Output directory
        save_intermediate: Whether to save intermediate results
    """
    ensure_dir(output_dir)
    
    base_name = os.path.splitext(image_name)[0]
    
    # Save merged result (always)
    merged_bgr = cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)
    merged_path = os.path.join(output_dir, f'{base_name}_enhanced.png')
    cv2.imwrite(merged_path, merged_bgr)
    
    # Save intermediate results if requested
    if save_intermediate:
        # Original
        original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        original_path = os.path.join(output_dir, f'{base_name}_original.png')
        cv2.imwrite(original_path, original_bgr)
        
        # Dehazed
        dehazed_bgr = cv2.cvtColor(dehazed, cv2.COLOR_RGB2BGR)
        dehazed_path = os.path.join(output_dir, f'{base_name}_dehazed.png')
        cv2.imwrite(dehazed_path, dehazed_bgr)
        
        # Clarity enhanced
        clarity_bgr = cv2.cvtColor(clarity_enhanced, cv2.COLOR_RGB2BGR)
        clarity_path = os.path.join(output_dir, f'{base_name}_clarity.png')
        cv2.imwrite(clarity_path, clarity_bgr)


def main():
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load models
    dehaze_model, clarity_model = load_models(
        args.dehaze_model,
        args.clarity_model,
        device
    )
    
    # Get input files
    if os.path.isfile(args.input):
        image_files = [args.input]
    elif os.path.isdir(args.input):
        image_files = get_image_files(args.input)
    else:
        print(f"Error: Input path {args.input} does not exist")
        return
    
    if len(image_files) == 0:
        print("No images found to process")
        return
    
    print(f"\nProcessing {len(image_files)} images...")
    print(f"Output directory: {args.output}")
    print(f"Save intermediate results: {args.save_intermediate}\n")
    
    # Process each image
    total_time = 0
    
    for idx, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        print(f"[{idx}/{len(image_files)}] Processing {image_name}...", end=' ')
        
        start_time = time.time()
        
        try:
            # Process image
            original, dehazed, clarity_enhanced, merged = process_image(
                image_path, dehaze_model, clarity_model, args
            )
            
            # Save results
            save_results(
                image_name, original, dehazed, clarity_enhanced, merged,
                args.output, args.save_intermediate
            )
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            print(f"Done ({elapsed:.2f}s)")
        
        except Exception as e:
            print(f"Error: {e}")
    
    # Summary
    print("\n" + "="*50)
    print(f"Processing completed!")
    print(f"Total images: {len(image_files)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per image: {total_time/len(image_files):.2f}s")
    print(f"Results saved to: {args.output}")
    print("="*50)


if __name__ == '__main__':
    main()
