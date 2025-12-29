"""
Evaluation Script for Underwater Image Enhancement
Calculates PSNR and SSIM metrics between enhanced and reference images
"""

import os
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm
import json
from pathlib import Path

from models.dehaze_model import DehazeModel
from models.clarity_model import ClarityModel
from utils import calculate_psnr, calculate_ssim, numpy_to_tensor, tensor_to_numpy


def load_image(image_path, size=None):
    """Load and preprocess image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if size is not None:
        img = cv2.resize(img, size)
    
    return img


def evaluate_models(args):
    """Evaluate models on test dataset"""
    device = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load models
    print("Loading models...")
    dehaze_model = DehazeModel(device=device)
    dehaze_model.create_model()
    dehaze_model.load(args.dehaze_model)
    print(f"Dehaze model loaded from {args.dehaze_model}")
    
    clarity_model = ClarityModel(device=device)
    clarity_model.create_model()
    clarity_model.load(args.clarity_model)
    print(f"Clarity model loaded from {args.clarity_model}\n")
    
    # Get image files
    raw_dir = Path(args.raw_dir)
    ref_dir = Path(args.reference_dir)
    
    raw_files = sorted([f for f in raw_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
    ref_files = sorted([f for f in ref_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
    
    # Match files by name
    raw_dict = {f.stem: f for f in raw_files}
    ref_dict = {f.stem: f for f in ref_files}
    
    matched_pairs = []
    for name in raw_dict:
        if name in ref_dict:
            matched_pairs.append((raw_dict[name], ref_dict[name]))
    
    print(f"Found {len(matched_pairs)} matched image pairs\n")
    
    if len(matched_pairs) == 0:
        print("Error: No matched pairs found!")
        return
    
    # Limit to specified number
    if args.num_images > 0:
        matched_pairs = matched_pairs[:args.num_images]
        print(f"Evaluating on {len(matched_pairs)} images\n")
    
    # Metrics storage
    metrics = {
        'dehaze_only': {'psnr': [], 'ssim': []},
        'full_pipeline': {'psnr': [], 'ssim': []},
        'per_image': []
    }
    
    # Process images
    print("Processing images...")
    for raw_path, ref_path in tqdm(matched_pairs, desc="Evaluating"):
        try:
            # Load images
            raw_img = load_image(str(raw_path))
            ref_img = load_image(str(ref_path), size=(raw_img.shape[1], raw_img.shape[0]))
            
            # Convert to tensor (numpy_to_tensor already adds batch dimension)
            raw_tensor = numpy_to_tensor(raw_img).to(device)
            
            # Dehaze
            with torch.no_grad():
                dehazed_tensor = dehaze_model.infer(raw_tensor)
                enhanced_tensor = clarity_model.infer(dehazed_tensor)
            
            # Convert back to numpy
            dehazed_img = tensor_to_numpy(dehazed_tensor.squeeze(0).cpu())
            enhanced_img = tensor_to_numpy(enhanced_tensor.squeeze(0).cpu())
            
            # Calculate metrics for dehaze only
            dehaze_psnr = calculate_psnr(dehazed_img, ref_img)
            dehaze_ssim = calculate_ssim(dehazed_img, ref_img)
            
            # Calculate metrics for full pipeline
            full_psnr = calculate_psnr(enhanced_img, ref_img)
            full_ssim = calculate_ssim(enhanced_img, ref_img)
            
            # Store metrics
            metrics['dehaze_only']['psnr'].append(dehaze_psnr)
            metrics['dehaze_only']['ssim'].append(dehaze_ssim)
            metrics['full_pipeline']['psnr'].append(full_psnr)
            metrics['full_pipeline']['ssim'].append(full_ssim)
            
            metrics['per_image'].append({
                'image': raw_path.name,
                'dehaze_psnr': float(dehaze_psnr),
                'dehaze_ssim': float(dehaze_ssim),
                'full_psnr': float(full_psnr),
                'full_ssim': float(full_ssim)
            })
            
        except Exception as e:
            print(f"\nError processing {raw_path.name}: {e}")
            continue
    
    # Calculate statistics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nTotal images evaluated: {len(metrics['dehaze_only']['psnr'])}")
    
    print("\n--- Dehazing Only ---")
    print(f"PSNR: {np.mean(metrics['dehaze_only']['psnr']):.4f} ± {np.std(metrics['dehaze_only']['psnr']):.4f}")
    print(f"SSIM: {np.mean(metrics['dehaze_only']['ssim']):.4f} ± {np.std(metrics['dehaze_only']['ssim']):.4f}")
    
    print("\n--- Full Pipeline (Dehaze + Clarity) ---")
    print(f"PSNR: {np.mean(metrics['full_pipeline']['psnr']):.4f} ± {np.std(metrics['full_pipeline']['psnr']):.4f}")
    print(f"SSIM: {np.mean(metrics['full_pipeline']['ssim']):.4f} ± {np.std(metrics['full_pipeline']['ssim']):.4f}")
    
    # Show improvement
    psnr_improvement = np.mean(metrics['full_pipeline']['psnr']) - np.mean(metrics['dehaze_only']['psnr'])
    ssim_improvement = np.mean(metrics['full_pipeline']['ssim']) - np.mean(metrics['dehaze_only']['ssim'])
    
    print("\n--- Improvement from Clarity Enhancement ---")
    print(f"PSNR: {psnr_improvement:+.4f} dB")
    print(f"SSIM: {ssim_improvement:+.4f}")
    
    # Top and bottom performers
    sorted_by_psnr = sorted(metrics['per_image'], key=lambda x: x['full_psnr'], reverse=True)
    
    print("\n--- Top 5 Best Results (by PSNR) ---")
    for i, item in enumerate(sorted_by_psnr[:5], 1):
        print(f"{i}. {item['image']}: PSNR={item['full_psnr']:.2f}, SSIM={item['full_ssim']:.4f}")
    
    print("\n--- Top 5 Worst Results (by PSNR) ---")
    for i, item in enumerate(sorted_by_psnr[-5:], 1):
        print(f"{i}. {item['image']}: PSNR={item['full_psnr']:.2f}, SSIM={item['full_ssim']:.4f}")
    
    print("\n" + "="*60)
    
    # Save results to JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'summary': {
                'num_images': len(metrics['dehaze_only']['psnr']),
                'dehaze_only': {
                    'psnr_mean': float(np.mean(metrics['dehaze_only']['psnr'])),
                    'psnr_std': float(np.std(metrics['dehaze_only']['psnr'])),
                    'ssim_mean': float(np.mean(metrics['dehaze_only']['ssim'])),
                    'ssim_std': float(np.std(metrics['dehaze_only']['ssim']))
                },
                'full_pipeline': {
                    'psnr_mean': float(np.mean(metrics['full_pipeline']['psnr'])),
                    'psnr_std': float(np.std(metrics['full_pipeline']['psnr'])),
                    'ssim_mean': float(np.mean(metrics['full_pipeline']['ssim'])),
                    'ssim_std': float(np.std(metrics['full_pipeline']['ssim']))
                },
                'improvement': {
                    'psnr': float(psnr_improvement),
                    'ssim': float(ssim_improvement)
                }
            },
            'per_image': metrics['per_image']
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate underwater image enhancement models')
    parser.add_argument('--raw_dir', type=str, default='raw-890/raw-890',
                        help='Directory containing raw underwater images')
    parser.add_argument('--reference_dir', type=str, default='reference-890/reference-890',
                        help='Directory containing reference images')
    parser.add_argument('--dehaze_model', type=str, default='./saved_models/dehaze_model_final.pth',
                        help='Path to trained dehaze model')
    parser.add_argument('--clarity_model', type=str, default='./saved_models/clarity_model_final.pth',
                        help='Path to trained clarity model')
    parser.add_argument('--output', type=str, default='./evaluation_results.json',
                        help='Path to save evaluation results (JSON)')
    parser.add_argument('--num_images', type=int, default=-1,
                        help='Number of images to evaluate (-1 for all)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for inference')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_models(args)
