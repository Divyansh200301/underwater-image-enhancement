# 🌊 Underwater Image Dehazing & Clarity Enhancement System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.1](https://img.shields.io/badge/pytorch-2.5.1-red.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/cuda-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Gradio](https://img.shields.io/badge/demo-gradio-orange.svg)](https://gradio.app/)

A complete PyTorch-based system for enhancing underwater images through dehazing and clarity enhancement. Features both command-line tools and an interactive Gradio UI.

## 📋 Features

- **Dual-Model Architecture**:
  - U-Net based dehazing model to remove haze and color distortion
  - Residual network with attention mechanism for clarity enhancement
  
- **Flexible Training**:
  - Train models individually or jointly
  - Support for paired and unpaired training data
  - Automatic checkpoint saving and resume capability
  
- **Multiple Interfaces**:
  - Command-line inference script for batch processing
  - Interactive Gradio web UI for single image enhancement
  
- **Comprehensive Utilities**:
  - Image quality metrics (PSNR, SSIM)
  - Training visualization and loss curves
  - Sample output generation during training

## 🗂️ Project Structure

```
.
├── data/
│   ├── __init__.py
│   └── dataloader.py          # Dataset loading and preprocessing
├── models/
│   ├── __init__.py
│   ├── dehaze_model.py        # U-Net dehazing model
│   └── clarity_model.py       # Clarity enhancement model
├── dataset/                   # Your dataset folders (need to be created)
│   ├── raw/                   # Raw underwater images
│   ├── challenging/           # Challenging underwater images
│   └── reference/             # Reference/ground truth images (optional)
├── saved_models/              # Trained model checkpoints
├── output/                    # Inference results and samples
├── train_dehaze.py            # Train dehazing model
├── train_clarity.py           # Train clarity model
├── train_both.py              # Train both models together
├── infer.py                   # Command-line inference
├── app.py                     # Gradio web UI
├── utils.py                   # Utility functions
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd "Underwater Image Dehazing and Clarity Enhancement System"

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

**⚠️ Note**: Datasets are not included in this repository due to size constraints.

#### Option 1: Use Your Own Dataset

Create dataset directory structure:

```bash
mkdir -p dataset/raw dataset/challenging dataset/reference
```

Place your images in the appropriate folders:
- `dataset/raw/` - Raw underwater images for training
- `dataset/challenging/` - Challenging underwater images (optional)
- `dataset/reference/` - Reference/ground truth images (optional, for paired training)

#### Option 2: Download Public Datasets

You can use public underwater image datasets:
- **UIEB Dataset**: [Underwater Image Enhancement Benchmark](https://li-chongyi.github.io/proj_benchmark.html)
- **EUVP Dataset**: [Enhancement of Underwater Visual Perception](http://irvlab.cs.umn.edu/resources/euvp-dataset)
- **SUIM Dataset**: [Semantic Underwater Image Dataset](http://irvlab.cs.umn.edu/resources/suim-dataset)

**Note**: If you don't have reference images, the system will use unpaired training mode.

### 3. Training

#### Option A: Train Dehazing Model Only

```bash
python train_dehaze.py --raw_dir raw-890/raw-890 --reference_dir reference-890/reference-890 --epochs 50 --batch_size 8
```

#### Option B: Train Clarity Model Only

```bash
python train_clarity.py --raw_dir raw-890/raw-890 --reference_dir reference-890/reference-890 --epochs 50 --batch_size 8
```

#### Option C: Train Both Models Together

```bash
# Sequential training (recommended)
python train_both.py --mode sequential --epochs 50 --batch_size 8

# Joint end-to-end training
python train_both.py --mode joint --epochs 50 --batch_size 8
```

**Training Arguments**:
- `--raw_dir`: Directory with raw images (default: `raw-890/raw-890`)
- `--reference_dir`: Directory with reference images (default: `reference-890/reference-890`)
- `--challenging_dir`: Directory with challenging images (default: `challenging-60/challenging-60`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 8)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--image_size`: Image size for training (default: 256)
- `--save_dir`: Directory to save models (default: `./saved_models`)
- `--save_freq`: Save model every N epochs (default: 5)
- `--resume`: Path to checkpoint to resume from

### 4. Inference

#### Command-Line Inference

Process a single image:
```bash
python infer.py --input path/to/image.jpg --output ./output
```

Process an entire folder:
```bash
python infer.py --input path/to/folder/ --output ./output --save_intermediate
```

**Inference Arguments**:
- `--input`: Input image file or directory (required)
- `--output`: Output directory (default: `./output`)
- `--dehaze_model`: Path to dehaze model (default: `./saved_models/dehaze_model_final.pth`)
- `--clarity_model`: Path to clarity model (default: `./saved_models/clarity_model_final.pth`)
- `--merge_alpha`: Alpha for merging results, 0-1 (default: 0.5)
- `--save_intermediate`: Save intermediate results (dehazed, clarity)
- `--device`: Device to use (auto/cuda/cpu, default: auto)

#### Interactive Web UI

```bash
python app.py
```

Then open your browser to `http://localhost:7860`

The UI allows you to:
- Upload underwater images via drag & drop
- Adjust merge weight between dehazing and clarity enhancement
- View all processing stages (original, dehazed, clarity enhanced, merged)
- Download results

## 📊 Model Architecture

### Dehazing Model (U-Net)
- Encoder-decoder architecture with skip connections
- 4 downsampling and 4 upsampling blocks
- Base channels: 64 (configurable)
- Loss: MSE loss
- Parameters: ~31M (with base_channels=64)

### Clarity Enhancement Model
- Residual blocks with channel attention
- Dual-branch architecture (sharpness + contrast)
- Loss: L1 loss + edge-aware loss
- Parameters: ~8M (with n_features=64)

## 🎯 Performance Tips

1. **GPU Usage**: For faster training, use a CUDA-enabled GPU
   ```bash
   # Check if CUDA is available
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Batch Size**: Adjust based on your GPU memory
   - 16GB VRAM: batch_size=16-32
   - 8GB VRAM: batch_size=8-16
   - CPU: batch_size=4-8

3. **Training Time**: 
   - ~50 epochs takes 2-4 hours on modern GPU
   - Monitor loss curves in `./saved_models/`

4. **Best Practices**:
   - Use paired data if available for better results
   - Train for at least 30-50 epochs
   - Monitor sample outputs in `./output/train_samples/`
   - Adjust merge_alpha in inference to balance dehazing vs clarity

## 📈 Monitoring Training

During training, the system:
- Prints loss every N batches
- Saves sample outputs every N epochs to `./output/train_samples/`
- Saves model checkpoints every N epochs to `./saved_models/`
- Generates loss curves as PNG files

View training progress:
```bash
# Sample outputs
ls output/train_samples/

# Loss curves
ls saved_models/*.png

# Model checkpoints
ls saved_models/*.pth
```

## 🔧 Troubleshooting

**Issue**: Out of memory error
- Solution: Reduce `--batch_size` or `--image_size`

**Issue**: Models not found when running app.py
- Solution: Train models first or check paths in `./saved_models/`

**Issue**: Poor results
- Solution: Train for more epochs or check if paired data is being used

**Issue**: Slow inference
- Solution: Use GPU (`--device cuda`) or reduce image size

## 📝 Metrics

The system includes utilities for calculating:
- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better
- **SSIM** (Structural Similarity Index): Closer to 1 is better

Use these in `utils.py`:
```python
from utils import calculate_psnr, calculate_ssim

psnr_value = calculate_psnr(image1, image2)
ssim_value = calculate_ssim(image1, image2)
```

## 🤝 Contributing

Feel free to:
- Report bugs or issues
- Suggest improvements
- Add new features
- Improve documentation

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- U-Net architecture inspired by Ronneberger et al.
- Attention mechanisms based on modern CNN architectures
- Underwater image datasets from the research community

## 📧 Contact

For questions or issues, please create an issue in the project repository.

---

**Happy Enhancing! 🌊📸**
