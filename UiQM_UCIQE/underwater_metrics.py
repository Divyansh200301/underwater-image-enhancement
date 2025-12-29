"""
Underwater-specific Image Quality Metrics
Implements UIQM (Underwater Image Quality Measure) and UCIQE (Underwater Color Image Quality Evaluation)
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import color


def uicm(img):
    """
    Underwater Image Colorfulness Measure (UICM)
    Measures the colorfulness/chrominance of underwater images
    
    Args:
        img: RGB image (H, W, 3) in range [0, 255]
    
    Returns:
        UICM score (higher is better)
    """
    img = img.astype(np.float32)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    
    RG = R - G
    YB = (R + G) / 2 - B
    
    mean_rg = np.mean(RG)
    mean_yb = np.mean(YB)
    
    std_rg = np.std(RG)
    std_yb = np.std(YB)
    
    uicm_score = -0.0268 * np.sqrt(mean_rg**2 + mean_yb**2) + 0.1586 * np.sqrt(std_rg**2 + std_yb**2)
    
    return uicm_score


def uism(img):
    """
    Underwater Image Sharpness Measure (UISM)
    Measures the sharpness of underwater images using Sobel edge detection
    
    Args:
        img: RGB image (H, W, 3) in range [0, 255]
    
    Returns:
        UISM score (higher is better)
    """
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Sobel edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Edge magnitude
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # EME (Enhancement Measure Estimation)
    k1, k2 = 8, 8  # Number of blocks
    h, w = gray.shape
    block_h, block_w = h // k1, w // k2
    
    eme = 0.0
    for i in range(k1):
        for j in range(k2):
            block = edges[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            if block.size > 0:
                max_val = np.max(block)
                min_val = np.min(block)
                if min_val > 0:
                    eme += np.log(max_val / min_val + 1e-8)
    
    uism_score = eme / (k1 * k2)
    
    return uism_score


def uiconm(img):
    """
    Underwater Image Contrast Measure (UIConM)
    Measures the contrast of underwater images using LOGAMEE
    
    Args:
        img: RGB image (H, W, 3) in range [0, 255]
    
    Returns:
        UIConM score (higher is better)
    """
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # LOGAMEE (Logarithmic AME Enhancement)
    k1, k2 = 8, 8
    h, w = gray.shape
    block_h, block_w = h // k1, w // k2
    
    logamee = 0.0
    for i in range(k1):
        for j in range(k2):
            block = gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            if block.size > 0:
                max_val = np.max(block)
                min_val = np.min(block)
                if min_val > 0:
                    logamee += np.log(max_val / min_val + 1e-8)
    
    uiconm_score = logamee / (k1 * k2)
    
    return uiconm_score


def calculate_uiqm(img):
    """
    Calculate UIQM (Underwater Image Quality Measure)
    UIQM = c1 * UICM + c2 * UISM + c3 * UIConM
    
    Args:
        img: RGB image (H, W, 3) in range [0, 255]
    
    Returns:
        UIQM score (higher is better, typical range 0-5)
    """
    # Coefficients from the original paper
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    
    uicm_score = uicm(img)
    uism_score = uism(img)
    uiconm_score = uiconm(img)
    
    uiqm_score = c1 * uicm_score + c2 * uism_score + c3 * uiconm_score
    
    return uiqm_score, {
        'uicm': uicm_score,
        'uism': uism_score,
        'uiconm': uiconm_score
    }


def calculate_uciqe(img):
    """
    Calculate UCIQE (Underwater Color Image Quality Evaluation)
    
    Args:
        img: RGB image (H, W, 3) in range [0, 255]
    
    Returns:
        UCIQE score (higher is better, typical range 0-1)
    """
    img = img.astype(np.float32) / 255.0
    
    # Convert to LAB color space
    lab = color.rgb2lab(img)
    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]
    
    # Chroma
    chroma = np.sqrt(A**2 + B**2)
    
    # Saturation
    saturation = chroma / (np.sqrt(chroma**2 + L**2) + 1e-8)
    
    # Calculate metrics
    sigma_c = np.std(chroma)  # Standard deviation of chroma
    mu_s = np.mean(saturation)  # Mean saturation
    
    # Contrast of luminance (using top-bottom difference)
    h, w = L.shape
    top_L = L[:h//2, :]
    bottom_L = L[h//2:, :]
    con_l = np.mean(top_L) - np.mean(bottom_L)
    
    # UCIQE calculation (coefficients from original paper)
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    
    uciqe_score = c1 * sigma_c + c2 * con_l + c3 * mu_s
    
    return uciqe_score, {
        'sigma_c': sigma_c,
        'mu_s': mu_s,
        'con_l': con_l
    }


if __name__ == '__main__':
    # Test with a sample image
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        uiqm_score, uiqm_components = calculate_uiqm(img)
        uciqe_score, uciqe_components = calculate_uciqe(img)
        
        print(f"Image: {img_path}")
        print(f"\nUIQM Score: {uiqm_score:.4f}")
        print(f"  - UICM (Colorfulness): {uiqm_components['uicm']:.4f}")
        print(f"  - UISM (Sharpness): {uiqm_components['uism']:.4f}")
        print(f"  - UIConM (Contrast): {uiqm_components['uiconm']:.4f}")
        
        print(f"\nUCIQE Score: {uciqe_score:.4f}")
        print(f"  - σc (Chroma Std): {uciqe_components['sigma_c']:.4f}")
        print(f"  - μs (Saturation): {uciqe_components['mu_s']:.4f}")
        print(f"  - ConL (Contrast): {uciqe_components['con_l']:.4f}")
