"""
Gradio UI for Underwater Image Enhancement
Provides interactive interface for image dehazing and clarity enhancement
"""

import os
import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image

from models.dehaze_model import DehazeModel
from models.clarity_model import ClarityModel
from utils import numpy_to_tensor, tensor_to_numpy


# Global model instances
dehaze_model = None
clarity_model = None
device = None


def load_models():
    """Load both models at startup"""
    global dehaze_model, clarity_model, device
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading models on {device}...")
    
    # Load dehaze model
    dehaze_model = DehazeModel(device=device)
    dehaze_model.create_model()
    
    dehaze_path = './saved_models/dehaze_model_final.pth'
    if os.path.exists(dehaze_path):
        dehaze_model.load(dehaze_path)
        print("Dehaze model loaded successfully")
    else:
        print(f"Warning: Dehaze model not found at {dehaze_path}")
        print("Using untrained model - results may be poor")
    
    # Load clarity model
    clarity_model = ClarityModel(device=device)
    clarity_model.create_model()
    
    clarity_path = './saved_models/clarity_model_final.pth'
    if os.path.exists(clarity_path):
        clarity_model.load(clarity_path)
        print("Clarity model loaded successfully")
    else:
        print(f"Warning: Clarity model not found at {clarity_path}")
        print("Using untrained model - results may be poor")
    
    print("Models loaded and ready!")


def process_image(image, merge_alpha=0.5):
    """
    Process uploaded image through enhancement pipeline
    
    Args:
        image: Input image (numpy array from Gradio)
        merge_alpha: Blending weight for final merged result
        
    Returns:
        original, dehazed, clarity_enhanced, merged
    """
    global dehaze_model, clarity_model, device
    
    if image is None:
        return None, None, None, None
    
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Store original
    original = image.copy()
    
    # Convert to tensor
    img_tensor = numpy_to_tensor(image).to(device)
    
    # Run through pipeline
    with torch.no_grad():
        # Dehazing
        dehazed_tensor = dehaze_model.infer(img_tensor)
        
        # Clarity enhancement
        clarity_tensor = clarity_model.infer(dehazed_tensor)
        
        # Merge results
        merged_tensor = merge_alpha * dehazed_tensor + (1 - merge_alpha) * clarity_tensor
        merged_tensor = torch.clamp(merged_tensor, 0, 1)
    
    # Convert back to numpy arrays
    dehazed = tensor_to_numpy(dehazed_tensor)
    clarity_enhanced = tensor_to_numpy(clarity_tensor)
    merged = tensor_to_numpy(merged_tensor)
    
    return original, dehazed, clarity_enhanced, merged


def create_gradio_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Underwater Image Enhancement") as demo:
        gr.Markdown(
            """
            # Underwater Image Dehazing & Clarity Enhancement
            
            Upload an underwater image to enhance it using deep learning models.
            The system applies dehazing and clarity enhancement to produce clearer, more vibrant images.
            
            **Pipeline:**
            1. **Dehazing**: Remove haze and color distortion
            2. **Clarity Enhancement**: Improve sharpness and contrast
            3. **Merged**: Combine both enhancements for optimal results
            """
        )
        
        with gr.Row():
            with gr.Column():
                # Input
                input_image = gr.Image(
                    label="Upload Underwater Image",
                    type="numpy",
                    height=400
                )
                
                # Controls
                merge_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label="Merge Weight (0=Clarity Only, 1=Dehaze Only)",
                    info="Adjust the balance between dehazing and clarity enhancement"
                )
                
                # Buttons
                with gr.Row():
                    process_btn = gr.Button("🚀 Enhance Image", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear")
        
        # Outputs
        gr.Markdown("### Results")
        
        with gr.Row():
            output_original = gr.Image(label="Original", height=300)
            output_dehazed = gr.Image(label="Dehazed", height=300)
        
        with gr.Row():
            output_clarity = gr.Image(label="Clarity Enhanced", height=300)
            output_merged = gr.Image(label="Final Merged Result", height=300)
        
        # Examples
        gr.Markdown("### 📸 Example Images")
        gr.Markdown(
            "If you don't have underwater images handy, try uploading one of your own or use the examples below if provided."
        )
        
        # Event handlers
        def process_wrapper(image, alpha):
            """Wrapper to handle processing and return all outputs"""
            original, dehazed, clarity, merged = process_image(image, alpha)
            return original, dehazed, clarity, merged
        
        def clear_all():
            """Clear all inputs and outputs"""
            return None, None, None, None, None
        
        process_btn.click(
            fn=process_wrapper,
            inputs=[input_image, merge_slider],
            outputs=[output_original, output_dehazed, output_clarity, output_merged]
        )
        
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[input_image, output_original, output_dehazed, output_clarity, output_merged]
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            ### 📝 Notes:
            - Adjust the merge weight slider to find the optimal balance for your image
            - Processing time depends on your hardware (GPU recommended)
            
            ### 🔧 Model Information:
            - **Dehazing Model**: U-Net based architecture
            - **Clarity Model**: Residual network with attention mechanism
            - **Device**: """ + device + """
            """
        )
    
    return demo


def main():
    """Main function to launch Gradio app"""
    print("="*60)
    print("Underwater Image Enhancement System")
    print("="*60)
    
    # Load models
    load_models()
    
    print("\n" + "="*60)
    print("Starting Gradio interface...")
    print("="*60 + "\n")
    
    # Create and launch interface
    demo = create_gradio_interface()
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create public link
        show_error=True
    )


if __name__ == '__main__':
    main()
