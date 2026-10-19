import os
import sys

# --- ROBUST IMPORT HANDLING FOR LINUX SERVERS ---
try:
    import cv2
except ImportError as e:
    print("\n" + "="*50)
    print("CRITICAL ERROR: OpenCV (cv2) is not installed correctly or missing system libraries.")
    print("Because you are on a Linux Server (DGX), you MUST install the headless version.")
    print("Please run this command exactly:")
    print("    pip uninstall -y opencv-python opencv-contrib-python")
    print("    pip install opencv-python-headless")
    print("    apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0")
    print("="*50 + "\n")
    sys.exit(1)

import gradio as gr
import numpy as np
import urllib.request
import torch

# Ensure we have the necessary weights downloaded
def download_weights():
    os.makedirs('weights', exist_ok=True)
    gfpgan_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    realesrgan_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    
    if not os.path.exists('weights/GFPGANv1.3.pth'):
        print("Downloading GFPGAN weights (Face Restoration)...")
        urllib.request.urlretrieve(gfpgan_url, 'weights/GFPGANv1.3.pth')
        
    if not os.path.exists('weights/RealESRGAN_x4plus.pth'):
        print("Downloading Real-ESRGAN weights (4K Upscaling)...")
        urllib.request.urlretrieve(realesrgan_url, 'weights/RealESRGAN_x4plus.pth')

print("Checking pre-trained AI weights...")
download_weights()

try:
    from gfpgan import GFPGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
except ImportError:
    print("Missing AI packages! Run: pip install gfpgan realesrgan basicsr facexlib")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading Models onto {device}...")

# Initialize RealESRGAN (For the background/clothes upscaling)
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus.pth',
    model=model,
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=(device.type == 'cuda'), # Use FP16 only if on GPU (DGX A100)
    device=device
)

# Initialize GFPGAN (For hallucinating missing facial features)
face_enhancer = GFPGANer(
    model_path='weights/GFPGANv1.3.pth',
    upscale=4,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=upsampler,
    device=device
)

def enhance_image(image, boost_color=True):
    if image is None:
        return None
        
    # Convert RGB (Gradio) to BGR (OpenCV/GFPGAN)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 1. AI ENHANCEMENT: GFPGAN + Real-ESRGAN
    print("Processing Image through GFPGAN and Real-ESRGAN...")
    _, _, output = face_enhancer.enhance(img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
    
    # 2. OPTIONAL: Color and Contrast Boosting (Like Cutout.pro's auto-enhance)
    if boost_color:
        print("Applying Color & Contrast Boost...")
        # Convert to HSV to boost saturation gently
        hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        # Boost saturation by 30% for that vibrant commercial look
        s = s * 1.3
        s = np.clip(s, 0, 255)
        
        hsv = cv2.merge([h, s, v])
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Contrast boosting (CLAHE)
        lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        output = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Convert BGR back to RGB for Gradio display
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    
    return output_rgb

# --- GRADIO WEB INTERFACE ---
print("Launching Web Interface...")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📸 Professional AI Image Enhancer
        Powered by GFPGAN (Face Restoration) and Real-ESRGAN (4K Upscaling).
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Original Image (Upload old/blurry photo)")
            boost_toggle = gr.Checkbox(label="Enable Cinematic Color & Contrast Boost", value=True)
            submit_btn = gr.Button("Enhance Image 🚀", variant="primary")
            
        with gr.Column():
            # Gradio Image component has a built-in interactive zoom/pan for inspection
            output_img = gr.Image(label="Restored 4K Image", interactive=False)
            
    submit_btn.click(
        fn=enhance_image,
        inputs=[input_img, boost_toggle],
        outputs=[output_img]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
