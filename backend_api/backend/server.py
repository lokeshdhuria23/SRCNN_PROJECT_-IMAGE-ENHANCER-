import os
import sys
import io
import cv2
import numpy as np
import urllib.request
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI(title="Cutout Clone AI Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_enhancer = None
colornet = None

def download_file(url, path):
    import requests
    response = requests.get(url, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
    response.raise_for_status()
    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def download_weights():
    os.makedirs('weights', exist_ok=True)
    gfpgan_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    realesrgan_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    color_model = 'https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel'
    color_prototxt = 'https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt'
    pts_in_hull = 'https://github.com/richzhang/colorization/raw/caffe/colorization/resources/pts_in_hull.npy'
    
    if not os.path.exists('weights/GFPGANv1.3.pth'):
        print("Downloading GFPGAN weights...")
        download_file(gfpgan_url, 'weights/GFPGANv1.3.pth')
    if not os.path.exists('weights/RealESRGAN_x4plus.pth'):
        print("Downloading Real-ESRGAN weights...")
        download_file(realesrgan_url, 'weights/RealESRGAN_x4plus.pth')
    if not os.path.exists('weights/colorization_release_v2.caffemodel'):
        print("Downloading Colorization Caffemodel... (~120MB)")
        download_file(color_model, 'weights/colorization_release_v2.caffemodel')
    if not os.path.exists('weights/colorization_deploy_v2.prototxt'):
        print("Downloading Colorization Prototxt...")
        download_file(color_prototxt, 'weights/colorization_deploy_v2.prototxt')
    if not os.path.exists('weights/pts_in_hull.npy'):
        print("Downloading pts_in_hull...")
        download_file(pts_in_hull, 'weights/pts_in_hull.npy')

def initialize_models():
    global face_enhancer, colornet
    if face_enhancer is not None:
        return
        
    print("Checking weights...")
    download_weights()
    
    # Initialize Colorizer
    print("Loading Colorization Model...")
    colornet = cv2.dnn.readNetFromCaffe('weights/colorization_deploy_v2.prototxt', 'weights/colorization_release_v2.caffemodel')
    pts = np.load('weights/pts_in_hull.npy')
    class8 = colornet.getLayerId("class8_ab")
    conv8 = colornet.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    colornet.getLayer(class8).blobs = [pts.astype("float32")]
    colornet.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    try:
        from gfpgan import GFPGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        print("CRITICAL ERROR: Missing AI libraries.")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Models onto {device}...")

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path='weights/RealESRGAN_x4plus.pth', model=model, tile=400, tile_pad=10, pre_pad=0, half=(device.type == 'cuda'), device=device)
    face_enhancer = GFPGANer(model_path='weights/GFPGANv1.3.pth', upscale=4, arch='clean', channel_multiplier=2, bg_upsampler=upsampler, device=device)
    print("Models successfully loaded!")

def colorize_image(img_bgr):
    """Colorizes a B&W image using Zhang et al. DNN."""
    scaled = img_bgr.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    
    # Resize to 224x224, the expected network input size
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    
    colornet.setInput(cv2.dnn.blobFromImage(L))
    ab = colornet.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img_bgr.shape[1], img_bgr.shape[0]))
    
    L_original = cv2.split(lab)[0]
    colorized_lab = np.concatenate((L_original[:, :, np.newaxis], ab), axis=2)
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    colorized_bgr = np.clip(colorized_bgr, 0, 1)
    
    return (colorized_bgr * 255).astype("uint8")

def color_boost(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = s * 1.3
    s = np.clip(s, 0, 255)
    hsv = cv2.merge([h, s, v])
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    img_boosted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    lab = cv2.cvtColor(img_boosted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def enhance_edges(img, intensity: float = 1.0):
    if intensity <= 0.01:
        return img
    gaussian_blur = cv2.GaussianBlur(img, (0, 0), 3.0)
    unsharp_image = cv2.addWeighted(img, 1.0 + (1.5 * intensity), gaussian_blur, -(1.5 * intensity), 0)
    kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(unsharp_image, -1, kernel)
    return cv2.addWeighted(unsharp_image, 1.0 - (0.4 * intensity), sharpened, 0.4 * intensity, 0)

@app.on_event("startup")
async def startup_event():
    initialize_models()

@app.post("/api/enhance")
async def enhance_image(
    image: UploadFile = File(...),
    boost_color: bool = Form(True),
    enhance_faces: bool = Form(True),
    sharpness_level: float = Form(0.0),
    colorize: bool = Form(False)
):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return Response(content="Invalid image file", status_code=400)
    
    # 1. Colorize B&W first if requested
    if colorize:
        img_bgr = colorize_image(img_bgr)
    
    # 2. AI Upscaling
    if enhance_faces:
        _, _, output = face_enhancer.enhance(img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
    else:
        output, _ = face_enhancer.bg_upsampler.enhance(img_bgr, outscale=4)
    
    # 3. Post-Processing
    if sharpness_level > 0:
        output = enhance_edges(output, intensity=sharpness_level)
    if boost_color:
        output = color_boost(output)
        
    is_success, buffer = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not is_success:
        return Response(content="Image encoding failed", status_code=500)
        
    io_buf = io.BytesIO(buffer)
    return Response(content=io_buf.getvalue(), media_type="image/jpeg")