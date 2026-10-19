# SRCNN & AI Image Enhancer 🚀

A comprehensive, full-stack AI Image Restoration and Super-Resolution suite powered by Deep Learning. Enhance low-resolution images, restore degraded faces, upscale textures to 4K, and colorize old black-and-white photos with high precision.

---

## ✨ Features

- 🔍 **Super-Resolution (SRCNN & Real-ESRGAN)**: Upscale low-resolution images up to 4K quality while recovering fine textures and crisp edges.
- 👤 **Face Restoration (GFPGAN)**: Automatically detect facial regions and restore blurred or degraded facial details using generative facial priors.
- 🎨 **Deep Colorization**: Automatically inject vibrant, natural color into historical and monochrome images using a deep convolutional colorization model.
- ⚡ **Interactive Before/After Slider**: High-performance side-by-side interactive visual comparison powered by Next.js and `react-compare-slider`.
- 🌐 **Modern Next.js Frontend**: Dark-themed, responsive web UI with drag-and-drop file upload, clipboard paste support, and real-time adjustment sliders.
- 🚀 **FastAPI Backend**: Asynchronous REST API ready for local deployment or GPU servers (NVIDIA DGX / A100 / Colab).
- 🎛️ **Gradio Web Interface**: Quick demo interface (`app.py`) for standalone deployment and testing.

---

## 📁 Project Structure

```plaintext
SRCNN_PROJECT/
├── app.py                      # Standalone Gradio interface for GPU/local testing
├── best_srcnn.pth              # Pretrained SRCNN PyTorch model weights
├── srcnn.pth                   # Base SRCNN model weights
├── test.py                     # Single-image inference script for SRCNN
├── requirements.txt            # Python dependencies for the core model & Gradio
├── setup_remote.sh             # Linux/GPU server setup script
├── backend_api/
│   └── backend/
│       ├── server.py           # FastAPI backend server with GFPGAN & Real-ESRGAN
│       ├── requirements.txt    # Backend dependencies
│       └── weights/            # Auto-downloaded model weight checkpoints
├── frontend/
│   ├── src/
│   │   └── app/
│   │       ├── page.tsx        # Next.js interactive image enhancer UI
│   │       ├── layout.tsx
│   │       └── globals.css
│   ├── package.json            # Node.js dependencies
│   └── tsconfig.json
└── data/                       # Sample test images
```

---

## 🛠️ Quick Start

### 1. Backend (FastAPI)

Navigate to the backend directory and install the requirements:

```bash
cd backend_api/backend
pip install -r requirements.txt
```

Run the FastAPI server:

```bash
python server.py
# The server will automatically download model weights on the first run and listen on http://localhost:8000
```

### 2. Frontend (Next.js)

Navigate to the frontend directory:

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to interact with the enhancement interface.

### 3. Standalone Gradio App

Alternatively, you can run the Gradio interface directly:

```bash
pip install -r requirements.txt
python app.py
```

---

## 🧠 Models & Technologies

- **SRCNN**: Super-Resolution Convolutional Neural Network
- **Real-ESRGAN**: Practical restoration of general and complex degraded real-world images
- **GFPGAN**: Practical face restoration algorithm towards real-world face degradation
- **FastAPI & Uvicorn**: High-performance Python async backend
- **Next.js 14+ & React**: Modern front-end with server/client components and TypeScript

---

## 📜 License

This project is licensed under the MIT License.
