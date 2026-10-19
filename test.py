import torch
import cv2
from model import SRCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SRCNN().to(device)
model.load_state_dict(torch.load("best_srcnn.pth", map_location=device, weights_only=True))
model.eval()

# 👉 YOUR IMAGE PATH
img = cv2.imread(r"C:\Users\lokes\OneDrive\Desktop\coding\SRCNN_PROJECT\data\img3.webp")

# Preprocess
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, _ = img.shape
scale = 4

lr = cv2.resize(img, (w//scale, h//scale), interpolation=cv2.INTER_CUBIC)
lr = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)

input_img = lr / 255.0
input_img = torch.tensor(input_img).permute(2,0,1).unsqueeze(0).float().to(device)

# Predict
with torch.no_grad():
    output = model(input_img)

# Postprocess
output = output.squeeze().permute(1,2,0).cpu().numpy()
output = (output * 255).clip(0,255).astype("uint8")

# Save
cv2.imwrite("enhanced_output.png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

print(" Output saved as enhanced_output.png")