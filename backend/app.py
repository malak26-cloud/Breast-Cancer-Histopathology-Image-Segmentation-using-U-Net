from flask import Flask, request, jsonify, send_file
from PIL import Image
import torch
from torchvision import transforms

import io
import numpy as np
from model import UNet  


app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = UNet(n_channels=3, n_classes=1).to(device)
checkpoint = torch.load(r"C:\Users\exact\OneDrive\Desktop\histopathological image segmentation for breast cancer\unet\checkpoints\checkpoint_epoch_9_acc_0.9222.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)  
    return image

def postprocess_mask(mask_tensor):
    mask = torch.sigmoid(mask_tensor).squeeze().cpu().detach().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255  
    return Image.fromarray(mask)


@app.route('/segment', methods=['POST'])
def segment():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        print("Received file:", file.filename)

        
        image = preprocess_image(file.read()).to(device)
        print("Image preprocessed.")

        
        with torch.no_grad():
            output = model(image)
        print("Model inference completed.")

        
        mask_image = postprocess_mask(output)
        print("Mask postprocessed.")

        
        img_io = io.BytesIO()
        mask_image.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        print("Error:", str(e))  
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
