# from flask import Flask, request, jsonify, send_file
# import torch
# from PIL import Image
# import io
# import numpy as np
# from torchvision import transforms
# from model import UNet

# # Initialize Flask app
# app = Flask(__name__)

# # Load model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = UNet(3,1)  # Replace with your UNet model class
# checkpoint_path = r"C:\Users\exact\OneDrive\Desktop\histopathological image segmentation for breast cancer\unet\checkpoints\checkpoint_epoch_9_acc_0.9222.pth"

# # Load model weights
# checkpoint = torch.load(checkpoint_path, map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# # Define transformation for input image
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # Resize image for your model if necessary
#     transforms.ToTensor()
# ])

# def prepare_image(image):
#     """Prepare the image for model input."""
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     return image.to(device)

# @app.route('/segment', methods=['POST'])
# def segment_image():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No image file provided'}), 400

#     file = request.files['file']

#     try:
#         # Open the image file
#         img = Image.open(file.stream)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

#     # Prepare the image and make the prediction
#     input_tensor = prepare_image(img)
#     with torch.no_grad():
#         output = model(input_tensor)

#     # Convert the output tensor to a NumPy array and apply thresholding
#     output = torch.sigmoid(output).squeeze().cpu().numpy()
#     mask = (output > 0.5).astype(np.uint8) * 255  # Convert to binary mask

#     # Convert the mask to a PIL image
#     mask_img = Image.fromarray(mask)

#     # Save the mask to a BytesIO object to return as a response
#     byte_io = io.BytesIO()
#     mask_img.save(byte_io, 'PNG')
#     byte_io.seek(0)

#     return send_file(byte_io, mimetype='image/png')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify, send_file
from PIL import Image
import torch
from torchvision import transforms
import io
import numpy as np
from model import UNet  # Import the model from your model.py

# Initialize the Flask app
app = Flask(__name__)

# Load model and checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model from model.py
model = UNet(n_channels=3, n_classes=1).to(device)
checkpoint = torch.load(r"C:\Users\exact\OneDrive\Desktop\histopathological image segmentation for breast cancer\unet\checkpoints\checkpoint_epoch_9_acc_0.9222.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define a function to preprocess the image
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Define a function to postprocess the mask and convert it to an image
def postprocess_mask(mask_tensor):
    mask = torch.sigmoid(mask_tensor).squeeze().cpu().detach().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255  # Threshold to binary mask
    return Image.fromarray(mask)

# Define the API endpoint
@app.route('/segment', methods=['POST'])
def segment():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        print("Received file:", file.filename)

        # Preprocess the image
        image = preprocess_image(file.read()).to(device)
        print("Image preprocessed.")

        # Run the model to get the mask
        with torch.no_grad():
            output = model(image)
        print("Model inference completed.")

        # Postprocess the mask
        mask_image = postprocess_mask(output)
        print("Mask postprocessed.")

        # Save mask as image in-memory
        img_io = io.BytesIO()
        mask_image.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        print("Error:", str(e))  # Print the error to console
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
