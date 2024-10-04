import os
from glob import glob
import random
import os.path as osp
import cv2
import torch
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split

# Define paths for images and masks
BASE_PATH = r"C:\Users\exact\OneDrive\Desktop\histopathological image segmentation for breast cancer\unet\Dataset"
IMAGES_PATH = osp.join(BASE_PATH, 'images')
LABELS_PATH = osp.join(BASE_PATH, 'masks')

def get_image_paths():
    imgs_paths = glob(osp.join(IMAGES_PATH, "*.jpg"))
    masks_paths = glob(osp.join(LABELS_PATH, "*.jpg"))
    img_mask_tuples = list(zip(imgs_paths, masks_paths))
    random.shuffle(img_mask_tuples)
    train_tuples, test_tuples = train_test_split(img_mask_tuples, test_size=0.2, random_state=42)
    return train_tuples, test_tuples

def get_image(path, normalize=True, resize=(512, 512)):
    """Load and preprocess an image."""
    image = io.imread(path)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, resize)
    if normalize:
        return image.astype(np.float32) / 255.0
    return image

class BCDataset(torch.utils.data.Dataset):
    def __init__(self, img_mask_tuples, augmentations=None):
        self.img_mask_tuples = img_mask_tuples
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.img_mask_tuples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.img_mask_tuples[idx]
        
        # Load and process the image and mask
        image = get_image(img_path)
        mask = get_image(mask_path, normalize=False)
        mask[mask > 0] = 1  # Binarize the mask

        # Apply augmentations if provided
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert to tensor format
        image = torch.tensor(image).permute(2, 0, 1)  # Change to (C, H, W)
        mask = torch.tensor(mask).unsqueeze(0) if mask.ndim == 2 else torch.tensor(mask).permute(2, 0, 1)

        return image, mask
