# import os
# from glob import glob
# import random
# import os.path as osp
# import cv2
# import torch
# import numpy as np
# from skimage import io
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

# # Define paths for images and masks
# BASE_PATH = r"C:\Users\exact\OneDrive\Desktop\histopathological image segmentation for breast cancer\unet\Dataset"
# IMAGES_PATH = osp.join(BASE_PATH, 'images')  # Images folder
# LABELS_PATH = osp.join(BASE_PATH, 'masks')   # Masks folder

# # Function to get image and mask paths
# def get_image_paths():
#     imgs_paths = glob(osp.join(IMAGES_PATH, "*.jpg"))  # Collect all jpg images
#     masks_paths = glob(osp.join(LABELS_PATH, "*.jpg"))
#     img_mask_tuples = list(zip(imgs_paths, masks_paths))  # Create list of tuples of (image, mask)
#     random.shuffle(img_mask_tuples)  # Shuffle the tuples to randomize data
#     train_tuples, test_tuples = train_test_split(img_mask_tuples, test_size=0.2)  # Split data into train and test
#     return train_tuples, test_tuples

# # Function to load and preprocess image (resize, normalize)
# def get_tiff_image(path, normalized=True, resize=(512, 512)):
#     image = io.imread(path)  # Load image
#     if image.ndim == 3:  # If image is RGB (3 channels)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
#     image = cv2.resize(image, resize)  # Resize image
#     if normalized:
#         return image.astype(np.float32) / 255.0  # Normalize image
#     return image

# # Function to count images in a directory
# def count_images_in_directory(directory, extension=".jpg"):
#     image_count = len([file for file in os.listdir(directory) if file.endswith(extension)])  # Count images by extension
#     print(f"Total number of images in '{directory}': {image_count}")
#     return image_count

# # Function to count masks in a directory
# def count_masks_in_directory(directory, extension=".jpg"):
#     mask_count = len([file for file in os.listdir(directory) if file.endswith(extension)])  # Count masks by extension
#     print(f"Total number of masks in '{directory}': {mask_count}")
#     return mask_count

# # Class to handle the dataset and loading images and masks
# class BCDataset(torch.utils.data.Dataset):
#     def __init__(self, img_mask_tuples, augmentations=None):
#         self.img_mask_tuples = img_mask_tuples  # List of image-mask tuples
#         self.augmentations = augmentations  # Data augmentation, if any
        
#     def __len__(self):
#         return len(self.img_mask_tuples)  # Return total number of tuples
    
#     def __getitem__(self, idx):
#         img_path, mask_path = self.img_mask_tuples[idx]
#         image = get_tiff_image(img_path)
#         mask = get_tiff_image(mask_path, normalized=False)
#         mask[mask > 0] = 1

#         if self.augmentations:
#             augmented = self.augmentations(image=image, mask=mask)
#             image = augmented['image']
#             mask = augmented['mask']

#     # Convert the image and mask to torch tensors and permute the dimensions
#         image = torch.tensor(image).permute(2, 0, 1)  # Change from (H, W, C) to (C, H, W)
#         mask = torch.tensor(mask).permute(2, 0, 1) if mask.ndim == 3 else torch.tensor(mask).unsqueeze(0)

#         return image, mask

       
        
    

# # Example: Count the number of images and masks in their respective directories
# count_images_in_directory(IMAGES_PATH)
# count_masks_in_directory(LABELS_PATH)

# # Example usage: Get training and testing tuples
# train_tuples, test_tuples = get_image_paths()
# print(f"Total training samples: {len(train_tuples)}")
# print(f"Total testing samples: {len(test_tuples)}")
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
