import tifffile
from skimage import io
import cv2
import torch
from glob import glob
import os.path as osp
import random
from sklearn.model_selection import train_test_split

BASE_PATH = '/kaggle/input/breast-cancer-cell-segmentation'
IMAGES_PATH = osp.join(BASE_PATH, 'Images')
LABELS_PATH = osp.join(BASE_PATH, 'Masks')

def get_image_paths():
    imgs_paths = glob(osp.join(IMAGES_PATH, "*.tif"))
    masks_paths = [osp.join(LABELS_PATH, i.rsplit("/", 1)[-1].split("_ccd")[0] + ".TIF") for i in imgs_paths]
    img_mask_tuples = list(zip(imgs_paths, masks_paths))
    random.shuffle(img_mask_tuples)
    train_tuples, test_tuples = train_test_split(img_mask_tuples, test_size=0.2)
    return train_tuples, test_tuples

def get_tiff_image(path, normalized=True, resize=(512, 512)):
    image = io.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, resize)
    if normalized:
        return image / 255.0
    return image

class BCDataset(torch.utils.data.Dataset):
    def __init__(self, img_mask_tuples, augmentations=None):
        self.img_mask_tuples = img_mask_tuples
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.img_mask_tuples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.img_mask_tuples[idx]
        image = get_tiff_image(img_path)
        mask = get_tiff_image(mask_path, normalized=False)
        mask[mask > 0] = 1
        
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask
