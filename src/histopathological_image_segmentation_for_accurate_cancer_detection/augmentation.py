from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, RandomBrightnessContrast, ElasticTransform

def get_augmentations():
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        RandomBrightnessContrast(p=0.2),
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5)
    ])
