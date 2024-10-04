# import albumentations as A

# def get_augmentations():
#     return A.Compose([
#         A.RandomRotate90(),
#         A.Flip(),
#         A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=None, p=0.5),
#         A.RandomBrightnessContrast(p=0.2),
#         A.Normalize(),
#         A.Resize(512, 512)  # Adjust based on your desired size
#     ])

import albumentations as A

def get_augmentations():
    return A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=None, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        A.Resize(512, 512)  # Adjust based on your desired size
    ])
