import os
import sys
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import UNet
from data_loading import get_image_paths, BCDataset
from augmentation import get_augmentations

def dice_loss(inputs, target):
    inputs = torch.sigmoid(inputs)
    smooth = 1e-6  # Avoid division by zero
    iflat = inputs.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    loss = 1 - (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return loss

def plot_training_loss(losses, save_path='graphs/training_loss_curve.png'):
    plt.figure()
    plt.plot(losses, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    plt.savefig(save_path)
    plt.close()

def train_model():
    train_tuples, test_tuples = get_image_paths()
    augmentations = get_augmentations()
    
    train_dataset = BCDataset(train_tuples, augmentations=augmentations)
    test_dataset = BCDataset(test_tuples)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1).to(device)
    
    # Use Adam optimizer for faster convergence
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()  # Using built-in loss

    num_epochs = 10
    losses = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = dice_loss(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)

        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        losses.append(avg_epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_epoch_loss:.4f}")

    torch.save(model.state_dict(), 'unet_model.pth')
    plot_training_loss(losses)

if __name__ == '__main__':
    train_model()
