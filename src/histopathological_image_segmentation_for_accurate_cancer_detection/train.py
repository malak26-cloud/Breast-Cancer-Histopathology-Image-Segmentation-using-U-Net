import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.histopathological_image_segmentation_for_accurate_cancer_detection.model import UNet
from src.histopathological_image_segmentation_for_accurate_cancer_detection.data_loading import get_image_paths, BCDataset
from src.histopathological_image_segmentation_for_accurate_cancer_detection.augmentation import get_augmentations

def dice_loss(inputs, target):
    inputs = torch.sigmoid(inputs)
    smooth = 1.
    iflat = inputs.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    loss = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
    return loss

def plot_training_loss(losses, save_path='graphs/training_loss_curve.png'):
    plt.figure()
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def train_model():
    # Get data paths and augmentations
    train_tuples, test_tuples = get_image_paths()
    augmentations = get_augmentations()
    
    # Prepare datasets and loaders
    train_dataset = BCDataset(train_tuples, augmentations=augmentations)
    test_dataset = BCDataset(test_tuples)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Set up device, model, criterion, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.5)

    # Training loop
    num_epochs = 100
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
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_epoch_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'unet_model.pth')

    # Plot training loss
    plot_training_loss(losses)

if __name__ == '__main__':
    train_model()
