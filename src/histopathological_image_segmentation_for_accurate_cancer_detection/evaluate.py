import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from src.histopathological_image_segmentation_for_accurate_cancer_detection.model import UNet
from src.histopathological_image_segmentation_for_accurate_cancer_detection.data_loading import get_image_paths, BCDataset

def evaluate_model():
    _, test_tuples = get_image_paths()
    test_dataset = BCDataset(test_tuples)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load('unet_model.pth'))
    model.to(device)

    model.eval()
    predictions, labels_list = [], []
    with torch.no_grad():
        for images, labels in iter(test_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            pred = torch.sigmoid(output)
            pred = (pred > 0.5).float()
            predictions.append(pred.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    labels_list = np.concatenate(labels_list)

    accuracy = accuracy_score(labels_list.flatten(), predictions.flatten())
    precision = precision_score(labels_list.flatten(), predictions.flatten())
    recall = recall_score(labels_list.flatten(), predictions.flatten())
    f1 = f1_score(labels_list.flatten(), predictions.flatten())
    iou = jaccard_score(labels_list.flatten(), predictions.flatten())

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"IOU (Jaccard Score): {iou}")

if __name__ == '__main__':
    evaluate_model()
