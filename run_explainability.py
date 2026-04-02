import os
import torch
import torch.nn as nn
from data.dataset_factory import get_dataloaders
from models.cnn_kan_model import CNNBaseline, CNNKAN
from explainability.gradcam import GradCAM, overlay_cam
import numpy as np
import cv2

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs('results/plots/gradcam', exist_ok=True)
    
    # 1. Load data
    _, _, test_loader = get_dataloaders('nih', data_dir='./data', batch_size=4)
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    # 2. Load Models
    cnn_model = CNNBaseline(num_classes=14)
    cnn_model.load_state_dict(torch.load('results/checkpoints/best_cnn_nih.pth', map_location=device))
    cnn_model.to(device)
    cnn_model.eval()

    kan_model = CNNKAN(num_classes=14)
    kan_model.load_state_dict(torch.load('results/checkpoints/best_kan_nih.pth', map_location=device))
    kan_model.to(device)
    kan_model.eval()

    # 3. Setup GradCAM on the backbone's feature extractor
    target_layer_cnn = cnn_model.backbone.features
    cam_cnn = GradCAM(cnn_model, target_layer_cnn)
    
    target_layer_kan = kan_model.backbone.features
    cam_kan = GradCAM(kan_model, target_layer_kan)
    
    # 4. Generate visualiztions
    for i in range(len(images)):
        img_tensor = images[i].unsqueeze(0)
        
        # We'll trigger CAM for whichever class the model predicts highest
        # (Though we could also force it to visualize the true label if it's 1-hot)
        try:
            heatmap_cnn = cam_cnn(img_tensor)
            heatmap_kan = cam_kan(img_tensor)
            
            save_path_cnn = f'results/plots/gradcam/img_{i}_cnn.png'
            save_path_kan = f'results/plots/gradcam/img_{i}_kan.png'
            
            overlay_cam(img_tensor, heatmap_cnn, save_path=save_path_cnn)
            overlay_cam(img_tensor, heatmap_kan, save_path=save_path_kan)
            
            print(f"[{i+1}/4] Saved Grad-CAMs for Image {i} - Check 'results/plots/gradcam'")
        except Exception as e:
            print(f"Error processing image {i}: {e}")

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    main()
