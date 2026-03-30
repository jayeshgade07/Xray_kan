import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks to capture feature maps and gradients during backprop
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]
        
    def __call__(self, x, class_idx=None):
        """
        Generate Grad-CAM heatmap for a target class.
        x: input image tensor [1, 3, 224, 224]
        """
        self.model.eval()
        outputs = self.model(x)
        
        if class_idx is None:
            class_idx = outputs.argmax(dim=1).item()
            
        self.model.zero_grad()
        score = outputs[0, class_idx]
        score.backward(retain_graph=True)
        
        # Pool the gradients across spatial dimensions
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the channels by the gradients
        activations = self.activations[0].detach()
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]
            
        # Average the channels to get the heatmap
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        
        # ReLU mapping and normalization
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        return heatmap

def overlay_cam(img_tensor, heatmap, save_path=None):
    """
    Overlays the Grad-CAM heatmap onto the original image tensor and saves it.
    """
    img = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
    
    # Denormalize ImageNet scaling to visualize the base X-Ray
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_jet = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Convert jet colormap to RGB for matplotlib/saving
    heatmap_jet = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)
    
    superimposed_img = heatmap_jet * 0.4 + (img * 255) * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    if save_path:
        # Convert back to BGR before saving with cv2
        cv2.imwrite(save_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
        
    return superimposed_img
