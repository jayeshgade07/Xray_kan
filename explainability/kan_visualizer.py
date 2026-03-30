import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def extract_kan_functions(kan_layer, input_range=(-2, 2), num_points=100):
    """
    Extracts the 1D function mappings learned by the KAN layer using B-splines.
    This reveals the true "interpretable" non-linear relationships.
    """
    x = torch.linspace(input_range[0], input_range[1], num_points)
    x_tensor = x.view(-1, 1).expand(-1, kan_layer.in_features) # (num_points, in_features)
    
    # Isolate the response of each specific edge (in_feature -> out_feature)
    responses = np.zeros((kan_layer.out_features, kan_layer.in_features, num_points))
    
    with torch.no_grad():
        basis = kan_layer.b_splines(x_tensor) # (num_points, in_features, grid_size+spline_order)
        
        for out_f in range(kan_layer.out_features):
            for in_f in range(kan_layer.in_features):
                b_vals = basis[:, in_f, :] # (num_points, grid_size+spline_order)
                w_vals = kan_layer.spline_weight[out_f, in_f, :].unsqueeze(0) # (1, grid_size+spline_order)
                
                base_act = kan_layer.base_activation(x_tensor[:, in_f])
                base_w = kan_layer.base_weight[out_f, in_f]
                
                spline_out = torch.sum(b_vals * w_vals, dim=1)
                base_out = base_act * base_w
                
                val = base_out + spline_out
                responses[out_f, in_f, :] = val.numpy()
                    
    return x.numpy(), responses

def plot_kan_activations(kan_layer, target_class=0, class_name="Disease", top_k_features=5, save_dir='results/plots'):
    """
    Plots the non-linear transformation functions for the most important features 
    determining a specific target class.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    x, responses = extract_kan_functions(kan_layer.cpu())
    
    # Feature importance = variance of the learned function across the typical input domain
    variances = np.var(responses[target_class], axis=1) # Shape: (in_features,)
    top_indices = np.argsort(variances)[-top_k_features:][::-1]
    
    plt.figure(figsize=(10, 6))
    for idx in top_indices:
        plt.plot(x, responses[target_class, idx, :], 
                 label=f"CNN Feature {idx} (Var: {variances[idx]:.2f})", linewidth=2)
        
    plt.title(f"Learned KAN Activations mapping to {class_name} (Class {target_class})")
    plt.xlabel("CNN Feature Output Value")
    plt.ylabel("KAN Edge Logit Contribution")
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(save_dir, f'kan_activations_class_{target_class}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved KAN visualization to {save_path}")
