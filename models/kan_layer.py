import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Linear approximation layer using B-splines.
    This acts as a high-capacity interpretable replacement for nn.Linear.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=nn.SiLU):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Base linear transformation
        self.base_activation = base_activation()
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Grid parameters for B-splines
        h = 1 / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h
        self.register_buffer("grid", grid)
        
        # B-spline weights
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # Corrected initialization: Use small random noise instead of ones to prevent exploding logits
            nn.init.normal_(
                self.spline_weight, 
                mean=0.0, 
                std=self.scale_noise / (math.sqrt(self.in_features) * (self.grid_size + self.spline_order))
            )

    def b_splines(self, x):
        # x is (batch_size, in_features)
        x = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left_term = (x - grid[:-(k + 1)]) / (grid[k:-1] - grid[:-(k + 1)]) * bases[:, :, :-1]
            right_term = (grid[k + 1:] - x) / (grid[k + 1:] - grid[1:-k]) * bases[:, :, 1:]
            bases = left_term + right_term
        return bases.contiguous()

    def forward(self, x):
        # Base Linear logic
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # Spline component logic
        spline_basis = self.b_splines(x) # (batch, in_features, grid_size+spline_order)
        spline_basis = spline_basis.view(x.size(0), -1) # flatten
        spline_weights = self.spline_weight.view(self.out_features, -1) # flatten
        
        spline_output = F.linear(spline_basis, spline_weights)
        return base_output + spline_output
