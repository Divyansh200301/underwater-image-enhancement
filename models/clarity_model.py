"""
Clarity enhancement model
Improves sharpness, contrast, and overall image quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for feature extraction"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class AttentionBlock(nn.Module):
    """Channel attention block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ClarityNet(nn.Module):
    """
    Clarity enhancement network
    
    Architecture:
    - Feature extraction with residual blocks
    - Channel attention mechanism
    - Progressive refinement
    - Skip connections for detail preservation
    """
    def __init__(self, n_channels=3, n_features=64, n_residual_blocks=6):
        super().__init__()
        
        # Initial feature extraction
        self.conv_in = nn.Sequential(
            nn.Conv2d(n_channels, n_features, kernel_size=7, padding=3),
            nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks with attention
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(n_features))
            res_blocks.append(AttentionBlock(n_features))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Feature fusion
        self.conv_mid = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True)
        )
        
        # Sharpness enhancement branch
        self.sharp_branch = nn.Sequential(
            nn.Conv2d(n_features, n_features // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features // 2, n_features // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_features // 2),
            nn.ReLU(inplace=True)
        )
        
        # Contrast enhancement branch
        self.contrast_branch = nn.Sequential(
            nn.Conv2d(n_features, n_features // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features // 2, n_features // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_features // 2),
            nn.ReLU(inplace=True)
        )
        
        # Fusion and output
        self.conv_out = nn.Sequential(
            nn.Conv2d(n_features, n_features // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features // 2, n_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        # Store input for residual connection
        input_img = x
        
        # Feature extraction
        feat = self.conv_in(x)
        feat_res = self.res_blocks(feat)
        feat = feat + feat_res  # Global residual
        feat = self.conv_mid(feat)
        
        # Dual branch processing
        sharp_feat = self.sharp_branch(feat)
        contrast_feat = self.contrast_branch(feat)
        
        # Combine features
        combined = torch.cat([sharp_feat, contrast_feat], dim=1)
        
        # Generate output
        output = self.conv_out(combined)
        output = torch.sigmoid(output)
        
        # Add residual connection from input
        # This helps preserve original structure while enhancing clarity
        output = output * 0.8 + input_img * 0.2
        output = torch.clamp(output, 0, 1)
        
        return output


class ClarityModel:
    """
    Wrapper class for clarity enhancement model with training and inference methods
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None
    
    def create_model(self, n_features=64, n_residual_blocks=6):
        """
        Create clarity enhancement model
        
        Args:
            n_features: Number of feature channels
            n_residual_blocks: Number of residual blocks
        """
        self.model = ClarityNet(n_channels=3, n_features=n_features, 
                                n_residual_blocks=n_residual_blocks)
        self.model = self.model.to(self.device)
        print(f"Clarity model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def setup_training(self, learning_rate=1e-4):
        """
        Setup optimizer and loss function
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Use L1 loss for better edge preservation
        self.criterion = nn.L1Loss()
        
        print(f"Training setup complete with lr={learning_rate}")
    
    def train_step(self, inputs, targets):
        """
        Single training step
        
        Args:
            inputs: Input images tensor
            targets: Target images tensor
            
        Returns:
            Loss value
        """
        self.model.train()
        
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Calculate loss (L1 loss for edge preservation)
        loss = self.criterion(outputs, targets)
        
        # Add edge-aware loss component with NaN check
        edge_loss = self._edge_loss(outputs, targets)
        
        # Check for NaN/Inf and skip if found
        if torch.isnan(edge_loss) or torch.isinf(edge_loss):
            edge_loss = torch.tensor(0.0).to(self.device)
        
        total_loss = loss + 0.1 * edge_loss
        
        # Final NaN check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: NaN/Inf detected in loss, skipping batch")
            return 0.0
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return total_loss.item()
    
    def _edge_loss(self, pred, target):
        """Calculate edge-aware loss using Sobel filters with numerical stability"""
        # Sobel filter for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        
        # Calculate edges for each channel
        pred_edges = []
        target_edges = []
        
        for i in range(3):  # RGB channels
            pred_ch = pred[:, i:i+1, :, :]
            target_ch = target[:, i:i+1, :, :]
            
            pred_x = F.conv2d(pred_ch, sobel_x, padding=1)
            pred_y = F.conv2d(pred_ch, sobel_y, padding=1)
            # Add epsilon for numerical stability
            pred_edge = torch.sqrt(pred_x**2 + pred_y**2 + 1e-6)
            
            target_x = F.conv2d(target_ch, sobel_x, padding=1)
            target_y = F.conv2d(target_ch, sobel_y, padding=1)
            # Add epsilon for numerical stability
            target_edge = torch.sqrt(target_x**2 + target_y**2 + 1e-6)
            
            pred_edges.append(pred_edge)
            target_edges.append(target_edge)
        
        pred_edges = torch.cat(pred_edges, dim=1)
        target_edges = torch.cat(target_edges, dim=1)
        
        return F.l1_loss(pred_edges, target_edges)
    
    def validate(self, val_loader):
        """
        Validation loop
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def infer(self, image_tensor):
        """
        Run inference on a single image
        
        Args:
            image_tensor: Input image tensor (1, C, H, W) or (C, H, W)
            
        Returns:
            Enhanced image tensor
        """
        self.model.eval()
        
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
        
        return output
    
    def save(self, path):
        """
        Save model weights
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """
        Load model weights
        
        Args:
            path: Path to model weights
        """
        if self.model is None:
            self.create_model()
        
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model = self.model.to(self.device)
        print(f"Model loaded from {path}")


def create_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Factory function to create clarity model
    
    Args:
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        ClarityModel instance
    """
    model = ClarityModel(device=device)
    model.create_model()
    return model


if __name__ == '__main__':
    # Test model creation
    model = create_model()
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 256, 256).to(model.device)
    output = model.infer(dummy_input)
    print(f"Output shape: {output.shape}")
