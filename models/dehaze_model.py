"""
Dehazing model based on U-Net architecture
Removes haze and color distortion from underwater images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad x1 to match x2 size if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for image dehazing
    
    Architecture:
    - Encoder: 4 downsampling blocks
    - Bottleneck: Double conv
    - Decoder: 4 upsampling blocks with skip connections
    - Output: 3-channel RGB image
    """
    def __init__(self, n_channels=3, n_classes=3, base_channels=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)
        
        # Output
        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        output = torch.sigmoid(logits)
        
        return output


class DehazeModel:
    """
    Wrapper class for dehazing model with training and inference methods
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None
    
    def create_model(self, base_channels=64):
        """
        Create U-Net model for dehazing
        
        Args:
            base_channels: Base number of channels (default: 64)
        """
        self.model = UNet(n_channels=3, n_classes=3, base_channels=base_channels)
        self.model = self.model.to(self.device)
        print(f"Dehazing model created with {sum(p.numel() for p in self.model.parameters())} parameters")
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
        
        # Combined loss: MSE + perceptual-like loss
        self.criterion = nn.MSELoss()
        
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
        
        # Calculate loss
        loss = self.criterion(outputs, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
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
            Dehazed image tensor
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
    Factory function to create dehazing model
    
    Args:
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        DehazeModel instance
    """
    model = DehazeModel(device=device)
    model.create_model()
    return model


if __name__ == '__main__':
    # Test model creation
    model = create_model()
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 256, 256).to(model.device)
    output = model.infer(dummy_input)
    print(f"Output shape: {output.shape}")
