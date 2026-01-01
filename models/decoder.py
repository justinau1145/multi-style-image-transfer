import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """Reconstructs an image from VGG-19 features.

    This network acts as the inverse of the VGG-19 encoder truncated at 
    `relu4_1`. It mirrors the encoder's architecture, replacing pooling 
    layers with nearest-neighbor upsampling to recover spatial 
    resolution.

    Attributes:
        decoder: Stack of convolutional and upsampling layers converting 
        512-channel feature maps back to a 3-channel RGB image.
    """
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        
        # Inverse of relu4_1 to relu3_1
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3, 1, 0),
            nn.ReLU(inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(inplace=True),
        )
        
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(inplace=True),
        )
        
        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(inplace=True),
        )
        
        # Inverse of relu3_1 to relu2_1
        self.conv5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3, 1, 0),
            nn.ReLU(inplace=True),
        )
        
        self.conv6 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3, 1, 0),
            nn.ReLU(inplace=True),
        )
        
        # Inverse of relu2_1 to relu1_1
        self.conv7 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, 1, 0),
            nn.ReLU(inplace=True),
        )
        
        self.conv8 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.ReLU(inplace=True),
        )
        
        # Final layer to RGB (no ReLU)
        self.conv9 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3, 1, 0),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to reconstruct the image.

        Args:
            x: Feature maps. These are typically the output of the AdaIN 
            layer.

        Returns:
            torch.Tensor: Reconstructed images. Output values are not 
            constrained to a specific range and may require 
            post-processing such as clamping or denormalization.
        """
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv6(x)
        x = self.conv7(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv8(x)
        x = self.conv9(x)

        return x
