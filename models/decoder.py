import torch
import torch.nn as nn

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
        
        # Mirror of VGG encoder (inverse architecture)
        self.decoder = nn.Sequential(
            # Inverse of relu4_1 to relu3_1
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3, 1, 0),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(inplace=True),
            
            # Inverse of relu3_1 to relu2_1
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3, 1, 0),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3, 1, 0),
            nn.ReLU(inplace=True),
            
            # Inverse of relu2_1 to relu1_1
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, 1, 0),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.ReLU(inplace=True),
            
            # Final layer to RGB
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
        return self.decoder(x)
