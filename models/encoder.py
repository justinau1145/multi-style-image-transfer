import torch
import torch.nn as nn
from torchvision import models

class VGGEncoder(nn.Module):
    """VGG-19 convolutional encoder truncated at the `relu4_1` layer.

    This module uses a pretrained VGG-19 network and retains only the
    convolutional layers up to and including `relu4_1`. It is intended 
    for feature extraction and does not update weights during training.

    Attributes:
        encoder: Truncated VGG-19 feature extractor
    """
    def __init__(self) -> None:
        """Initialize the VGG-19 encoder.

        Loads VGG-19 pretrained on ImageNet, truncates the network at
        `relu4_1`, and freezes all parameters to prevent gradient 
        updates.
        """
        super(VGGEncoder, self).__init__()
        
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        self.slice1 = vgg[:2]    # Up to relu1_1
        self.slice2 = vgg[2:7]   # Up to relu2_1
        self.slice3 = vgg[7:12]  # Up to relu3_1
        self.slice4 = vgg[12:21] # Up to relu4_1

        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, x: torch.Tensor, 
                output_last_only: bool = True) -> torch.Tensor:
        """Passes the input through the truncated VGG-19 network.

        Args:
            x: Input tensor expected to be normalized with ImageNet 
            statistics

        Returns:
            torch.Tensor: Feature maps extracted from the `relu4_1` 
            layer.
        """
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        
        if output_last_only:
            return h4
        else:
            return [h1, h2, h3, h4]