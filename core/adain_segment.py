import torch
import torch.nn.functional as F
import numpy as np
from typing import List

def calc_mean_std(feat: torch.Tensor, 
                  eps: float = 1e-5) -> tuple[torch.Tensor, 
                                              torch.Tensor]:
    """Compute channel-wise mean and standard deviation of feature maps.

    Statistics are computed independently for each channel across the
    spatial dimensions (H, W), and reshaped for broadcasting.

    Args:
        features: Feature tensor
        eps: Small constant to avoid division by zero. Defaults to 1e-5.

    Returns:
        Tuple containing:
            - mean: Spatial mean
            - std: Spatial standard deviation
    """
    size = feat.size()
    assert len(size) == 4, "Input must be 4D tensor"
    
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    
    return feat_mean, feat_std

def multi_adain(content_feat: torch.Tensor, 
                style_feats: List[torch.Tensor], masks: List[torch.Tensor],
                device: torch.device) -> torch.Tensor:
    """
    Apply different AdaIN transformations to different regions.
    
    This function segments the content features into different regions
    using masks and applies different style statistics to each region.
    
    Args:
        content_feat: Content feature map.
        style_feats: List of style feature maps.
        masks: List of binary masks for each region
        device: Device for computation.

    Returns:
        torch.Tensor: Multi-style normalized features.
    """
    size = content_feat.size()
    _, C, H, W = size
    
    content_mean, content_std = calc_mean_std(content_feat)
    
    # Normalize content features
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    output = torch.zeros_like(content_feat).to(device)
    
    # Process each style-region pair
    for style_feat, mask in zip(style_feats, masks):
        style_mean, style_std = calc_mean_std(style_feat)
        stylized = normalized_feat * style_std.expand(size) + style_mean.expand(size)
        output = output + stylized * mask
    
    # Create total mask to find uncovered regions
    total_mask = torch.zeros((size[0], 1, H, W), device=device)
    for mask in masks:
        total_mask = torch.clamp(total_mask + mask, 0, 1)
    
    output = output + content_feat * (1 - total_mask)
    
    return output


def multi_adain_alpha(content_feat: torch.Tensor, 
                      style_feats: List[torch.Tensor], 
                      masks: List[torch.Tensor], alpha: float,
                      device: torch.device) -> torch.Tensor:
    """
    Apply multi-style AdaIN with intensity control.
    
    Args:
        content_feat: Content feature map.
        style_feats: List of style feature maps.
        masks: List of region masks.
        alpha: Style transfer intensity (0-1).
        device: Computation device.
    
    Returns:
        torch.Tensor: Blended features with controlled style intensity
    """
    stylized_feat = multi_adain(content_feat, style_feats, masks, device)

    return alpha * stylized_feat + (1 - alpha) * content_feat


# Alias for convenience
multi_adain = multi_adain
