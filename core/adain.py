import torch

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

def adaptive_instance_normalization(content_feat: torch.Tensor, 
                                  style_feat: torch.Tensor) -> torch.Tensor:
    """Apply Adaptive Instance Normalization (AdaIN).

    AdaIN aligns the channel-wise mean and standard deviation of the
    content features with those of the style features, enabling
    style transfer while preserving content structure.

    Args:
        content_feat: Content feature map
        style_feat: Style feature map of shape

    Returns:
        torch.Tensor: The stylized feature maps

    Raises:
        AssertionError: If input tensors are not 4D or if their batch
            size and channel dimensions do not match.
    """
    assert content_feat.size()[:2] == style_feat.size()[:2], \
        "Content and style features must have same batch size and channels"
    
    size = content_feat.size()
    
    # Calculate statistics
    content_mean, content_std = calc_mean_std(content_feat)
    style_mean, style_std = calc_mean_std(style_feat)
    
    # Normalize content features
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    
    # Apply style statistics
    output = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    
    return output


adain = adaptive_instance_normalization