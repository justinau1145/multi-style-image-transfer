import torch
import torch.nn.functional as F

def calc_mean_std(feat: torch.Tensor, 
                  eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute channel-wise mean and standard deviation of feature maps.
    
    Statistics are computed independently for each channel across the
    spatial dimensions (H, W), and reshaped for broadcasting.

    Args:
        feat: Feature tensor of shape [batch_size, c, h, w]
        eps: Small constant to avoid division by zero. Defaults to 1e-6.

    Returns:
        Tuple containing:
            - mean: Spatial mean of shape [batch_size, c, 1, 1]
            - std: Spatial standard deviation of shape [batch_size, c, 1, 1]
    """
    batch_size, c = feat.size()[:2]
    feat_mean = feat.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    feat_std = feat.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + eps
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: torch.Tensor, 
                                   style_feat: torch.Tensor) -> torch.Tensor:
    """Apply Adaptive Instance Normalization (AdaIN).

    AdaIN aligns the channel-wise mean and standard deviation of the
    content features with those of the style features, enabling
    style transfer while preserving content structure.

    Args:
        content_feat: Content feature map of shape [batch_size, c, h, w]
        style_feat: Style feature map of shape [batch_size, c, h, w]

    Returns:
        torch.Tensor: The stylized feature maps of shape [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_feat)
    style_mean, style_std = calc_mean_std(style_feat)
    
    normalized_features = style_std * (content_feat - content_mean) / content_std + style_mean
    return normalized_features

def regional_adain(content_feat: torch.Tensor, 
                   style_feat: torch.Tensor, 
                   mask: torch.Tensor, 
                   alpha: float = 1.0) -> torch.Tensor:
    """
    Applies AdaIN to a specific region with intensity control.
    
    Args:
        content_feat: Content feature map.
        style_feat: Style feature map.
        mask: Binary mask [1, 1, H, W].
        alpha: Style weight (0.0 = pure content, 1.0 = full style).
    """
    stylized_feat = adaptive_instance_normalization(content_feat, style_feat)
    controlled_feat = alpha * stylized_feat + (1.0 - alpha) * content_feat    
    mask_resized = F.interpolate(mask, size=content_feat.shape[2:], mode='nearest')
    return controlled_feat * mask_resized
