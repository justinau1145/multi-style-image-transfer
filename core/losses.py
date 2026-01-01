import torch
import torch.nn.functional as F
from adain_segment import calc_mean_std
import torch.nn as nn

def content_loss(output_features: torch.Tensor, 
                 target_features: torch.Tensor) -> torch.Tensor:
    """Compute the content loss.

    This loss measures the discrepancy between the output feature
    representations and the target content features using mean
    squared error (MSE).

    Args:
        output_features: Feature map from the generated image
        target_features: Feature map from the content image.

    Returns:
        torch.Tensor: Scalar tensor representing the content loss.
    """
    return F.mse_loss(output_features, target_features)


def style_loss(output_features: torch.Tensor, 
               style_features: torch.Tensor) -> torch.Tensor:
    """Compute the style loss using channel-wise statistics.

    The style loss is defined as the sum of mean squared errors
    between the channel-wise means and standard deviations of
    the output and style feature maps. 

    Args:
        output_features: Feature map from the generated image.
        style_features: Feature map from the style image.

    Returns:
        torch.Tensor: Scalar tensor representing the style loss.
    """
    output_mean, output_std = calc_mean_std(output_features)
    style_mean, style_std = calc_mean_std(style_features)
    
    loss = F.mse_loss(output_mean, style_mean) + F.mse_loss(output_std, style_std)
    return loss

def total_loss(
    encoder: nn.Module, decoder: nn.Module, content_img: torch.Tensor, 
    style_img: torch.Tensor, content_weight: float = 1.0, 
    style_weight: float = 10.0) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute total loss, content loss, and style loss for training.
    
    Args:
        encoder: VGG encoder.
        decoder: The decoder network being trained.
        content_img: Input content batch.
        style_img: Input style batch.
        content_weight: to 1.0.
        style_weight: Defaults to 10.0.

    Returns:
        Tuple containing:
            - total_loss: The scalar tensor to backpropagate.
            - loss_dict: A dictionary of float values for 
                logging/monitoring.
    """
    from adain import adaptive_instance_normalization
    
    # Encode content and style
    content_feat = encoder(content_img)
    style_feat = encoder(style_img)
    
    # Apply AdaIN
    t = adaptive_instance_normalization(content_feat, style_feat)
    
    # Decode
    output = decoder(t)
    
    # Get features for loss computation
    output_feat = encoder(output, output_last_only=False)
    content_feat_layers = encoder(content_img, output_last_only=False)
    style_feat_layers = encoder(style_img, output_last_only=False)
    
    # Content loss (only on relu4_1)
    loss_c = content_loss(output_feat[-1], t)
    
    # Style loss (on relu1_1, relu2_1, relu3_1, relu4_1)
    loss_s = 0
    for output_f, style_f in zip(output_feat, style_feat_layers):
        loss_s += style_loss(output_f, style_f)
    
    # Total loss
    loss = content_weight * loss_c + style_weight * loss_s
    
    loss_dict = {
        'content_loss': loss_c.item(),
        'style_loss': loss_s.item(),
        'total_loss': loss.item()
    }
    
    return loss, loss_dict
