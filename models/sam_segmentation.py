import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import cv2


class SAMSegmenter:
    """
    Wrapper for Segment Anything Model (SAM) 

    Provides automatic mask generation for sementatic segmentation.
    
    Attributes:
        device: Device for model inference.
        sam: SAM model.
        mask_generator: Automatic mask generator.
    """
    
    def __init__(self, model_type: str = "vit_h", 
                 checkpoint_path: str = "sam_vit_h_4b8939.pth",
                 device: str = "cuda") -> None:
        """
        Initialize SAM model for segmentation.
        
        Args:
            model_type: SAM model variant.
            checkpoint_path: Path to SAM checkpoint file.
            device: Device for inference.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(self.device)
        self.sam.eval()
        
        # Initialize mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam, points_per_side=32, pred_iou_thresh=0.86, 
            stability_score_thresh=0.92, crop_n_layers=1, 
            crop_n_points_downscale_factor=2, min_mask_region_area=100,
        )
    
    def generate_masks(self, image: np.ndarray, 
                       num_masks: int = None) -> list[dict]:
        """
        Generate semantic segmentation masks for an image.
        
        Args:
            image: RGB image as numpy array.
            num_masks: Maximum number of masks to return. 
                If None, returns all masks.
        
        Returns:
            List: List of mask dictionaries, each containing:
                - 'segmentation': Binary mask.
                - 'area': Mask area in pixels.
                - 'bbox': Bounding box.
                - 'predicted_iou': Quality score.
        """
        masks = self.mask_generator.generate(image)
        
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        if num_masks is not None:
            masks = masks[:num_masks]
        
        return masks
    
    def visualize_masks(self, image: np.ndarray, 
                       masks: list[dict], alpha: float = 0.5) -> np.ndarray:
        """
        Visualize segmentation masks overlaid on image.
        
        Args:
            image: RGB image.
            masks: List of mask dictionaries.
            alpha: Transparency for overlay (0-1).

        Returns:
            np.ndarray: Image with colored mask overlay
        """
        overlay = image.copy()
        
        for i, mask in enumerate(masks):
            color = np.random.randint(0, 255, 3).tolist()
            mask_area = mask['segmentation']
            overlay[mask_area] = (overlay[mask_area] * (1 - alpha) + 
                                  np.array(color) * alpha).astype(np.uint8)
        
        return overlay
    
    def create_mask_tensor(self, mask: dict, 
                          target_size: tuple[int, int]) -> torch.Tensor:
        """
        Convert SAM mask to PyTorch tensor with specified size.
        
        Args:
            mask: Mask dictionary from SAM.
            target_size: Target.
        
        Returns:
            torch.Tensor: Binary mask tensor.
        """
        segmentation = mask['segmentation'].astype(np.float32)
        
        # Resize if needed
        if segmentation.shape != target_size:
            segmentation = cv2.resize(segmentation, 
                                      (target_size[1], target_size[0]),
                                      interpolation=cv2.INTER_LINEAR)
        
        mask_tensor = torch.from_numpy(segmentation).unsqueeze(0).unsqueeze(0)
        
        return mask_tensor

def load_image_for_sam(image_path: str) -> np.ndarray:
    """
    Load image in format required by SAM (RGB numpy array).
    
    Args:
        image_path: Path to image file.
    
    Returns:
        np.ndarray: RGB image array.
    """
    image = Image.open(image_path).convert('RGB')
    return np.array(image)