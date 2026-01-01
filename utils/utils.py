import torch
from torchvision import transforms
from torchvision.utils import save_image as tv_save_image
from torch.utils.data import Dataset
from PIL import Image
import os

# ImageNet normalization parameters
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_image(image_path: str) -> torch.Tensor:
    """Load an image and apply normalization (no resizing).
    
    This matches the reference test.py implementation which processes
    images at their original size during inference.

    Args:
        image_path: Path to the image file.

    Returns:
        torch.Tensor: Normalized image tensor of shape [1, 3, H, W]
    """
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    return transform(image).unsqueeze(0)


def denormalize(tensor: torch.Tensor, device) -> torch.Tensor:
    """Reverses ImageNet normalization for visualization.
    
    Args:
        tensor: Normalized image tensor
        device: Device where tensor is located

    Returns:
        torch.Tensor: Denormalized tensor clamped to [0, 1]
    """
    std = torch.Tensor(STD).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor(MEAN).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Saves a tensor as an image file.

    Args:
        tensor: Image tensor to save
        path: Output file path
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', 
                exist_ok=True)
    
    # Use torchvision's save_image
    tv_save_image(tensor, path, nrow=1)


def get_train_transform(size: int = 512) -> transforms.Compose:
    """Get training transformation with random cropping.

    Args:
        size: The spatial resolution to resize to before cropping.

    Returns:
        transforms.Compose: A composition of torchvision transforms.
    """
    return transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


class ImageDataset(Dataset):
    """Dataset for loading images from a directory tree.

    Images are recursively discovered under the given root directory,
    filtered by common image extensions, and preprocessed using the
    provided transform.
    """
    def __init__(self, root_dir: str, transform) -> None:
        """
        Args:
            root_dir: Root directory containing images.
            transform: Image preprocessing transform.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Get all image files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    self.image_paths.append(os.path.join(root, file))
    
    def __len__(self) -> int:
        """Return the number of valid images found."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and preprocess a single image.

        Args:
            idx: Index of the image.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image
