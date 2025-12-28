import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

# ImageNet normalization parameters
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_transform(size: int = 256, crop: bool = False) -> transforms.Compose:
    """Creates the standard preprocessing pipeline for VGG networks.

    Constructs a transformation pipeline that resizes, crops, and 
    normalizes images to match the statistics expected by 
    ImageNet-pretrained models.

    Args:
        size: The spatial resolution for the output tensor.

    Returns:
        transforms.Compose: A composition of torchvision transforms.
    """
    transform_list = []
    
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    else:
        transform_list.append(transforms.Resize(size))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    return transforms.Compose(transform_list)


def get_train_transform(size: int = 512) -> transforms.Compose:
    """
    Get training transformation with random croppings

    Args:
        size: The spatial resolution for the output tensor.

    Returns:
        transforms.Compose: A composition of torchvision transforms.
    """
    return transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def load_image(image_path: str, size: int = 256, 
               crop: bool = False) -> torch.Tensor:
    """
    Load an image from disk and apply preprocessing.

    The image is loaded using PIL, converted to RGB format,
    and processed using the provided transform.

    Args:
        path: Path to the image file.
        transform: Preprocessing transform to apply.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = Image.open(image_path).convert('RGB')
    transform = get_transform(size, crop)
    return transform(image).unsqueeze(0)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverses ImageNet normalization for visualization.
    
    Args:
        tensor: Normalized image tensor

    Returns:
        torch.Tensor: Denormalized tensor.
    """
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    return tensor * std + mean


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Saves a tensor as an image file.

    Handles denormalization, clamping to valid pixel ranges, 
    and CPU transfer.

    Args:
        tensor: Image tensor, assumed to be normalized.
        path: Output file path.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize
    tensor = denormalize(tensor)
    
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    transform = transforms.ToPILImage()
    image = transform(tensor.cpu())
    
    # Create directory if needed
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', 
                exist_ok=True)

    image.save(path)


class ImageDataset(Dataset):
    """
    Dataset for loading images from a directory tree.

    Images are recursively discovered under the given root directory,
    filtered by common image extensions, and preprocessed using the
    provided transform. This dataset is used for both content and style 
    images.
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
        """
        Load and preprocess a single image.

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