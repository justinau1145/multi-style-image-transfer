import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Project imports
from encoder import VGGEncoder
from decoder import Decoder
from losses import total_loss
from utils import ImageDataset, get_train_transform, save_image

# Configurations
content_dir = './data/content'
style_dir = './data/style'
save_dir = './output'
epochs = 5
batch_size = 32
lr = 1e-4
image_size = 512
num_workers = 0
content_weight = 1.0
style_weight = 10.0
checkpoint_interval = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Models
encoder = VGGEncoder().to(device)
decoder = Decoder().to(device)
encoder.eval()

# Optimizer
optimizer = optim.Adam(decoder.parameters(), lr=lr, weight_decay=5e-5)

# Datasets
content_transform = get_train_transform(image_size)
style_transform = get_train_transform(image_size)

content_dataset = ImageDataset(content_dir, transform=content_transform)
style_dataset = ImageDataset(style_dir, transform=style_transform)

content_loader = DataLoader(content_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
style_loader = DataLoader(style_dataset, batch_size=batch_size, 
                         shuffle=True, num_workers=num_workers)

# Create output directory
os.makedirs(save_dir, exist_ok=True)

for epoch in range(epochs):
    decoder.train()
    epoch_loss = 0
    epoch_content_loss = 0
    epoch_style_loss = 0
    
    # Create iterators
    style_iter = iter(style_loader)
    
    print(f"\nEpoch {epoch+1}/{epochs}")
    
    for i, content_imgs in enumerate(content_loader):
        # Get style images
        try:
            style_imgs = next(style_iter)
        except StopIteration:
            style_iter = iter(style_loader)
            style_imgs = next(style_iter)
        
        # Move to device
        content_imgs = content_imgs.to(device)
        style_imgs = style_imgs.to(device)
        
        # Forward pass
        loss, loss_dict = total_loss(
            encoder, decoder, content_imgs, style_imgs,
            content_weight=content_weight,
            style_weight=style_weight
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track losses
        epoch_loss += loss_dict['total_loss']
        epoch_content_loss += loss_dict['content_loss']
        epoch_style_loss += loss_dict['style_loss']
        
        # Print progress
        if (i + 1) % 50 == 0:
            print(f"  Batch [{i+1}/{len(content_loader)}] - "
                  f"Loss: {loss_dict['total_loss']:.4f}, "
                  f"Content: {loss_dict['content_loss']:.4f}, "
                  f"Style: {loss_dict['style_loss']:.4f}")
    
    # Print epoch statistics
    avg_loss = epoch_loss / len(content_loader)
    avg_content = epoch_content_loss / len(content_loader)
    avg_style = epoch_style_loss / len(content_loader)
    
    print(f"\n  Epoch {epoch+1} Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Average Content Loss: {avg_content:.4f}")
    print(f"  Average Style Loss: {avg_style:.4f}")
    
    # Save epoch checkpoint
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = os.path.join(save_dir, f"decoder_epoch_{epoch+1}.pth")
        torch.save(decoder.state_dict(), checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

# Save final model
final_path = os.path.join(save_dir, "decoder_final.pth")
torch.save(decoder.state_dict(), final_path)