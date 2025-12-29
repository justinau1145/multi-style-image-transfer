import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Project imports
from models.encoder import VGGEncoder
from models.decoder import Decoder
from core.losses import total_loss
from utils.utils import ImageDataset, get_train_transform, save_image

def parse_args():
    parser = argparse.ArgumentParser(description='Train AdaIN style transfer decoder')
    
    parser.add_argument('--content_dir', type=str, default='./data/content',
                        help='Directory containing content images')
    parser.add_argument('--style_dir', type=str, default='./data/style',
                        help='Directory containing style images')
    parser.add_argument('--save_dir', type=str, default='./output',
                        help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Size to resize images to')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='Weight for content loss')
    parser.add_argument('--style_weight', type=float, default=10.0,
                        help='Weight for style loss')
    parser.add_argument('--checkpoint_interval', type=int, default=2,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Models
    encoder = VGGEncoder().to(device)
    decoder = Decoder().to(device)
    encoder.eval()

    # Optimizer
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=5e-5)

    # Datasets
    content_transform = get_train_transform(args.image_size)
    style_transform = get_train_transform(args.image_size)

    content_dataset = ImageDataset(args.content_dir, transform=content_transform)
    style_dataset = ImageDataset(args.style_dir, transform=style_transform)

    content_loader = DataLoader(content_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=args.num_workers)
    style_loader = DataLoader(style_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers)

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        decoder.train()
        epoch_loss = 0
        epoch_content_loss = 0
        epoch_style_loss = 0
        
        # Create iterators
        style_iter = iter(style_loader)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
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
                content_weight=args.content_weight,
                style_weight=args.style_weight
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
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f"decoder_epoch_{epoch+1}.pth")
            torch.save(decoder.state_dict(), checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_path = os.path.join(args.save_dir, "decoder_final.pth")
    torch.save(decoder.state_dict(), final_path)


if __name__ == '__main__':
    main()
