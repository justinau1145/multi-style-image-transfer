import torch
import numpy as np
import argparse
import os

# Project imports
from models.encoder import VGGEncoder
from models.decoder import Decoder
from models.sam_segmentation import SAMSegmenter, load_image_for_sam
from core.adain_segment import regional_adain
from utils.utils import load_image, denormalize, save_image

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Style Transfer with SAM Segmentation')
    
    # Paths
    parser.add_argument('--content_path', type=str, required=True,
                        help='Path to the content image')
    parser.add_argument('--style_paths', type=str, nargs='+', required=True,
                        help='List of paths to style images (e.g., style1.jpg style2.jpg)')
    parser.add_argument('--decoder_path', type=str, default='./decoder_weights.pth',
                        help='Path to the trained AdaIN decoder checkpoint')
    parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_h_4b8939.pth',
                        help='Path to SAM checkpoint file')
    parser.add_argument('--alphas', type=float, nargs='+', default=[1.0],
                        help='Style intensity for each style image (0.0 to 1.0)')
    parser.add_argument('--output_name', type=str, default='output_stylized',
                        help='Base name for the output image')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    alphas = args.alphas
    while len(alphas) < len(args.style_paths):
        alphas.append(alphas[-1])
    
    styles_config = [
        {'path': p, 'alpha': a} for p, a in zip(args.style_paths, alphas)
    ]

    # Models
    encoder = VGGEncoder().to(device).eval()
    decoder = Decoder().to(device).eval()
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
    segmenter = SAMSegmenter(model_type="vit_h", 
                             checkpoint_path=args.sam_checkpoint, 
                             device=str(device))

    content_img_np = load_image_for_sam(args.content_path)
    content_tensor = load_image(args.content_path).to(device)

    object_masks = segmenter.generate_masks(content_img_np, 
                                            num_masks=len(styles_config) - 1)
    bg_mask = segmenter.get_complement_mask(object_masks, content_img_np.shape)
    
    all_segments = object_masks + [bg_mask]

    with torch.no_grad():
        content_feat = encoder(content_tensor)
        combined_feat = torch.zeros_like(content_feat)

        for i, mask_dict in enumerate(all_segments):
            config = styles_config[i]
            style_tensor = load_image(config['path']).to(device)
            style_feat = encoder(style_tensor)
            
            mask_tensor = segmenter.create_mask_tensor(mask_dict).to(device)
            
            segment_feat = regional_adain(
                content_feat, style_feat, mask_tensor, alpha=config['alpha']
            )
            combined_feat += segment_feat

        output = decoder(combined_feat)

    output = denormalize(output, device)
    
    save_image(output, args.output_name)
    print(f"Output saved to: {args.output_name}")

if __name__ == '__main__':
    main()
