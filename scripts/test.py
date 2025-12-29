import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argeparse

# Project imports
from models.encoder import VGGEncoder
from models.decoder import Decoder
from core.adain_segment import multi_adain, multi_adain_alpha
from models.sam_segmentation import SAMSegmenter, load_image_for_sam
from utils.utils import load_image, save_image

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-region style transfer using AdaIN and SAM')
    
    parser.add_argument('--content', type=str, required=True,
                        help='Path to content image')
    parser.add_argument('--styles', type=str, nargs='+', required=True,
                        help='Paths to style images')
    parser.add_argument('--decoder', type=str, default='./decoder_final.pth',
                        help='Path to decoder checkpoint')
    parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_h_4b8939.pth',
                        help='Path to SAM checkpoint')
    parser.add_argument('--output', type=str, default='./output_multi_style.jpg',
                        help='Path to save output image')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Style transfer intensity (0-1)')
    parser.add_argument('--size', type=int, default=512,
                        help='Output image size')
    parser.add_argument('--sam_model', type=str, default='vit_h',
                        help='SAM model type')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--visualize_masks', action='store_true',
                        help='Save visualization of segmentation masks')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Models
    encoder = VGGEncoder().to(device)
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(args.decoder, map_location=device))

    encoder.eval()
    decoder.eval()

    # SAM
    sam_segmenter = SAMSegmenter(model_type=args.sam_model, 
                                 checkpoint_path=args.sam_checkpoint, device=args.device)

    # Load and segment content image
    image_np = load_image_for_sam(args.content)
    num_regions = len(args.styles)
    masks = sam_segmenter.generate_masks(image_np, num_masks=num_regions)

    if args.visualize_masks:
        overlay = sam_segmenter.visualize_masks(image_np, masks)
        mask_vis_path = args.output.replace('.jpg', '_masks.jpg')
        Image.fromarray(overlay).save(mask_vis_path)

    content_img = load_image(args.content, size=args.size).to(device)

    with torch.no_grad():
        content_feat = encoder(content_img)

    _, _, feat_h, feat_w = content_feat.shape

    style_feats = []
    for style_path in args.styles:
        style_img = load_image(style_path, size=args.size).to(device)
        with torch.no_grad():
            style_feat = encoder(style_img)
        style_feats.append(style_feat)

    mask_tensors = []
    for i, mask in enumerate(masks[:len(args.styles)]):
        mask_tensor = sam_segmenter.create_mask_tensor(mask, (args.size, args.size)).to(device)
        
        mask_feat = F.interpolate(mask_tensor, size=(feat_h, feat_w),
                                  mode='bilinear',align_corners=False)
        
        mask_tensors.append(mask_feat)

    if args.alpha < 1.0:
        output_feat = multi_adain_alpha(
            content_feat, 
            style_feats[:len(mask_tensors)], 
            mask_tensors, 
            args.alpha, 
            device
        )
    else:
        output_feat = multi_adain(
            content_feat, 
            style_feats[:len(mask_tensors)], 
            mask_tensors, 
            device
        )

    with torch.no_grad():
        output = decoder(output_feat)

    save_image(output, args.output)
    print(f"Output saved to: {args.output}")


if __name__ == '__main__':
    main()
