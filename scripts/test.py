import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Project imports
from encoder import VGGEncoder
from decoder import Decoder
from adain_segment import multi_adain, multi_adain_alpha
from sam_segmentation import SAMSegmenter, load_image_for_sam
from utils import load_image, save_image

# Configuration
content_path = './data/content/000000000013.jpg'
style_paths = [
    './data/style/221.jpg',
    './data/style/van_gogh.jpg',
]
decoder_path = './decoder_final.pth'
sam_checkpoint = 'sam_vit_h_4b8939.pth'
output_path = './output_multi_style.jpg'
alpha = 1.0
size = 512
sam_model = 'vit_h'
device_name = 'cuda'
visualize_masks = True
mask_indices = None

device = torch.device(device_name if torch.cuda.is_available() else 'cpu')

# Models
encoder = VGGEncoder().to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(decoder_path, map_location=device))

encoder.eval()
decoder.eval()

# SAM
sam_segmenter = SAMSegmenter(model_type=sam_model, 
                             checkpoint_path=sam_checkpoint, device=device_name)

# Load and segment content image
image_np = load_image_for_sam(content_path)
num_regions = len(style_paths)
masks = sam_segmenter.generate_masks(image_np, num_masks=num_regions)

if visualize_masks:
    overlay = sam_segmenter.visualize_masks(image_np, masks)
    mask_vis_path = output_path.replace('.jpg', '_masks.jpg')
    Image.fromarray(overlay).save(mask_vis_path)

# Auto assign mask indices
if mask_indices is None:
    mask_indices = list(range(len(style_paths)))

content_img = load_image(content_path, size=size).to(device)

with torch.no_grad():
    content_feat = encoder(content_img)

_, _, feat_h, feat_w = content_feat.shape

style_feats = []
for style_path in style_paths:
    style_img = load_image(style_path, size=size).to(device)
    with torch.no_grad():
        style_feat = encoder(style_img)
    style_feats.append(style_feat)

mask_tensors = []
for i, mask_idx in enumerate(mask_indices):
    if mask_idx >= len(masks):
        continue
    
    mask = masks[mask_idx]
    mask_tensor = sam_segmenter.create_mask_tensor(mask, (size, size)).to(device)
    
    mask_feat = F.interpolate(mask_tensor, size=(feat_h, feat_w),
                              mode='bilinear',align_corners=False)
    
    mask_tensors.append(mask_feat)

if alpha < 1.0:
    output_feat = multi_adain_alpha(
        content_feat, 
        style_feats[:len(mask_tensors)], 
        mask_tensors, 
        alpha, 
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

save_image(output, output_path)
print(f"Output saved to: {output_path}")