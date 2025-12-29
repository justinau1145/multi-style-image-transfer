# Segmented Style Transfer

This project implements region aware style transfer by combining Segment Anything Model (SAM) with Adaptive Instance Normalization (AdaIN) based on the paper [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868). While standard image style transfer applies 1 artistic style to the entire image, this allows you to apply different artistic styles to different regions if the image. 

## Features

- **Regional Style Transfer** - Use SAM to segment image and apply different artistic styles to different parts of the image
- **Intensity Control** - Adjust the style strength using the *alpha* parameter
- **Training Pipeline** - Train own decoder on custom datasets

## Results

## Requirements

- torch
- torchvision
- numpy
- Pillow
- segment-anything
- open-cv

## Usage

### Training

Download the data to the ./data/ folder. The content images come from the COCO dataset and the style images come Wikiart dataset. Run the script train.py. 

Alternatively, you can download the pre-trained decoder weights under the decoder_final_pth file. 

### Testing

Download the SAM checkpoint [vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth). Run the script test.py.
