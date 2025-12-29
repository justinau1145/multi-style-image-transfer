# Segmented Style Transfer

This project implements region aware style transfer by combining Segment Anything Model (SAM) with Adaptive Instance Normalization (AdaIN) based on the paper [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868). While standard image style transfer applies 1 artistic style to the entire image, this allows you to apply different artistic styles to different regions if the image. 

## Features

- **Regional Style Transfer** - Use SAM to segment image and apply different artistic styles to different parts of the image
- **Intensity Control** - Adjust the style strength using the ***alpha*** parameter
- **Training Pipeline** - Train decoder on custom content and style datasets

## Results

## Requirements

- torch
- torchvision
- numpy
- Pillow
- opencv
- segment-anything

## Usage

Clone the repository

```bash
git clone https://github.com/justinau1145/segmented-style-transfer.git
cd segmented-style-transfer
pip install -r requirements.txt
```

### Training

Download the data to the ./data/ folder. The content images come from the [COCO](https://cocodataset.org/#download) dataset and the style images come [Wikiart](https://www.kaggle.com/c/painter-by-numbers) dataset. Run the script train.py. 

```bash
python -m scripts.train \
    --content_dir ./data/content \
    --style_dir ./data/style \
    --save_dir ./checkpoints \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-4 \
    --image_size 512 \
    --num_workers 0 \
    --content_weight 1.0 \
    --style_weight 10.0 \
    --checkpoint_interval 2 \
    --device cuda
```

Alternatively, you can download the pre-trained decoder weights trained using Google Colab under the [decoder_final.pth](https://github.com/justinau1145/segmented-style-transfer/releases/latest) file.

### Testing

Download the SAM checkpoint [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth). Run the script test.py.

```bash
python -m scripts.test \
    --content ./data/content/photo.jpg \
    --styles ./data/style/style1.jpg ./data/style/style2.jpg \
    --decoder ./decoder_final.pth \
    --sam_checkpoint ./sam_vit_h_4b8939.pth \
    --output ./output/result.jpg \
    --alpha 1.0 \
    --size 512 \
    --sam_model vit_h \
    --device cuda \
    --visualize_masks
```

## References

- [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)
- [Segment Anything](https://arxiv.org/abs/2304.02643)

## Acknowledgments

- AdaIN implementation based on [Xun Huang and Serge Belongie's paper](https://arxiv.org/abs/1703.06868)
- Segmentation powered by [Meta's SAM](https://github.com/facebookresearch/segment-anything)
- VGG-19 encoder from PyTorch's pretrained models
