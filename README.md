# Segmented Style Transfer

This project implements region aware style transfer by combining Segment Anything Model (SAM) with Adaptive Instance Normalization (AdaIN) based on the paper [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868). While standard image style transfer applies 1 artistic style to the entire image, this allows you to apply different artistic styles to different regions if the image. 

## Features

- **Regional Style Transfer** - Use SAM to segment image and apply different artistic styles to different parts of the image
- **Intensity Control** - Adjust the style strength using the *alpha* parameter
- **Training Pipeline** - Train own decoder on custom datasets

## Results

## Setup
