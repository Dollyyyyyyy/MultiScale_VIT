# Multi-Scale Vision Transformer (MSViT)

This repository contains the implementation of the Multi-Scale Vision Transformer (MSViT), a novel architecture that combines Vision Transformers (ViTs) with Convolutional Neural Networks (CNNs) and Spatial Pyramid Pooling (SPP) to enhance generalizability and scalability for computer vision tasks. This project specifically focuses on addressing ViTs' challenges with handling images of varying resolutions and scales.

## Overview

The MSViT architecture improves upon traditional ViTs by:
- Introducing **dynamic patch sizes** for better adaptability to multi-scale inputs.
- Incorporating **CNNs for local feature extraction**.
- Using **SPP for robust multi-scale feature representation**.

The proposed model demonstrates significant improvements in accuracy, scalability, and computational efficiency compared to standard ViTs.

## Getting Started

### Prerequisites

Ensure you have Python 3.7+ installed along with the following packages:
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- TensorBoard

Install dependencies using:
pip install -r requirements.txt

##### Dataset Preparation
The project uses the MNIST dataset. Ensure the dataset files are placed under the data/MNIST/raw/ directory. Preprocessed files are already included in this repository.

##### Training the Model
To train the MSViT model, run:
python train.py
You can modify training parameters like learning rate, batch size, and number of epochs in the script.

##### Testing the Model
To test the model on multi-scale images:
python test.py

##### Visualizing Attention Maps
To visualize the attention maps for specific layers and heads, use:
python attention_map.py

Ensure to update the image_path in the script to point to the image you want to analyze.

## Results
Training Curves
training_curve.png and training_curve_SPP.png show the training and validation losses for the vanilla ViT and MSViT models, respectively.
Performance Comparison
accuracy_vs_scale.png illustrates how MSViT outperforms vanilla ViT on multi-scale test sets.
Attention Visualization
The script attention_map.py provides a visual representation of the attention mechanism, highlighting MSViT's improvements in spatial relationships.

## Paper
Refer to the Multiscale_ViT_Paper.pdf for detailed methodology, results, and discussions.

## Authors
This project was conducted by:

Jinzhi Yang
Sophie Li
Albon Wu
Jiarui Wan
Shiyu Fu
Hanzhi Bian

## Acknowledgments
Inspired by "Vision Transformers" by Dosovitskiy et al. (2021).
Spatial Pyramid Pooling by Kaiming He et al. (2015).
