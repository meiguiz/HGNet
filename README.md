# HGNet

This repository contains the implementation of the GHNet from the paper:

#### HGNet: GIoU Loss Optimization and Hierarchical RGB-D Feature Fusion for Generative Robotic Grasping

Robot grasping experiment video: [video](https://youtu.be/uCN0iav4FqQ)

## Requirements

- numpy
- opencv-python
- matplotlib
- scikit-image
- imageio
- torch
- torchvision
- torchsummary
- tensorboardX
- pyrealsense2
- Pillow

## Datasets

This repository supports both the [Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp) and
[Jacquard Dataset](https://jacquard.liris.cnrs.fr/).

#### Cornell Grasping Dataset

1. Download the and extract [Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp). 
2. Convert the PCD files to depth images by running `python -m utils.dataset_processing.generate_cornell_depth <Path To Dataset>`

#### Jacquard Dataset

1. Download and extract the [Jacquard Dataset](https://jacquard.liris.cnrs.fr/).


## Model Training

A model can be trained using the `train_network.py` script.

Example for Cornell dataset:

```bash
python train.py --dataset cornell --dataset-path <Path To Dataset> --description training_cornell
```

Example for Jacquard dataset:

```bash
python train.py --dataset jacquard --dataset-path <Path To Dataset> --description training_jacquard
```

## Model Evaluation

The trained network can be evaluated using the `evaluate.py` script.

Example for Cornell dataset:

```bash
python evaluate.py --network <Path to Trained Network> --dataset cornell --dataset-path <Path to Dataset> --iou-eval
```

## Grasp Visualization
```bash
python evaluate.py --network <Path to Trained Network> --dataset cornell --dataset-path <Path to Dataset> --iou-eval --vis
```
