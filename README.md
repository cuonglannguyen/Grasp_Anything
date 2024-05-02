# Grasp-Anything
This is the repository of the paper "Grasp-Anything++ Assignment"
## Table of contents
   1. [Installation](#installation)
   1. [Datasets](#datasets)
   1. [Training](#training)
   1. [Testing](#testing)

## Installation
- Create a virtual environment
```bash
$ conda create -n granything python=3.9
$ conda activate granything
```

- Install pytorch
```bash
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
$ pip install -r requirements.txt
```

## Datasets
- Grasp-Anything++
- For other datasets, please obtain following their instructions: [Cornell](https://www.kaggle.com/datasets/oneoneliu/cornell-grasp), [Jacquard](https://jacquard.liris.cnrs.fr/), [OCID-grasp](https://github.com/stefan-ainetter/grasp_det_seg_cnn), and [VMRD](https://gr.xjtu.edu.cn/zh/web/zeuslan/dataset).
- All datasets should be include in the following hierarchy:
```
|- data/
    |- grasp-anything++
```

## Training
We use GR-ConvNet with CLIP embedding as our default deep network. To train GR-ConvNet on different datasets, you can use the following command:
```bash
$ python train_network.py --dataset <dataset> --dataset-path <dataset> --description <your_description> --use-depth 0
```
For example, if you want to train a GR-ConvNet on Grasp-Anything++, use the following command:
```bash
$ python train_network.py --dataset grasp-anything++ --dataset-path data/grasp-anything++ --description training_cornell --use-depth 0
```

## Testing
For testing procedure, we can apply the similar commands to test different baselines on different datasets:
```bash
python evaluate.py --network <path_to_pretrained_network> --dataset <dataset> --dataset-path data/<dataset> --iou-eval
```
Important note: `<path_to_pretrained_network>` is the path to the pretrained model obtained by training procedure. Usually, the pretrained models obtained by training are stored at `logs/<timstamp>_<training_description>`. You can select the desired pretrained model to evaluate. We do not have to specify neural architecture as the codebase will automatically detect the neural architecture.


## Acknowledgement
Our codebase is developed based on [Kumra et al.](https://github.com/skumra/robotic-grasping).
