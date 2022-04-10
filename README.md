# Pose Baseline 3D PyTorch

A PyTorch reimplementation of the paper - 3D human pose estimation.

Julieta Martinez, Rayat Hossain, Javier Romero, James J. Little. A simple yet effective baseline for 3d human pose estimation. In ICCV, 2017.
PDF: https://arxiv.org/pdf/1705.03098.pdf

You can check original Tensorflow implementation written by [Julieta Martinez et al.](https://github.com/una-dinosauria/3d-pose-baseline)

This repo reproduces the work by [@weigq](https://github.com/weigq/3d_pose_baseline_pytorch).

## Dependencies
* [PyTorch](http://pytorch.org/) >= 1.0.0

## Datasets
Human3.6m

## Installation
```
git clone https://github.com/jaroslaw1007/Pose_Baseline_3D_PyTorch.git
```

## Usage

### Train

Train on Human3.6m groundtruth 2d joints.

```
python main.py --training --max_norm
```

### Test
```
python main.py --training
```

![](https://i.imgur.com/5rxl05I.png)


|  | direct. | discuss. | eat. | greet. | phone | photo | pose | purch. | sit | sitd. | somke | wait | walkd. | walk | walkT | avg |
| :--: | :--: | :--: | :--: | :--: |  :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| [Julieta Martinez et al.](https://github.com/una-dinosauria/3d-pose-baseline) | 37.7 | 44.4 | 40.3 | 42.1 | 48.2 | 54.9 | 44.4 | 42.1 | 54.6 | 58.0 | 45.1 | 46.4 | 47.6 | 36.4 | 40.4 | 45.5|
| [weigq](https://github.com/weigq/3d_pose_baseline_pytorch) | 35.7 | 42.3 | 39.7 | 40.7 | 44.5 | 53.3 | 42.8 | 40.1 | 52.5 | 53.9 | 42.8 | 43.1 | 44.1 | 33.4 | 36.3 | 43.0 |
| This version | 35.5 | 41.7 | 39.0 | 40.4 | 44.4 | 52.4 | 42.7 | 38.2 | 53.6 | 54.6 | 42.6 | 42.8 | 44.1 | 33.9 | 36.9 | 42.8 |


## Citing

```
@inproceedings{martinez_2017_3dbaseline,
  title={A simple yet effective baseline for 3d human pose estimation},
  author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.},
  booktitle={ICCV},
  year={2017}
}
```
