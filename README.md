# CVPR2020 Memory aggregation networks for efficient interactive video object segmentation

This is the pytorch implementation of the CVPR2020 paper "Memory aggregation networks for efficient interactive video object segmentation".


![avatar](./teaser/1836-teaser.gif)
## Preparation
### Dependencies
 - Python 3.7
 - Pytorch 1.0
 - Numpy
 - tensorboardX
 - davisinteractive (Please refer to [this link](https://interactive.davischallenge.org/user_guide/installation/))
 
### Pretrained model
Download [deeplabV3+ model pretrained on COCO](https://drive.google.com/file/d/15temSaxnKmGPvNxrKPN6W2lSsyGfCtTB/view?usp=sharing) to this repo.

### Dataset
Download [DAVIS2017](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) and [scribbles](https://data.vision.ee.ethz.ch/csergi/share/DAVIS-Interactive/DAVIS-2017-scribbles-trainval.zip) into one folder. Please refer to [DAVIS](https://davischallenge.org/davis2017/code.html).

If you need the file "DAVIS2017/ImageSets/2017/v_a_l_instances.txt", please refer to the link https://drive.google.com/file/d/1aLPaQ_5lyAi3Lk3d2fOc_xewSrfcrQlc/view?usp=sharing

## Train and Test
```
sh run_local.sh
```
## Evaluation
You can download [our model](https://drive.google.com/file/d/1JjYNha40rtEYKKKFtDv06myvpxagl5dW/view?usp=sharing) for evaluation.


## Citation
Please cite this paper in your publications if it helps your research:
```
@inproceedings{miao2020memory,
  title={Memory aggregation networks for efficient interactive video object segmentation},
  author={Miao, Jiaxu and Wei, Yunchao and Yang, Yi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10366--10375},
  year={2020}
}
```
