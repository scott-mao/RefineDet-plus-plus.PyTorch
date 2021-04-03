A higher performance [PyTorch](http://pytorch.org/) implementation of [RefineDet++: Single-Shot Refinement Neural Network for Object Detection](http://www.cbsr.ia.ac.cn/users/sfzhang/files/TCSVT_RefineDet++.pdf ).

The Alignment Convolution is implemented with Deformable Convolution and referred to the baseline of [RepPoints](https://arxiv.org/pdf/1904.11490.pdf). Alignment Convolutions for three anchors are implemented by three branches, the details of codes might be different with the original ones.

With accurately calculated offset rather than the learned, Alignment Convolution is a good solution for the feature misalignment problem among 1.5 stage object detection methods

### Table of Contents
- <a href='#Majorfeatures'>Major features</a>
- <a href='#performance'>Performance</a>
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-refinedet'>Train</a>
- <a href='#evaluate'>Evaluate</a>
- <a href='#todo'>Future Work</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Major features
* Alignment Convolution 

* Multi-scale test

* VGG backbone with bn layers

* ResNet and ResNeXt backbone

* 320, 512 and 1024 input size

* Ablation for model architecture

* PR curve for coco format dataset

* Original RefineDet model.

## Performance

#### VOC2007 Test

##### mAP 

| Arch | Paper | Our PyTorch Version |
|:-:|:-:|:-:|
| RefineDet512++ | 82.5% | 81.86% |
| RefineDet512++ ms | 84.2% | 84.00% |

ms: multi scale test, the results are obtained by training with pre-trained vgg.

#### [SSDD](https://github.com/HaoIrving/SSDD_coco.git) (remote ship detection dataset of Radar images)

##### COCO AP 

| Arch | Our PyTorch Version |
|:-:|:-:|
| RefineDet512++ | 62.94% | 

## Installation
```
./make.sh
pip install visdom
```
This code use [MMDetection](https://mmdetection.readthedocs.io/) as the basic environment, since we need the Deformable Convolution implementation of it, look into it for the installation method. [SOLO](https://github.com/WXinlong/SOLO.git) is also acceptable (based on an older version of MMDetection), other version is obscure.

## Datasets
### VOC Dataset
PASCAL VOC: Visual Object Classes

##### Download VOC2007 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```
##### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```
### COCO2017
recommend to [MMDetection](https://github.com/open-mmlab/mmdetection) for the downloading method.

## Training RefineDet++
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:              https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, this code assume you have downloaded the file in the `./weights` dir:

```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train RefineDet512++ with vggbn backbone (train from scratch).

```Shell
python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/voc_4e3_512vggbn/ --ngpu 4  --model 512_vggbn --batch_size 32 --dataset VOC -max 240
```
- To train RefineDet512++ with pre-trained vgg backbone.
```
python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/voc_4e3_512vgg/ --ngpu 4  --model 512_vggbn --batch_size 32 --dataset VOC -max 240 --pretrained
```

## Evaluate RefineDet++
To evaluate RefineDet512++ with vggbn backbone:

```Shell
CUDA_VISIBLE_DEVICES=3 python eval_refinedet_voc07.py --prefix weights/voc_4e3_512vggbn  --model 512_vggbn 
```
To evaluate RefineDet512++ with vgg backbone.
```
CUDA_VISIBLE_DEVICES=3 python eval_refinedet_voc07.py --prefix weights/voc_4e3_512vgg  --model 512_vggbn  -wobn
```
### Training and Evaluate RefineDet
add `-woalign` in the command.

## TODO
* [ ] Support for ResNet and ResNeXt backbone (implemented fully, but the training is hard to converge for slightly large learning rate, sting working on this.).
* [ ] Report performance on COCO2017 dataset.

## References
- Multi-scale test is modified from [Original RefineDet Implementation (CAFFE)](https://github.com/sfzhang15/RefineDet)
- This code is built based on [luuuyi/RefineDet.PyTorch](https://github.com/luuuyi/RefineDet.PyTorch), [PytorchSSD](https://github.com/lzx1413/PytorchSSD.git), [FaceBoxes.PyTorch](https://github.com/zisianw/FaceBoxes.PyTorch) and [MMDetection](https://github.com/open-mmlab/mmdetection), many thanks to them.
