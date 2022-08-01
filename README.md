# CT<sup>2</sup>: Colorization Transformer via Color Tokens (official)

## Introduction
This is the author's official PyTorch <b>CT<sup>2</sup></b> implementation.

We present <b>C</b>olorization <b>T</b>ransformer via <b>C</b>olor <b>T</b>okens (<b>CT<sup>2</sup></b>) to colorize grayish images while dealing with incorrect semantic colors and undersaturation without any additional external priors.

<!-- ![test image size](https://github.com/shuchenweng/CT2/blob/main/application.png){:height="100%" width="100%"} -->
 <img src="https://github.com/shuchenweng/CT2/blob/main/application.png" align=center />
 

## Prerequisites
* Python 3.6
* PyTorch 1.10
* NVIDIA GPU + CUDA cuDNN

## Installation
Clone this repo: 
```
git clone https://github.com/shuchenweng/CT2.git
```
Install PyTorch and dependencies
```
http://pytorch.org
```
Install other python requirements
```
pip install -r requirement.txt
```
Download the pretrained vit model and move it to *segm/resources/vit_large_patch16_384.npz*
```
https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz
```


## Datasets
We use [ImageNet](https://www.image-net.org/) as our dataset, which includes 1.3M training data and covers 1000 categories. We test on the first 5k images of the public validation set, which is consistent with the previous [Colorization transformer](https://iclr.cc/virtual/2021/poster/2844). All the test images are center cropped and resized into 256 × 256 resolution.

### 1) Training
A training script example is below:
```
python -m torch.distributed.launch --nproc_per_node=8 -m segm.train --log-dir segm/vit-large --batch-size 48 --local_rank 0  --partial_finetune False --backbone vit_large_patch16_384 --color_position True --add_l1_loss True --l1_conv True --l1_weight 10 --amp
```

### 2) Testing
To test your training weights, you could excute the script below:
```
python -m torch.distributed.launch --nproc_per_node=1 -m segm.test --log-dir segm/vit-large --local_rank 0 --only_test True
```
We also publish the [pretrained weights](https://pan.baidu.com/s/1cak_aAHIaMTVpTLP0yqRyw) here. Download it and move it to *segm/vit-large* to enjoy the colorization!

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

Except where otherwise noted, this content is published under a [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.

## Citation
If you use this code for your research, please cite our papers [CT<sup>2</sup>: Colorization Transformer via Color Tokens](https://ci.idm.pku.edu.cn/Weng_ECCV22b.pdf)
```
@InProceedings{UniCoRN,
  author = {Weng, Shuchen and Sun, Jimeng and Li, Yu and Li, Si and Shi, Boxin},
  title = {CT2: Colorization Transformer via Color Tokens},
  booktitle = {{ECCV}},
  year = {2022}
}
```
