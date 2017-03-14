# AdaptiveAttention
Implementation of "[Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning](https://arxiv.org/pdf/1612.01887.pdf)"

### Requirements

To train the model require GPU with 12GB Memory, if you do not have GPU, you can directly use the pretrained model for inference. 

This code is written in Lua and requires [Torch](http://torch.ch/). The preprocssinng code is in Python, and you need to install [NLTK](http://www.nltk.org/) if you want to use NLTK to tokenize the question.

You also need to install the following package in order to sucessfully run the code.

- [cudnn.torch](https://github.com/soumith/cudnn.torch)
- [torch-hdf5](https://github.com/deepmind/torch-hdf5)
- [lua-cjson](http://www.kyne.com.au/~mark/software/lua-cjson.php)
- [iTorch](https://github.com/facebook/iTorch)

### Training

##### Download Dataset
The first thing you need to do is to download the data and do some preprocessing. Head over to the `data/` folder and run the correspodning ipython script. It will download, preprocess and generate coco_raw.json. 

Download [COCO](http://mscoco.org/) and Flickr30k image dataset, extract the image and put under somewhere. 
  
### Pretrained Model


### Evaluation



### Reference

If you use this code as part of any published research, please acknowledge the following paper

@misc{Lu2017Adaptive,
author = {Lu, Jiasen and Xiong, Caiming and Parikh, Devi and Richard, Socher},
title = {Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning},
journal = {CVPR},
year = {2017}
}

### License
[License](https://github.com/jiasenlu/AdaptiveAttention/edit/master/LICENCE.md)

### Acknowledgement 

Thanks Andrew Karpathy and his [NeuralTalk2](https://github.com/karpathy/neuraltalk2). 

Thanks [Torch](http://torch.ch/) team and Facebook [ResNet](https://github.com/facebook/fb.resnet.torch) implementation. 
