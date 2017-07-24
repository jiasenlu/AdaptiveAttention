# AdaptiveAttention
Implementation of "[Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning](https://arxiv.org/pdf/1612.01887.pdf)"

![teaser results](https://raw.github.com/jiasenlu/AdaptiveAttention/master/demo/fig1.png)

### Requirements

To train the model require GPU with 12GB Memory, if you do not have GPU, you can directly use the pretrained model for inference. 

This code is written in Lua and requires [Torch](http://torch.ch/). The preprocssinng code is in Python, and you need to install [NLTK](http://www.nltk.org/) if you want to use NLTK to tokenize the caption.

You also need to install the following package in order to sucessfully run the code.

- [cudnn.torch](https://github.com/soumith/cudnn.torch)
- [torch-hdf5](https://github.com/deepmind/torch-hdf5)
- [lua-cjson](http://www.kyne.com.au/~mark/software/lua-cjson.php)
- [iTorch](https://github.com/facebook/iTorch)

##### Pretrained Model
The pre-trained model for COCO can be download [here](https://filebox.ece.vt.edu/~jiasenlu/codeRelease/AdaptiveAttention/model/COCO/).
The pre-trained model for Flickr30K can be download [here](https://filebox.ece.vt.edu/~jiasenlu/codeRelease/AdaptiveAttention/model/Flickr30k/). 

##### Vocabulary File
Download the corresponding Vocabulary file for [COCO](https://filebox.ece.vt.edu/~jiasenlu/codeRelease/AdaptiveAttention/data/COCO/) and [Flickr30k](https://filebox.ece.vt.edu/~jiasenlu/codeRelease/AdaptiveAttention/data/Flickr30k/) 

##### Download Dataset
The first thing you need to do is to download the data and do some preprocessing. Head over to the `data/` folder and run the correspodning ipython script. It will download, preprocess and generate coco_raw.json. 

Download [COCO](http://mscoco.org/) and Flickr30k image dataset, extract the image and put under somewhere. 
 

### training a new model on MS COCO
First train the Language model without finetune the image. 
```
th train.lua -batch_size 20 
```
When finetune the CNN, load the saved model and train for another 15~20 epoch. 
```
th train.lua -batch_size 16 -startEpoch 21 -start_from 'model_id1_20.t7'
```


### More Result about spatial attention and visual sentinel

![teaser results](https://raw.github.com/jiasenlu/AdaptiveAttention/master/demo/fig2.png)

![teaser results](https://raw.github.com/jiasenlu/AdaptiveAttention/master/demo/fig3.png)

For more visualization result, you can visit [here](https://filebox.ece.vt.edu/~jiasenlu/demo/caption_atten/demo.html)
(it will load more than 1000 image and their result...)

### Reference
If you use this code as part of any published research, please acknowledge the following paper
```
@misc{Lu2017Adaptive,
author = {Lu, Jiasen and Xiong, Caiming and Parikh, Devi and Socher, Richard},
title = {Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning},
journal = {CVPR},
year = {2017}
}
```

### Acknowledgement 

This code is developed based on [NeuralTalk2](https://github.com/karpathy/neuraltalk2). 

Thanks [Torch](http://torch.ch/) team and Facebook [ResNet](https://github.com/facebook/fb.resnet.torch) implementation. 

### License

BSD 3-Clause License



