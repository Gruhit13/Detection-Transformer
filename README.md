# Detection-Transformer
A transformer-based object detection model. An implementation of paper End-to-End object detection using a transformer.

Transformers have gained popularity due to their attention mechanism which allows the model to focus their interest on a specific
part of the sentence in order to achieve the desired task. In this paper [End-to-End Object Detection using Transformer](https://arxiv.org/abs/2005.12872)
the author proposed a novel approach of using a transformer for detecting objects in an image. 

Hence this repo demonstrates a simple implementation of the same paper by taking reference from [DETR Implementation](https://github.com/facebookresearch/detr).

I used udacity self driving car dataset available on kaggle for training the model. And I prposed to detech maximum of 5 objects
per image. The default configuration of the model are displayed in config.py file. Despite of having an acceptable number of parameters the model
performed quite well on the dataset. 

## Result
![Result](https://github.com/Gruhit13/Detection-Transformer/blob/main/result/Result.jpg)
