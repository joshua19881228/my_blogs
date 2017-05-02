**TITLE**: Mask R-CNN

**AUTHOR**: Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick

**ASSOCIATION**: Facebook AI Research

**FROM**: [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)

## CONTRIBUTIONS ##

1. A conceptually simple, flexible, and general framework for object instance segmentation is presented.
2. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. 

## METHOD ##

Mask R-CNN is conceptually simple: Faster R-CNN has two outputs for each candidate object, a class label and a bounding-box offset; to this a third branch is added that outputs the object mask. 
