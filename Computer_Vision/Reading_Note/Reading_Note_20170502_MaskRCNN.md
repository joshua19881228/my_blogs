**TITLE**: Mask R-CNN

**AUTHOR**: Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick

**ASSOCIATION**: Facebook AI Research

**FROM**: [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)

## CONTRIBUTIONS ##

1. A conceptually simple, flexible, and general framework for object instance segmentation is presented.
2. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. 

## METHOD ##

Mask R-CNN is conceptually simple: Faster R-CNN has two outputs for each candidate object, a class label and a bounding-box offset; to this a third branch is added that outputs the object mask. The idea is illustrated in the following image.

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/Reading_Note_20170502_MaskRCNN_1.png" alt="" width="640"/>

In order to avoid competition across classes, the mask branch has a $ Km^{2} $ dimensional output for each ROI, which endoces $ K $ binary masks of resolution $ m \times m $, one for each of the $ K $ classes. When training, for an ROI associated with ground-truth class $ k $, loss is only computed on the $k$-th mask.

An $ m \times m $ mask from each ROI is predicted using a small FCN network. The input of the small FCN network is an *RoIAlign* feature, which using bilinear interpolation to compute the exact values of the input feature at four regularly sampled locations in each ROI bin.

