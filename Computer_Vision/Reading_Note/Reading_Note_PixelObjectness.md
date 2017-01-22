**TITLE**: Pixel Objectness

**AUTHOR**: Suyog Dutt Jain, Bo Xiong, Kristen Grauman

**ASSOCIATION**: The University of Texas at Austin

**FROM**: [arXiv:1701.05349](https://arxiv.org/abs/1701.05349)

## CONTRIBUTIONS ##

 An end-to-end learning framework for foreground object segmentation is proposed. Given a single novel image, a pixel-level mask is produced for all “object-like” regions even for object categories never seen during training. 

## METHOD ##

### Problem Formulation ###

Given an RGB image of size  $m \times n \times c$ as input, the problem is formulated as densely labeling each pixel in the images as eigher "object" or "background". The output is a binary map of size $m \times n$.

### Dataset ###

Two different datasets are used including 1) one dataset with *explicit* boundary-level annotations and 2) one dataset with *implicit* imagelevel object category annotations.

### Training ###

The network is first trained on a large scale object classification task, such as ImageNet 1000-category classification. This stage can be regarded as training on an *implicit* labeled dataset. Its image representation has a strong notion of objectness built inside it, even though it never observes any segmentation annotations. 

Then the network is trained on PASCAL 2012 segmentation dataset, which is an *explicit* labeled dataset. The 20 object labels are discarded, and mapped instead to the single generic "object-like" (foreground) label for training.
