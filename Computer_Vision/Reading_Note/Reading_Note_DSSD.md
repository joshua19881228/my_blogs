**TITLE**: DSSD: Deconvolutional Single Shot Detector

**AUTHER**: Cheng-Yang Fu, Wei Liu, Ananth Ranga, Ambrish Tyagi, Alexander C. Berg

**FROM**: [arXiv:1701.06659](https://arxiv.org/abs/1701.06659)

###CONTRIBUTIONS###

1. A combination of a state-of-the-art classifier (Residual-101) with a fast detection framework (SSD) is proposed.
2. Deconvolution layers are applied to introduce additional large-scale context in object detection and improve accuracy, especially for small objects.

###METHOD###

This is a successive work of SSD. Compared with original SSD, DSSD (Deconvolutional Single Shot Detector) adds additional deconvolutional layers and more sophisticated structure for category classifiction and bounding box coordinates regression. As shown in the following figure, the part till blue feature maps is same with original SSD. Then Deconvolution Module and Prediction Module are applied. 

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/DSSD_1.jpg" alt="" width="640"/>

