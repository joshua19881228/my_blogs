**TITLE**: DSSD: Deconvolutional Single Shot Detector

**AUTHER**: Cheng-Yang Fu, Wei Liu, Ananth Ranga, Ambrish Tyagi, Alexander C. Berg

**FROM**: [arXiv:1701.06659](https://arxiv.org/abs/1701.06659)

### CONTRIBUTIONS ###

1. A combination of a state-of-the-art classifier (Residual-101) with a fast detection framework (SSD) is proposed.
2. Deconvolution layers are applied to introduce additional large-scale context in object detection and improve accuracy, especially for small objects.

### METHOD ###

This is a successive work of SSD. Compared with original SSD, DSSD (Deconvolutional Single Shot Detector) adds additional deconvolutional layers and more sophisticated structure for category classifiction and bounding box coordinates regression. As shown in the following figure, the part till blue feature maps is same with original SSD. Then Deconvolution Module and Prediction Module are applied. 

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/DSSD_1.jpg" alt="" width="640"/>

Recent works such as [Beyond Skip Connections: Top-Down Modulation for Object Detection](https://arxiv.org/abs/1612.06851) and [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144) propose to incorporate fine details into the detection framework using deconvolutional layers and skip connections. DSSD utilizes this idea as well using Deconvolutional Module, shown in the following figure.

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/DSSD_3.jpg" alt="" width="480"/>

Several different structures for Prediction Module are proposed. These structures take the idea from ResNet as illustrated in the following figure.

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/DSSD_2.jpg" alt="" width="640"/>

### SOME IDEAS ###

1. Using ResNet-101 and more sophisticated structure for prediction is helpful to improve the performance, but the computation cost is high. 
2. The idea of using deconvolutional layers to enlarge the feature maps and using skip connections to combine detail features is becoming popular.