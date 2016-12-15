**TITLE**: Feature Pyramid Networks for Object Detection

**AUTHOR**: Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie

**ASSOCIATION**: Facebook AI Research, Cornell University and Cornell Tech

**FROM**: [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)

## CONTRIBUTIONS ##

A new topdown architecture with lateral connections is developed for building high-level semantic feature maps at all scales is proposed. This architecture, called a Feature Pyramid Network (FPN), shows significant improvement as a generic feature extractor in several applications.

## METHOD ##

The idea of this work is simple and is illustrated in the following figure

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/FPN.jpg" alt="" width="640"/>

Similar with SSD, predictions are made in different scales. But connects exist between different scales. Though the author calls the connections as lateral connections, it is actually skip connections among different layers, just like what ResNet does.

