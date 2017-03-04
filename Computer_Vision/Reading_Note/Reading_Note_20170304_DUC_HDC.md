**TITLE**: Understanding Convolution for Semantic Segmentation

**AUTHOR**: Panqu Wang, Pengfei Chen, Ye Yuan, Ding Liu, Zehua Huang, Xiaodi Hou, Garrison Cottrell

**ASSOCIATION**: UC San Diego, CMU, UIUC, TuSimpl

**FROM**: [arXiv:1702.08502](https://arxiv.org/abs/1702.08502)

## CONTRIBUTIONS ##

1. A method called dense upsampling convolution (DUC) is proposed, which instead of trying to recover the full-resolution label map at once, an array of upscaling filters are learnt to upscale the downsized feature maps into the final dense feature map of the desired size.
2. A simple hybrid dilation convolution (HDC) framework is proposed, which instead of using the same rate of dilation for the same spatial resolution, a range of dilation rates are used and are concatenated serially the same way as “blocks” in ResNet-101.

## METHOD ##

**DUC** is illustrated as the following figure.

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/DUC_HDC_1.jpg" alt="" width="640"/>

The key idea of DUC is to divide the whole label map into equal subparts which have the same height and width as the incoming feature map. Every feature map in the dark blue part is a corner or a part of the whole output. 

**HDC** is illustrated as the following figure.

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/DUC_HDC_2.jpg" alt="" width="480"/>

Instead of using the same dilation rate for all layers after the downsampling occurs, a different dilation rate for each layer is used. The pixels (marked in blue) contributes to the calculation of the center pixel (marked in red) through three convolution layers with kernel size 3 × 3. Subsequent convolutional layers have dilation rates of r = 1, 2, 3, respectively.