**TITLE**: FastMask: Segment Multi-scale Object Candidates in One Shot

**AUTHOR**: Hexiang Hu, Shiyi Lan, Yuning Jiang, Zhimin Cao, Fei Sha

**ASSOCIATION**: UCLA, Fudan University, Megvii Inc.

**FROM**: [arXiv:1703.03872](https://arxiv.org/abs/1703.03872)

## CONTRIBUTIONS ##

1. A novel weight-shared residual neck module is proposed to zoom out feature maps of CNN while preserving calibrated feature semantics, which enables efficient multi-scale training and inference.
2. A novel scale-tolerant head module is proposed which takes advantage of attention model and significantly reduces the impact of background noises caused by unmatched receptive fields.
3. A framework capable for one-shot segment proposal is made up, namely FastMask. The proposed framework achieves the the state-of-the-art results in accuracy while running in near real time on MS COCO benchmark.


## METHOD ##

### Network Architecture ###

The network architecture is illustrated in the following figure. 

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/Reading_Note_20170418_FastMask_1.png" alt="" width="640"/>

With the base feature map, a shared neck module is applied recursively to build feature maps with different scales. These feature maps are then fed to a one-by-one convolution to reduce their feature dimensionality. Then we extract dense sliding windows from those feature maps and do a batch normalization across all windows to calibrate and redistribute window feature maps. With a feature map downscaled by factor $m$, a sliding window of size $(k, k)$ corresponds to a patch of $(m \times k, m \times k)$ at original image. Finally, a unified head module is used to decode these window features and produce the output confidence score as well as object mask.

### Residual Neck ###

The *neck* module is actually used to downscale the feature maps so that features with different scales can be extracted. 

There are another two choices. One is *Max pooling neck*, which produces uncalibrated feature in encoding pushing the mean of downscaled feature higher than original. The other one is *Average pooling neck*, which smoothes out discriminative feature during encoding, making the top feature maps appear to be blurry.

*Residual neck* is then proposed to learn parametric necks that preserve feature semantics. The following figure illustrates the method.

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/Reading_Note_20170418_FastMask_2.png" alt="" width="480"/>

### Attentional Head ###

Given the feature map of a sliding window as the input, a spatial attention is generated through a fully connected layer, which takes the entire window feature to generate the attention score for each spatial location on the feature map. The spatial attention is then applied to window feature map via the element-wise multiplication across channels. Such operation enables the head module to enhance features on the salient region, where is supposed to be the rough location of the target object. Finally, the enhanced feature map will be fed into a fully connected laye to decode the segmentation mask of the object. This module is illustrated in the following figure.

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/Reading_Note_20170418_FastMask_2.png" alt="" width="640"/>

The feature pyramid is sparse in this work because of the downscale operation. The sparse feature pyramid raises the probability that there exists no suitable feature maps for an object to decode, and also raises the risk of introducing background noises when the object is decoded from an unsuitable feature map with too larger receptive field. So salient region is introduced in this head. With the capability of paying attention to the salient region, a decoding head could reduce the noises from the backgrounds of a sliding window and thus produce high quality segmentation results when the receptive field is unmatched with the scale of object. Also the salient region attention has the tolerance to shift disturbance.

## SOME IDEAS ##

1. This work shares the similar idea with most one-shot alogrithms, extracting sliding window in the feature map and endcode them with a following network.
2. How to extract sliding windows?
