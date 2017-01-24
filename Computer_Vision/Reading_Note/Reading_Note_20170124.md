**TITLE**: A New Convolutional Network-in-Network Structure and Its Applications in Skin Detection, Semantic Segmentation, and Artifact Reduction

**AUTHOR**: Yoonsik Kim, Insung Hwang, Nam Ik Cho

**ASSOCIATION**: Seoul National University

**FROM**: [arXiv:1701.06190](https://arxiv.org/abs/1701.06190)

## CONTRIBUTIONS ##

1. a new inception-like convolutional network-in-network structure is proposed, which consists of convolution and rectified linear unit (ReLU) layers only. That is, pooling and subsampling layer are excluded that reduce feature map size, because decimated features are not helpful at the reconstruction stage. Hence, it is able to do one-to-one (pixel wise) matching at the inner network and also intuitive analysis of feature map correlation.
2. Proposed architecture is applied to several pixel-wise labeling and restoration problems and it is shown to provide comparable or better performances compared to the state-of-the-art methods.

## METHOD ##

The network structure is inspired by Inception. The comparison of the structure is illustrated in the following figure.

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/20170124.jpg" alt="" width="640"/>

Pooling is removed in the proposed inception module and a larger size kernel instead is added to widen the receptive field which might have been reduced by the removal of pooling. The main inspiration of such modification is to maintain the large receptive field while keep the resolution of output same with input resolution at the same time.

## SOME IDEAS ##

As the network removes the operation that reduces the resolution of the feature maps, both forward and backward propagation could be very slow if the input size is large.