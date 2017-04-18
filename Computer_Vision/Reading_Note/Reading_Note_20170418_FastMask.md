**TITLE**: Deep Image Matting

**AUTHOR**: Ning Xu, Brian Price, Scott Cohen, Thomas Huang

**ASSOCIATION**: Beckman Institute for Advanced Science and Technology, University of Illinois at Urbana-Champaign, Adobe Research

**FROM**: [arXiv:1703.03872](https://arxiv.org/abs/1703.03872)

## CONTRIBUTIONS ##

1. A novel deep learning based algorithm is proposed that can predict alpha matte of an image based on both low-level features and high-level context.

## METHOD ##

The proposed deep model has two parts. 

1. The first part is a CNN based encoder-decoder network, which is similar with typical FCN networks that are used for semantic segmentation. This part takes the RGB image and its corresponding trimap as input. Its output is the alpha matte of the image. 
2. The second part is a small convolutional network that is used to refine the output of the first part. The input of this part is the original image and the predicted alpha matte from the first part.

The method is illustrated in the following figure.

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/Reading_Note_20170316_Matting_0.jpg" alt="" width="640"/>

### Matting encoder-decoder stage ###

The first network leverages two losses. One is **alpha-prediction loss** and the other one is **compositional loss**.

**Alpha-prediction loss** is the absolute difference between the ground truth alpha values and the predicted alpha values at each pixel, which defines as

$$\mathcal{L}_{\alpha}^{i} = \sqrt{(\alpha_{p}^{i} - \alpha_{g}^{i})^{2}+\epsilon^2}, \alpha_{p}^{i}, \alpha_{g} \in [0,1]$$

where $\alpha_{p}^{i}$ is the output of the prediction layer at pixel $i$ and $\alpha_{g}^{i}$ is the ground truth alpha value at pixel $i$. $\epsilon$ is a small value which is equal to $10^{-1}$ and is used to ensure differentiable property.

**Compositional loss** the absolute difference between the ground truth RGB colors and the predicted RGB colors composited by the ground truth foreground, the ground truth background and the predicted alpha mattes. The loss is defined as

$$\mathcal{L}_{c}^{i} = \sqrt{(c_{p}^{i} - c_{g}^{i})^{2}+\epsilon^2}$$

where $c$ denotes the RGB channel, $p$ denotes the image composited by the predicted alpha, and $g$ denotes the image composited by the ground truth alpha.

Since only the alpha values inside the unknown regions of trimaps need to be inferred, therefore weights are set on the two types of losses according to the pixel locations, which can help the network pay more attention on the important areas. Specifically, $w_{i} = 1$ if pixel $i$ is inside the unknown region of the trimap while $w_{i} = 0$ otherwise.

### Matting refinement stage ###

The input to the second stage of our network is the concatenation of an image patch and its alpha prediction from the first stage, resulting in a 4-channel input. This part is trained after the first part is converged. After the refinement part is also converged, finally fine-tune the the whole network together. Only the alpha prediction loss is used.

## SOME IDEAS ##

1. The *trimap* is a very strong prior. The question is how to get it.