**TITLE**: Fully Convolutional Instance-aware Semantic Segmentation

**AUTHOR**: Yi Li, Haozhi Qi, Jifeng Dai, Xiangyang Ji, Yichen Wei

**ASSOCIATION**: Microsoft Research Asia, Tsinghua University

**FROM**: [arXiv:1611.07709](https://arxiv.org/abs/1611.07709)

## CONTRIBUTIONS ##

An end-to-end fully convolutional approach for instance-aware semantic segmentation is proposed. The underlying convolutional representation and the score maps are fully shared for the mask prediction and classification sub-tasks, via a novel joint formulation with no extra parameters. The network structure is highly integrated and efficient. The per-ROI computation is simple, fast, and does not involve any warping or resizing operations.

## METHOD ##

The proposed method is highly related with a previous [work](http://joshua881228.webfactional.com/blog_reading-note-r-fcn-object-detection-via-region-based-fully-convolutional-networks_107/) or R-FCN. The following figure gives an illustration:

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/InstanceFCN.jpeg" alt="" width="640"/>

Different from the mentioned previous work, this work predicts two maps, ROI inside map and ROI outside map. The two score maps jointly account for mask prediction and classification sub-tasks. For mask prediction, a softmax operation produces the per-pixel foreground probability. For mask clssification, a max operation produces the per-pixel likelihood of "belonging to the object category".

For an input image, 300 ROIs with highest scores are generated from RPN. They pass through the bbox regression branch and give rise to another 300 ROIs. For each ROI, its classification scores and foreground mask (in probability) is predicted for all categories. NMS with an IoU threshold is used to filter out highly overlapping ROIs. The remaining ROIs are classified as the categories with highest classification scores. Their foreground masks are obtained by mask voting. For an ROI under consideration, the ROIs (from the 600) are found with IoU scores higher than 0.5. Their foreground masks of the category are averaged on a per-pixel basis, weighted by their classification scores. The averaged mask is binarized as the output.

## ADVANTAGES ##

1. End-to-end training and testing alleviate the simplicity of the system.
2. Utilizing the idea of R-FCN, its efficiency is proved.