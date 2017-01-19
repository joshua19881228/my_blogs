**TITLE**: Towards Accurate Multi-person Pose Estimation in the Wild

**AUTHOR**: George Papandreou, Tyler Zhu, Nori Kanazawa, Alexander Toshev, Jonathan Tompson, Chris Bregler, Kevin Murphy

**ASSOCIATION**: Google

**FROM**: [arXiv:1701.01779](https://arxiv.org/abs/1701.01779)

## CONTRIBUTIONS ##

A method for multi-person detection and 2D keypoint localization in the wild is proposed.


## METHOD ##

The multi-person pose estimation system is a two step cascade, as illustrated in the Following figure. 

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/TAMPEW_1.jpg" alt="" width="480"/>

In the first stage, a person detector is used to produce a bounding box around each person instance. In the second stage, a pose estimator is produced to the image crop extracted around each detected person instance in order to localize its keypoints.

### Person Box Detection ###

A Faster-RCNN system based on ResNet-Inception architecture is used for person box detection. The detector is first trained on 80 categories in COCO dataset. Then the model is further finetuned on dataset only with bounding boxes of person.

### Person Pose Estimation ###

A combined classification and regression approach is adoptted. Each spatial position is first classified whether it is in the vicinity of keypoints (K types) or not (which is a K-channel “heatmap”), then a 2-D local offset vector is predicted to get a more precise estimate of the corresponding keypoint location. The following figure illustrates the procedure.

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/TAMPEW_2.jpg" alt="" width="480"/>

The bounding box is first adjusted to a fixed aspect ratio (height/width = 1.37) and the patch is cropped from the image and resized to 353*257. A ResNet with 101 layers is used to produce heatmap and offsets. The following figure shows an input and ground-truth output of the network. 

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/TAMPEW_2.jpg" alt="" width="480"/>

## SOME IDEAS ##

1. The pipeline in two stages separated detection and pose estimation.
2. The relations between keypoints might be learnt in CNN, but it is not obvious.
