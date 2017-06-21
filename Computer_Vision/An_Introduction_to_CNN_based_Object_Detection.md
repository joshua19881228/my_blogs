# 1. Content #

## Brief Revisit to the "Ancient" Algorithm ##

* HOG (before \*2007)
* DPM (\*2010~2014)

## Epochal Evolution of R-CNN ##

* R-CNN \*2014
* Fast-RCNN \*2015
* Faster-RCNN \*2015

## Efficient One-shot Methods ##

* YOLO
* SSD

## Others ##

![Goal of Object Detection](https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Object_Detection_Figures/Goal_of_Detection.png "Goal of Object Detection =480")

# 2. Brief Revisit to the "Ancient" Algorithm #

## 2.1 Histograms of Gradients (HOG) ##

![Histograms of Gradients](https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Object_Detection_Figures/HOG.png "Histograms of Gradients =640")

* Calculate gradient for each pixel
* For each **Cell**, a histogram of gradient is computed
* For each **Block**, a HOG feature is extracted by concatenating histograms of each Cell

If Block size = 16\*16, Block stride = 8, Cell size = 8\*8, Bin size = 9, Slide-window size = 128\*64, then HOG feature is a 3780-d feature. #Block=((64-16)/8+1)\*((128-16)/8+1)=105, #Cell=(16/8)\*(16/8)=4, 105\*4\*9=3780

## 2.2 Deformable Part Models (DPM) ##

![DPM](https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Object_Detection_Figures/DPM.png "DPM =480")

$$ D_{i,l}(x,y) = \max \limits_{dx,dy} (R_{i,l}(x+dx, y+dy)-d_{i}\cdot \phi_{d}(dx,dy)) $$

This transformation spreads high filter scores to nearby locations, taking into account the deformation costs.

$$ score(x_{0},y_{0},l_{0}) = R-{0,l_{0}}(x_{0},y_{0})+ \sum_{i=1}^{n} D_{i, l_{0}-\lambda}(2(x_{0},y_{0})+v_{i})+b $$

The overall root scores at each level can be expressed by the sum of the root filter response at that level, plus shifted versions of transformed and sub-sampled part responses.

# 3. Epochal Evolution of R-CNN #

## 3.1 RCNN ##

### 3.1.1 Regions with CNN Features ###

![DPM](https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Object_Detection_Figures/RCNN.png "DPM =640")

* Region proposals (Selective Search, ~2k)
* CNN features (AlexNet, VGG-16, warped region in image)
* Classifier (Linear SVM per class)
* Bounding box (Class-specific regressor)
* Run-time speed (VGG-16, 47 s/img on single K40 GPU)

### 3.1.2 Experiment Result (AlexNet) ###

* Without FT, fc7 is worse than fc6, pool5 is quite competitive. Much of the CNN’s representational power comes from its convolutional layers, rather than from the much larger densely connected layers.
* With FT, The boost from fine-tuning is much larger for fc6 and fc7 than for pool5. Pool5 features are general. Learning domain-specific non-linear classifiers helps a lot.
* Bounding box regression helps reduce localization errors. 

### 3.1.3 Interesting Details – Training ###

* Pre-trained on ILSVRC2012 classification task
* Fine-tuned on proposals with N+1 classes without any modification to the network

    1. IOU>0.5 over ground-truth as positive samples, others as negative samples
    2. Each mini-batch contains 32 positive samples and 96 background samples

* SVM for each category
    
    1. Ground-truth window as positive samples
    2. IOU<0.3 over ground-truth as negative samples
    3. Hard negative mining is adopted

* Bounding-box regression

    1. Class-specific
    2. Features computed by CNN
    3. Only the proposals IOU>0.6 overlap ground-truth
    4. Coordinates in pixel

### 3.1.4 Interesting Details – FP Error Types ###

* Loc: poor localization, 0.1 < IOU < 0.5
* Sim: confusion with a similar category
* Oth: confusion with a dissimilar object category
* BG: a FP that fired on background

## 3.2 Fast-RCNN ##

### 3.2.1 What's Wrong with RCNN  ###
    
* Training is a multi-stage pipeline (Proposal, Fine-tune, SVMs, Regressors)
* Training is expensive in space and time (Extract feature from every proposal, Need to save to disk)
* Oject detection is slow (47 s/img on K40)

### 3.2.2 R-CNN with ROI Pooling ###

* Region proposals (Selective Search, ~2k)
* CNN features (AlexNet, VGG-16, ROI in feature map)
* Classifier (sub-network softmax)
* Bounding box (sub-network regressor)
* Run-time speed (VGG-16, 0.32 s/img on single K40 GPU)

### 3.2.3 ROI Pooling ###

* Inspired by Spatial Pyramid Pooling (SPPNet)
* Convert arbitrary input size to fixed length

    1. The input is an ROI area in feature map
    2. The input is divided into grids
    3. In each grid, pooling is used to extract features

### 3.2.4 Experiment Result (VGG16) ###

### 3.2.5 Interesting Details – Training ###

* Pre-trained on ILSVRC2012 classification task
* Fine-tuned with N+1 classes and two sibling layers

    1. Fine-tune the whole network
    2. Each mini-batch has 2 images and 64 ROIs from each images
    3. 25% of the ROIs have IOU>0.5 with ground-truth as positive samples
    4. The rest of the ROIs have IOU [0.1, 0.5) with ground-truth as background samples
    5. Multi-task loss, one loss for classification and one for bounding box regression
    6. ROI pooling back-propagation is similar with max-pooling

* Accelerate using truncated SVD

    Implemented by using two FCs without non-linear activation

* Training time

    1. 146x faster than R-CNN
    2. If accelerated with truncated SVD, 213x faster than R-CNN

### 3.2.6 Interesting Details – Design evaluation ###

* Does multi-task training help? Yes, it does!
* Test with multiple scales? Yes but with cost.
* Do SVMs outperform softmax? Interesting…

## 3.3 Faster-RCNN ##

### 3.3.1 Room to improve Fast-RCNN ###

* Region proposal has become the bottleneck
* 2s for Selective Search, 0.320s for Fast-RCNN
* Why not a unified end-to-end framework

### 3.3.2 Fast-RCNN with RPN (Region Proposal Network) ###

* Region proposals (RPN, ~300)
* Classifier (sub-network softmax)
* Bounding box (RPN regressor, sub-network regressor)
* Run-time speed (VGG-16, 0.198 s/img on single K40 GPU)

### 3.3.3 Region Proposal Network ###

* Anchors

    1. reference box, prior box, default box
    2. Works in a slide-window way
    3. Implemented by 3*3 kernel convolution
    4. Centered at the slide-window

* Translation-Invariant

    1. If objects translated, proposal should be translated
    2. Translated along slide-window

* Multi-Scale and Multi-Ratio

    1. A pyramid of anchors
    2. A set of different ratios
    3. Relies on single scale feature map

* Objectness and Localization

    1. Two siblings
    2. Objectness score
    3. Bounding box regression

### 3.3.4 Experiment Result ###

### 3.3.5 Interesting Details – Training ###

* Sharing Features for RPN and Fast-RCNN

    1. Alternating training
    2. Approximate joint training
    3. Non-approximate joint training

* Alternating Training

    1. Train RPN(@) using pre-trained model
    2. Train Fast-RCNN(#) using pre-trained model and @’s proposal
    3. Train RPN($) using #’s weight with shared layers fixed
    4. Train Fast-RCNN using $’s proposal with shared layers fixed

* Training RPN

    1. Each mini-batch arises from a single image
    2. Positive samples: the anchors with (1) the highest IOU and (2) IOU>0.7 overlap with any ground-truth
    3. Negative samples: IOU<0.3 overlap with all ground-truth
    4. Randomly sample 256 anchors, pos:neg = 1:1
    5. Loss function with Ncls=256, Nreg=~2400, λ=10
    6. Anchors cross image boundaries do not contribute at training stage

### 3.3.6 Interesting Details – Design Evaluation ###

* Ablation on RPN

    1. Sharing: Detector feature helps RPN
    2. RPN generate quite good proposals
    3. No cls, randomly selecting proposal, score matters
    4. No reg, worse localization error

* Timing

    1. Nearly cost free
    2. Less proposal

* Anchors

    1. Scale is more effective


# 4 Efficient One-shot Methods #

## 4.1 YOLO ##

### 4.1.1 You Only Look Once ###

* A simple forward on the full image (almost same with a classification task)
* Frame object detection as a regression problem (bounding box coordinates, class probabilities)
* Extremely fast (45 fps for base network, or 150 fps for fast version)
* Reasoning globally on the full context (no slide-window or region proposals)
* Generalizable representations of objects (stable from natural images to artwork)

### 4.1.2 Unified Detection ###

* The input is divided into S x S grid
* Each grid cell predicts B bounding boxes
* 5 predictions for one bounding box: x, y, w, h, score

    1. (x, y) center of the box relative to the bounds of grid
    2. w, h are width and height relative to the whole image
    3. the score is a measure of objectness

* Each grid cell predicts C conditional class probabilities
* Class-specific confidence score is defined as
* One predictor is “responsible” for an object having the highest IOU with the ground-truth
* The output is an SxSx(Bx5+C) tensor

### 4.1.3 Experiment Result ###

* Most effective among real-time detectors
* Most efficient among near real-time detectors

### 4.1.4 Limitations ###

* Too few bounding boxes

    1. Nearby objects
    2.  Small objects

* Data driven

    1. Sensitive to new or rare ration

### 4.1.5 Interesting Details – Training ###

* Pre-train the first 20 layers on ImageNet
* Pre-train on 224*224 images
* Fine-tune 24 layers on detection dataset
* Fine-tune on 448*448 images
* Tricks to balance loss

    1. Weight: localization vs. classification
    2. Weight: positive vs. negative of objectness
    3. Square root: large object vs. small object

* “Warm up” to start training

    1. For first epoch, raise 0.001 to 0.01
    2. 0.01 for 75 epochs
    3. 0.001 for 30 epochs
    4. 0.0001 for 30 epochs

## 4.2 SSD ##

### 4.2.1 Single Shot MultiBox Detector ###

* Combine anchor and one-shot prediction
* Extract multi-scale features
* Refine multi-scale and multi-ratio anchors
* Dilated convolution

### 4.2.2 Multi-scale Prediction ###

* Multi-scale and Multi-ratio anchors

    1. Each feature map cell corresponds to k anchors
    2. Similar to Faster-RCNN, but in multi-scale feature map and directly output category info

* Multi-scale feature maps for detection

    1. Additional layers are added to the base network
    2. Different filters are applied to different scale/ratio anchors
    3. (c+4)k filters for k anchors and c categories in one cell, (c+4)kmn outputs for  an m*n feature map

### 4.2.3 Experiment Result ###

### 4.2.4 Interesting Details – Training ###

* Matching anchors with ground-truth

    1. Match each ground-truth to a default box with the best IOU
    2. Match the anchor to any ground truth with IOU higher than a threshold

* Training objective
* Scales and aspect ratios for anchors

    1. Regularly spaced scales 
    2. {1,2,3,1/2,1/3} – 6 ratios

* Hard negative mining

    1. Sort the anchors using the highest confidence loss
    2. Pick the top ones so that neg:pos = 3:1

* Data augmentation

    1. Sample randomly from training images
    2. Entire input image
    3. Sample a patch so that min IOU with object is 0.1, 0.3, 0.5, 0.7 or 0.9
    4. Randomly sample a patch, [0.1, 1] of the image, aspect ration [1/2, 2], randomly flip

* Comparison

# 5 Others #

* PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection

    1. Variant of Faster-RCNN
    2. Design of architecture

* Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks

    1. Both local and global information are take into account
    2. Skip pooling uses the information of different scales

* R-FCN: Object Detection via Region-based Fully Convolutional Networks

    1. Position-sensitive RoI pooling

* Feature Pyramid Networks for Object Detection

    1. lateral connections is developed for building high-level semantic feature maps at all scales

* Beyond Skip Connections: Top-Down Modulation for Object Detection

    1. Similar with FPN

* YOLO9000: Better, Faster, Stronger

    1. Better, Faster, Stronger

* DSSD: Deconvolutional Single Shot Detector

