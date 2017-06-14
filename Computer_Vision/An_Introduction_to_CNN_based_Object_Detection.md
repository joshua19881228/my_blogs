# 1. Content #

## Brief Revisit to the "Ancient" Algorithm ##

* HOG (before \*2007)
* DPM (\*2010~2014)

## Epochal Evolution of R-CNN

* R-CNN \*2014
* Fast-RCNN \*2015
* Faster-RCNN \*2015

## Efficient One-shot Methods

* YOLO
* SSD

## Others

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

# 3. Epochal Evolution of R-CNN: RCNN #

## 3.1 Regions with CNN Features ##

* Region proposals (Selective Search, ~2k)
* CNN features (AlexNet, VGG-16, warped region in image)
* Classifier (Linear SVM per class)
* Bounding box (Class-specific regressor)
* Run-time speed (VGG-16, 47 s/img on single K40 GPU)

## 3.2 Experiment Result (AlexNet) ##

* Without FT, fc7 is worse than fc6, pool5 is quite competitive. Much of the CNN’s representational power comes from its convolutional layers, rather than from the much larger densely connected layers.
* With FT, The boost from fine-tuning is much larger for fc6 and fc7 than for pool5. Pool5 features are general. Learning domain-specific non-linear classifiers helps a lot.
* Bounding box regression helps reduce localization errors. 

## 3.3 Interesting Details – Training ##

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

## 3.4 Interesting Details – FP Error Types ##

* Loc: poor localization, 0.1<IOU<0.5
* Sim: confusion with a similar category
* Oth: confusion with a dissimilar object category
* BG: a FP that fired on background

