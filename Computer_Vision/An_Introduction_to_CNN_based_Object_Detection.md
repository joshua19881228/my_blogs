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
