**TITLE**:  Fused DNN: A deep neural network fusion approach to fast and robust pedestrian detection

**AUTHOR**: Xianzhi Du, Mostafa El-Khamy, Jungwon Lee, Larry S. Davis

**ASSOCIATION**: University of Maryland, Samsung Electronics

**FROM**: [arXiv:1610.03466](https://arxiv.org/abs/1610.03466)

## CONTRIBUTIONS ##

A deep neural network fusion architecture is proposed to address the pedestrian detection problem, called Fused Deep Neural Network (F-DNN).

## METHOD ##

The proposed network architecture consists of a pedestrian candidate generator, a classification network, and a pixel-wise semantic segmentation network. The pipeline of the proposed network fusion architecture is shown in the following figure:

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/FusedDNN.jpg" alt="" width="640"/>

**Pedestrian Candidate Generator** is implemented by SSD. It provides a large pool of pedestrian candidates varying in scales and aspect ratios. Pedestrian candidates generated should cover almost all the ground truth pedestrians, even though many false positives are introduced at the same time. 

**Classification Network** consists of multiple binary classification deep neural networks which are trained on the pedestrian candidates from *Pedestrian Candidate Generator*. 

**Soft-rejection based DNN Fusion** works as follows: Consider one pedestrian candidate and one classifier. If the classifier has high confidence about the candidate, we boost its original score from the candidate generator by multiplying with a confidence scaling factor greater than one. Otherwise, we decrease its score by a scaling factor less than one. To fuse all $M$ classifiers, the candidate’s original confidence score is multiplied with the product of the confidence scaling factors from all classifiers in the classification network.

$$ S_{FDNN} = S_{SSD} \times \prod_{m=1}^{M} a_{m} $$

where

$$ a_{m} = max(p_{m} \times \frac{1}{a_{c}} , b_{c})  $$

and $a_{c}$ and $b_{c}$ are chosen as 0.7 and 0.1 by cross validation.

**Pixel-wise Semantic Segmentation Network** is trained to get a binary map. DegreeDgreee to which each candidate's BB overlaps with the pedestrian category in the SS activation mask gives a measure of the confidence of the SS network in the candidate generator's results. If the generation pixels occupy at least 20% of the candidate BB area, its score is kept unaltered; Otherw, SNF is applied to scale the original confidence scores.

## SOME IDEAS ##

The idea of the work is simple. It seems a very tricky implementation of pedestrian detection. Though the author claims that it is efficient, it is hard to say how efficient it is using very complex cnn classifiers.
