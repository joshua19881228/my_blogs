**TITLE**: Beyond Skip Connections: Top-Down Modulation for Object Detection

**AUTHOR**: Abhinav Shrivastava, Rahul Sukthankar, Jitendra Malik, Abhinav Gupta

**ASSOCIATION**: CMU, UC Berkeley, Google Research

**FROM**: [arXiv:1612.06851](https://arxiv.org/abs/1612.06851)

## CONTRIBUTIONS ##

In this paper top-down modulations is proposed as a way to incorporate fine details into the detection framework. The standard bottom-up, feedforward ConvNet is supplemented with a top-down modulation (TDM) network, connected using lateral connections. These connections are responsible for the modulation of lower layer filters, and the top-down network handles the selection and integration of features.

## METHOD ##

The idea of this work is very similar with the work of *[Feature Pyramid Networks for Object Detection](http://joshua881228.webfactional.com/blog_reading-note-feature-pyramid-networks-for-object-detection_181/)*. An example of Top-Down Modulation (TDM) Network is illustrated as the following figure

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/TDM_1.jpg" alt="" width="640"/>

TDM is integrated with the bottom-up network with lateral connections. $C_{i}$ are bottom-up, feedforward feature blocks, $L_{i}$ are the lateral modules which transform low level features for the top-down contextual pathway. Finally, $T_{j,i}$, which represent flow of top-down information from index $j$ to $i$. 

In this paper, the $T$ blocks are implemented using single convolutional layer (with non-linear activation) optionally with upsampling operation. The features from $C$ (processed by $L$) and $T$ are concated then sent to a convolutional layer for combination, as the following figure shows

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/TDM_2.jpg" alt="" width="640"/>

At training stage, one new pair of lateral and top-down modules is added at a time and trained repeatedly from a pre-trained model.