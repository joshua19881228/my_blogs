**TITLE**: Densely Connected Convolutional Networks

**AUTHOR**: Gao Huang, Zhuang Liu, Kilian Q. Weinberger

**ASSOCIATION**: Cornell University, Tsinghua University

**FROM**: [arXiv:1608.06993](https://arxiv.org/abs/1608.06993)

## CONTRIBUTIONS ##

Dense Convolutional Network (DenseNet) is proposed, which embraces the observation that networks can be substantially deeper, more accurate and efficient to train if they contain shorter connections between layers close to the input and those close to the output.

## METHOD ##

DenseNet is a network architecture where each layer is directly connected to every other layer in a feed-forward fashion (within each dense block). For each layer, the feature maps of all preceding layers are treated as separate inputs whereas its own feature maps are passed on as inputs to all subsequent layers. The idea can be illustrated as the following figure:

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/DenseNet.jpeg" alt="" width="640"/>

## SOME IDEAS ##

In the work of Yoshua Bengio's Understanding intermediate layers using linear classifier probes, the author claims that the raw input is helpful at the beginning of the training of the network. So maybe the dense connection plays similar role in this work.

Using Caffe to implement DenseNet, large memory is of need because of the large number of split layers.
