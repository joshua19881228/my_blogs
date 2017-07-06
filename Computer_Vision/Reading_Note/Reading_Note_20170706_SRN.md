**TITLE**: Learning Spatial Regularization with Image-level Supervisions for Multi-label Image Classification

**AUTHOR**: Feng Zhu, Hongsheng Li, Wanli Ouyang, Nenghai Yu, Xiaogang Wang

**ASSOCIATION**: University of Science and Technology of China, University of Sydney, The Chinese University of Hong Kong

**FROM**: [arXiv:1702.05891](https://arxiv.org/abs/1702.05891)

## CONTRIBUTIONS ##

1. An end-to-end deep neural network for multi-label image classification is proposed, which exploits both semantic and spatial relations of labels by training learnable convolutions on the attention maps of labels. Such relations are learned with only image-level supervisions. Investigation and visualization of learned models demonstrate that our model can effectively capture semantic and spatial relations of labels.
2. The proposed algorithm has great generalization capability and works well on data with different types of labels.

## METHOD ##

The proposed Spatial Regularization Net (SRN) takes visual features from the main net as inputs and learns to regularize spatial relations between labels. Such relations are exploited based on the learned attention maps for the multiple labels. Label confidences from both main net and SRN are aggregated to generate final classification confidences. The whole network is a unified framework and is trained in an end-to-end manner.

The scheme of SRN is illustrated in the following figure.

![Overall Framework of SRN](https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/Reading_Note_20170706_SRN_0.png "Overall Framework of SRN =640")

To train the network, 

1. Finetune only the main net on the target dataset. Both $ f_{cnn} $ and $ f_{cls} $ are learned with cross-entropy loss for classification. 
2. Fix $ f_{cnn} $ and $ f_{cls} $. Train $ f_{att} $ and $ conv1 $ with cross-entropy loss for classification.
3. Train $ f_{sr} $ with cross-entropy loss for classification by fixing all other sub-networks.
4. The whole network is jointly finetuned with joint loss.

The main network follows the structure of ResNet-101. And it is finetuned on the target dataset. The output of Attention Map and Confidence Map has $ C $ channels which is same with the number of categories. Their outputs are merged by element-wise multiplication and average-pooled to a feature vector in step 2. In step 3, instead of an average-pooling, $ f_{sr} $ follows. $ f_{sr} $ is implemented as three convolution layers with ReLU nonlinearity followed by one fully-connected layer as shown in the following figure.

![Structure of fsr](https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/Reading_Note_20170706_SRN_1.png "Structure of fsr =480")

$ conv4 $ is composed of single-channel filters. In Caffe, it can be implemnted using "group". Such design is because one label may only semantically relate to a small number of other labels, and measuring spatial relations with those unrelated attention maps is unnecessary.
