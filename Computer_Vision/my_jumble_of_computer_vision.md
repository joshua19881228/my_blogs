I am going to maintain this page to record a few things about computer vision that I have read, am doing, or will have a look at. Previously I'd like to write short notes of the papers that I have read. It is a good way to remember and understand the ideas of the authors. But gradually I found that I forget much portion of what I had learnt because in addition to paper I also derive knowledges from others' blogs, online courses and reports, not recording them at all. Besides, I need a place to keep a list of what I should have a look at but do not at the time when I discover them. This page will be much like a catalog.

## Papers and Projects

### Object/Saliency Detection ###

* PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection ([PDF](http://arxiv.org/abs/1608.08021), [Project/Code](https://github.com/sanghoon/pva-faster-rcnn), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-pvanet-deep-but-lightweight-neural-networks-for-real-time-object-detection_137/))
* Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks ([PDF](http://arxiv.org/abs/1512.04143), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-inside-outside-net-detecting-objects-in-context-with-skip-pooling-and-recurrent-neural-networks_111/))
* Object Detection from Video Tubelets with Convolutional Neural Networks ([PDF](http://arxiv.org/abs/1604.04053), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-object-detection-from-video-tubelets-with-convolutional-neural-networks_109/))
* R-FCN: Object Detection via Region-based Fully Convolutional Networks ([PDF](http://arxiv.org/abs/1605.06409), [Project/Code](https://github.com/daijifeng001/r-fcn), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-r-fcn-object-detection-via-region-based-fully-convolutional-networks_107/))
* SSD: Single Shot MultiBox Detector ([PDF](http://arxiv.org/abs/1512.02325v2), [Project/Code](https://github.com/weiliu89/caffe/tree/ssd), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-ssd-single-shot-multibox-detector_100/))
* Pushing the Limits of Deep CNNs for Pedestrian Detection ([PDF](http://lib-arxiv-008.serverfarm.cornell.edu/abs/1603.04525), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-pushing-the-limits-of-deep-cnns-for-pedestrian-detection_91/))
* Object Detection by Labeling Superpixels([PDF](http://www.cv-foundation.org/openaccess/content_cvpr_2015/ext/3B_072_ext.pdf), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-object-detection-by-labeling-superpixels_74/))
* Crafting GBD-Net for Object Detection ([PDF](https://arxiv.org/abs/1610.02579), [Projct/Code](https://github.com/craftGBD/craftGBD))
	code for CUImage and CUVideo, the object detection champion of ImageNet 2016.
* Fused DNN: A deep neural network fusion approach to fast and robust pedestrian detection ([PDF](https://arxiv.org/abs/1610.03466))
* Training Region-based Object Detectors with Online Hard Example Mining ([PDF](https://arxiv.org/abs/1604.03540), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-training-region-based-object-detectors-with-online-hard-example-mining_159/))
* Detecting People in Artwork with CNNs ([PDF](https://arxiv.org/abs/1610.08871), [Project/Code](https://github.com/BathVisArtData/PeopleArt))
* Deeply supervised salient object detection with short connections ([PDF](https://arxiv.org/abs/1611.04849))

### Segmentation/Parsing ###
* Instance-aware Semantic Segmentation via Multi-task Network Cascades ([PDF](http://arxiv.org/abs/1512.04412), [Project/Code](https://github.com/daijifeng001/MNC))
* ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation ([PDF](http://arxiv.org/abs/1606.02147), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-enet-a-deep-neural-network-architecture-for-real-time-semantic-segmentation_134/))
* Learning Deconvolution Network for Semantic Segmentation ([PDF](http://arxiv.org/abs/1505.04366), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-learning-deconvolution-network-for-semantic-segmentation_119/))
* Semantic Object Parsing with Graph LSTM ([PDF](http://arxiv.org/abs/1603.07063), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-semantic-object-parsing-with-graph-lstm_110/))
* Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding ([PDF](http://arxiv.org/abs/1511.02680), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-bayesian-segnet-model-uncertainty-in-deep-convolutional-encoder-decoder-architectures-for-scene-understanding_72/))
* Learning to Segment Moving Objects in Videos ([PDF](http://arxiv.org/abs/1412.6504), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-learning-to-segment-moving-objects-in-videos_69/))
* Deep Structured Features for Semantic Segmentation ([PDF](http://arxiv.org/abs/1609.07916v1))

    >We propose a highly structured neural network architecture for semantic segmentation of images that combines i) a Haar wavelet-based tree-like convolutional neural network (CNN), ii) a random layer realizing a radial basis function kernel approximation, and iii) a linear classifier. While stages i) and ii) are completely pre-specified, only the linear classifier is learned from data. Thanks to its high degree of structure, our architecture has a very small memory footprint and thus fits onto low-power embedded and mobile platforms. We apply the proposed architecture to outdoor scene and aerial image semantic segmentation and show that the accuracy of our architecture is competitive with conventional pixel classification CNNs. Furthermore, we demonstrate that the proposed architecture is data efficient in the sense of matching the accuracy of pixel classification CNNs when trained on a much smaller data set. 
* CNN-aware Binary Map for General Semantic Segmentation ([PDF](https://arxiv.org/abs/1609.09220))
* Learning to Refine Object Segments ([PDF](https://arxiv.org/abs/1603.08695))
* Clockwork Convnets for Video Semantic Segmentation([PDF](https://arxiv.org/abs/1608.03609), [Project/Code](https://github.com/shelhamer/clockwork-fcn))
* Convolutional Gated Recurrent Networks for Video Segmentation ([PDF](https://arxiv.org/abs/1611.05435))

### Tracking ###
* Spatially Supervised Recurrent Convolutional Neural Networks for Visual Object Tracking ([PDF](http://arxiv.org/abs/1607.05781), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-spatially-supervised-recurrent-convolutional-neural-networks-for-visual-object-tracking_125/))
* Joint Tracking and Segmentation of Multiple Targets ([PDF](http://milanton.de/files/cvpr2015/cvpr2015-ext-abstr-anton.pdf), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-joint-tracking-and-segmentation-of-multiple-targets_70/))
* Deep Tracking on the Move: Learning to Track the World from a Moving Vehicle using Recurrent Neural Networks ([PDF](https://arxiv.org/abs/1609.09365))
* Convolutional Regression for Visual Tracking ([PDF](https://arxiv.org/abs/1611.04215))

### Pose Estimation ###
* Chained Predictions Using Convolutional Neural Networks ([PDF](http://arxiv.org/abs/1605.02346), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-chained-predictions-using-convolutional-neural-networks_108/))
* CRF-CNN: Modeling Structured Information in Human Pose Estimation ([PDF](https://arxiv.org/abs/1611.00468))

### Action Recognition/Event Detection
* Pooling the Convolutional Layers in Deep ConvNets for Action Recognition ([PDF](http://arxiv.org/abs/1511.02126), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-pooling-the-convolutional-layers-in-deep-convnets-for-action-recognition_73/))
* Two-Stream Convolutional Networks for Action Recognition in Videos ([PDF](http://arxiv.org/abs/1406.2199), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-two-stream-convolutional-networks-for-action-recognition-in-videos_71/))
* YouTube-8M: A Large-Scale Video Classification Benchmark ([PDF](https://arxiv.org/abs/1609.08675), [Project/Code](http://research.google.com/youtube8m))
* Spatiotemporal Residual Networks for Video Action Recognition ([PDF](https://arxiv.org/abs/1611.02155))

### Face 
* Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks ([PDF](https://arxiv.org/abs/1604.02878), [Project/Code](https://github.com/kpzhang93/MTCNN_face_detection_alignment))
* Deep Architectures for Face Attributes ([PDF](http://arxiv.org/abs/1609.09018))
* Face Detection with End-to-End Integration of a ConvNet and a 3D Model ([PDF](https://www.arxiv.org/abs/1606.00850), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-face-detection-with-end-to-end-integration-of-a-convnet-and-a-3d-model_145/), [Project/Code](https://github.com/tfwu/FaceDetection-ConvNet-3D))
* A CNN Cascade for Landmark Guided Semantic Part Segmentation ([PDF](https://arxiv.org/pdf/1609.09642.pdf), [Project/Code](http://www.cs.nott.ac.uk/~psxasj/papers/jackson2016guided/))
* Kernel Selection using Multiple Kernel Learning and Domain Adaptation in Reproducing Kernel Hilbert Space, for Face Recognition under Surveillance Scenario ([PDF](https://arxiv.org/abs/1610.00660))
* An All-In-One Convolutional Neural Network for Face Analysis ([PDF](https://arxiv.org/abs/1611.00851))

### Optical Flow 
* DeepFlow: Large displacement optical flow with deep matching ([PDF](https://hal.inria.fr/hal-00873592), [Project/Code](http://lear.inrialpes.fr/src/deepflow/))

### Image Processing
* Learning Recursive Filter for Low-Level Vision via a Hybrid Neural Network ([PDF](http://faculty.ucmerced.edu/mhyang/papers/eccv16_rnn_filter.pdf), [Project/Code](https://github.com/Liusifei/caffe-lowlevel))
* Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding([PDF](https://arxiv.org/abs/1510.00149), [Project/Code](https://github.com/songhan/Deep-Compression-AlexNet))
* A Learned Representation For Artistic Style([PDF](https://arxiv.org/abs/1610.07629))
* Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification ([PDF](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/data/colorization_sig2016.pdf), [Project/Code](https://github.com/satoshiiizuka/siggraph2016_colorization))
* Pixel Recurrent Neural Networks ([PDF](https://arxiv.org/abs/1601.06759))
* Conditional Image Generation with PixelCNN Decoders ([PDF](https://arxiv.org/abs/1606.05328), [Project/Code](https://github.com/dritchie/pixelCNN))
* RAISR: Rapid and Accurate Image Super Resolution ([PDF](https://arxiv.org/abs/1606.01299))

### CNN and Deep Learning ###
* UberNet: Training a `Universal' Convolutional Neural Network for Low-, Mid-, and High-Level Vision using Diverse Datasets and Limited Memory ([PDF](http://arxiv.org/abs/1609.02132), [Project/Code](http://cvn.ecp.fr/ubernet/))
* What makes ImageNet good for transfer learning? ([PDF](http://arxiv.org/abs/1608.08614), [Project/Code](http://minyounghuh.com/papers/analysis/), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-what-makes-imagenet-good-for-transfer-learning_138/))

    >The tremendous success of features learnt using the ImageNet classification task on a wide range of transfer tasks begs the question: what are the intrinsic properties of the ImageNet dataset that are critical for learning good, general-purpose features? This work provides an empirical investigation of various facets of this question: Is more pre-training data always better? How does feature quality depend on the number of training examples per class? Does adding more object classes improve performance? For the same data budget, how should the data be split into classes? Is fine-grained recognition necessary for learning good features? Given the same number of training classes, is it better to have coarse classes or fine-grained classes? Which is better: more classes or more examples per class?

* Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units ([PDF](http://arxiv.org/abs/1603.05201))
* Densely Connected Convolutional Networks ([PDF](http://arxiv.org/abs/1608.06993), [Project/Code](https://github.com/liuzhuang13/DenseNet))
* Decoupled Neural Interfaces using Synthetic Gradients ([PDF](https://arxiv.org/pdf/1608.05343.pdf)) 

    >Training directed neural networks typically requires forward-propagating data through a computation graph, followed by backpropagating error signal, to produce weight updates. All layers, or more generally, modules, of the network are therefore locked, in the sense that they must wait for the remainder of the network to execute forwards and propagate error backwards before they can be updated. In this work we break this constraint by decoupling modules by introducing a model of the future computation of the network graph.  These models predict what the result of the modeled sub-graph will produce using only local information. In particular we focus on modeling error gradients: by using the modeled synthetic gradient in place of true backpropagated error gradients we decouple subgraphs, and can update them independently and asynchronously.
    
* Rethinking the Inception Architecture for Computer Vision ([PDF](http://arxiv.org/abs/1512.00567), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-rethinking-the-inception-architecture-for-computer-vision_136/))

    In this paper, several network designing choices are discussed, including *factorizing convolutions into smaller kernels and asymmetric kernels*, *utility of auxiliary classifiers* and *reducing grid size using convolution stride rather than pooling*.

* Factorized Convolutional Neural Networks ([PDF](http://arxiv.org/abs/1608.04337), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-factorized-convolutional-neural-networks_133/))
* Do semantic parts emerge in Convolutional Neural Networks? ([PDF](http://arxiv.org/abs/1607.03738), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-do-semantic-parts-emerge-in-convolutional-neural-networks_131/))
* A Critical Review of Recurrent Neural Networks for Sequence Learning ([PDF](https://arxiv.org/abs/1506.00019))
* Image Compression with Neural Networks ([Project/Code](https://github.com/tensorflow/models/tree/master/compression))
* Graph Convolutional Networks ([Project/Code](http://tkipf.github.io/graph-convolutional-networks/))
* Understanding intermediate layers using linear classifier probes ([PDF](https://arxiv.org/abs/1610.01644), [Reading Note](http://joshua881228.webfactional.com/blog_reading-note-understanding-intermediate-layers-using-linear-classifier-probes_156/))
* Learning What and Where to Draw ([PDF](http://www.scottreed.info/files/nips2016.pdf), [Project/Code](https://github.com/reedscot/nips2016))
* On the interplay of network structure and gradient convergence in deep  learning ([PDF](https://arxiv.org/abs/1511.05297))
* Deep Learning with Separable Convolutions ([PDF](https://arxiv.org/abs/1610.02357))
* Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization ([PDF](https://arxiv.org/abs/1610.02391), [Project/Code](https://github.com/ramprs/grad-cam/))
* Optimization of Convolutional Neural Network using Microcanonical Annealing Algorithm ([PDF](https://arxiv.org/abs/1610.02306))
* Deep Pyramidal Residual Networks ([PDF](https://arxiv.org/abs/1610.02915))
* Impatient DNNs - Deep Neural Networks with Dynamic Time Budgets ([PDF](https://arxiv.org/abs/1610.02850))
* Uncertainty in Deep Learning ([PDF](http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf), [Project/Code](http://mlg.eng.cam.ac.uk/yarin/blog_2248.html))
	This is the PhD Thesis of Yarin Gal.
* Tensorial Mixture Models ([PDF](https://arxiv.org/abs/1610.04167), [Project/Code](https://github.com/HUJI-Deep/TMM))
* Multifaceted Feature Visualization: Uncovering the Different Types of Features Learned By Each Neuron in Deep Neural Networks ([PDF](https://arxiv.org/abs/1602.03616))
* Why Deep Neural Networks? ([PDF](https://arxiv.org/abs/1610.04161))
* Local Similarity-Aware Deep Feature Embedding ([PDF](https://arxiv.org/abs/1610.08904))
* A Review of 40 Years of Cognitive Architecture Research: Focus on Perception, Attention, Learning and Applications ([PDF](https://arxiv.org/abs/1610.08602))
* Professor Forcing: A New Algorithm for Training Recurrent Networks ([PDF](https://arxiv.org/abs/1610.09038))
* On the expressive power of deep neural networks([PDF](https://arxiv.org/abs/1606.05336))
* What Is the Best Practice for CNNs Applied to Visual Instance Retrieval? ([PDF](https://arxiv.org/abs/1611.01640))
* Deep Convolutional Neural Network Design Patterns ([PDF](https://arxiv.org/abs/1611.00847))
* Tricks from Deep Learning ([PDF](https://arxiv.org/abs/1611.03777))
* A Connection between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models ([PDF](https://arxiv.org/abs/1611.03852))
* Multi-Shot Mining Semantic Part Concepts in CNNs ([PDF](https://arxiv.org/abs/1611.04246))
* Aggregated Residual Transformations for Deep Neural Networks ([PDF](https://arxiv.org/abs/1611.05431))

### Video ###

* Video Pixel Networks ([PDF](https://arxiv.org/abs/1610.00527))
* Plug-and-Play CNN for Crowd Motion Analysis: An Application in Abnormal Event Detection ([PDF](https://arxiv.org/abs/1610.00307))
* EM-Based Mixture Models Applied to Video Event Detection ([PDF](https://arxiv.org/abs/1610.02923))
* Video Captioning and Retrieval Models with Semantic Attention ([PDF](https://arxiv.org/abs/1610.02947))
* Title Generation for User Generated Videos ([PDF](https://arxiv.org/abs/1608.07068))
* Review of Action Recognition and Detection Methods ([PDF](https://arxiv.org/abs/1610.06906))
* RECURRENT MIXTURE DENSITY NETWORK FOR SPATIOTEMPORAL VISUAL ATTENTION ([PDF](http://openreview.net/pdf?id=SJRpRfKxx))


### Machine Learning ###
* [计算机视觉与机器学习 【随机森林】](http://joshua881228.webfactional.com/blog_ji-suan-ji-shi-jue-yu-ji-qi-xue-xi-sui-ji-sen-lin_129/)
* [计算机视觉与机器学习 【深度学习中的激活函数】](http://joshua881228.webfactional.com/blog_ji-suan-ji-shi-jue-yu-ji-qi-xue-xi-shen-du-xue-xi-zhong-de-ji-huo-han-shu_128/)
* [我爱机器学习](https://www.52ml.net/) 机器学习干货站

### Embedded ###

* Caffeinated FPGAs: FPGA Framework For Convolutional Neural Networks ([PDF](https://arxiv.org/abs/1609.09671))
* Comprehensive Evaluation of OpenCL-based Convolutional Neural Network Accelerators in Xilinx and Altera FPGAs  ([PDF](https://arxiv.org/abs/1609.09296))

### Other ###

* Learning Aligned Cross-Modal Representations from Weakly Aligned Data ([PDF](http://cmplaces.csail.mit.edu/content/paper.pdf), [Project/Code](http://cmplaces.csail.mit.edu/))
* Multi-Task Curriculum Transfer Deep Learning of Clothing Attributes ([PDF](https://arxiv.org/abs/1610.03670))
* End-to-end Learning of Deep Visual Representations for Image Retrieval ([PDF](https://arxiv.org/abs/1610.07940))
* SoundNet: Learning Sound Representations from Unlabeled Video ([PDF](http://web.mit.edu/vondrick/soundnet.pdf))
* Bags of Local Convolutional Features for Scalable Instance Search ([PDF](https://arxiv.org/abs/1604.04653), [Project/Code](https://imatge-upc.github.io/retrieval-2016-icmr/))
* Universal Correspondence Network ([PDF](http://cvgl.stanford.edu/projects/ucn/choy_nips16.pdf), [Project/Code](http://cvgl.stanford.edu/projects/ucn/))
* Judging a Book By its Cover ([PDF](https://arxiv.org/abs/1610.09204))

## Interesting Finds ##

### Resources/Perspectives ###

* [arXiv(Computer Vision and Pattern Recognition)](http://arxiv.org/list/cs.CV/recent) 
    A good place to explore latest papers.
* [Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision)
    A curated list of awesome computer vision resources.
* [Awesome Deep Vision](https://github.com/kjw0612/awesome-deep-vision)
    A curated list of deep learning resources for computer vision.
* [Awesome MXNet](https://github.com/dmlc/mxnet/blob/master/example/README.md)
    This page contains a curated list of awesome MXnet examples, tutorials and blogs. 
* [Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow)
    A curated list of awesome TensorFlow experiments, libraries, and projects.
* [Deep Reinforcement Learning survey](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey)
    This paper list is a bit different from others. The author puts some opinion and summary on it. However, to understand the whole paper, you still have to read it by yourself!
* [TensorFlow 官方文档中文版](https://github.com/jikexueyuanwiki/tensorflow-zh)
* [TensorTalk](https://tensortalk.com/)
    A place to find latest work's codes.
* [OTB Results](https://github.com/foolwood/benchmark_results)
    Object tracking benchmark

### Projects ###

* [TensorFlow Examples](https://github.com/aymericdamien/TensorFlow-Examples)
    TensorFlow Tutorial with popular machine learning algorithms implementation. This tutorial was designed for easily diving into TensorFlow, through examples.It is suitable for beginners who want to find clear and concise examples about TensorFlow. For readability, the tutorial includes both notebook and code with explanations.
* [TensorFlow Tutorials](https://github.com/Hvass-Labs/TensorFlow-Tutorials)
    These tutorials are intended for beginners in Deep Learning and TensorFlow. Each tutorial covers a single topic. The source-code is well-documented. There is a YouTube video for each tutorial.
* [Home Surveilance with Facial Recognition](https://github.com/BrandonJoffe/home_surveillance)
* [Deep Learning algorithms with TensorFlow](https://github.com/blackecho/Deep-Learning-TensorFlow)
	This repository is a collection of various Deep Learning algorithms implemented using the TensorFlow library. This package is intended as a command line utility you can use to quickly train and evaluate popular Deep Learning models and maybe use them as benchmark/baseline in comparison to your custom models/datasets. 
* [TensorLayer](https://github.com/zsdonghao/tensorlayer) 
    TensorLayer is designed to use by both Researchers and Engineers, it is a transparent library built on the top of Google TensorFlow. It is designed to provide a higher-level API to TensorFlow in order to speed-up experimentations and developments. TensorLayer is easy to be extended and modified. In addition, we provide many examples and tutorials to help you to go through deep learning and reinforcement learning.
* [Easily Create High Quality Object Detectors with Deep Learning](http://blog.dlib.net/2016/10/easily-create-high-quality-object.html)
	Using [dlib](http://dlib.net/) to train a CNN to detect.
* [Command Line Neural Network](https://github.com/hugorut/neural-cli)
	Neuralcli provides a simple command line interface to a python implementation of a simple classification neural network. Neuralcli allows a quick way and easy to get instant feedback on a hypothesis or to play around with one of the most popular concepts in machine learning today.
* [LSTM for Human Activity Recognition](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/)
	Human activity recognition using smartphones dataset and an LSTM RNN. The project is based on Tesorflow. A MXNet implementation is [MXNET-Scala Human Activity Recognition](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/HumanActivityRecognition).
* [YOLO in caffe](https://github.com/xingwangsfu/caffe-yolo)
    This is a caffe implementation of the YOLO:Real-Time Object Detection.
* [SSD: Single Shot MultiBox Object Detector in mxnet](https://github.com/zhreshold/mxnet-ssd)
* [MTCNN face detection and alignment in MXNet](https://github.com/pangyupo/mxnet_mtcnn_face_detection)
	This is a python/mxnet implementation of [Zhang's](https://github.com/kpzhang93/MTCNN_face_detection_alignment) work .
* [CNTK Examples: Image/Detection/Fast R-CNN](https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Detection/FastRCNN)
* [Self Driving (Toy) Ferrari](https://github.com/RyanZotti/Self-Driving-Car)
* [Finding Lane Lines on the Road](https://github.com/udacity/CarND-LaneLines-P1)
* [Magenta](https://github.com/tensorflow/magenta)
	Magenta is a project from the Google Brain team that asks: Can we use machine learning to create compelling art and music? If so, how? If not, why not?
* [Adversarial Nets Papers](https://github.com/zhangqianhui/AdversarialNetsPapers)
	The classical Papers about adversarial nets 
* [Mushreco](http://mushreco.ml/en/)
	Make a photo of a mushroom and see which species it is. Determine over 200 different species. 

### News/Blogs ###
* [MIT Technology Review](https://www.technologyreview.com/)
    A good place to keep up the trends.
* [LAB41](https://gab41.lab41.org/)
    Lab41 is a Silicon Valley challenge lab where experts from the U.S. Intelligence Community (IC), academia, industry, and In-Q-Tel come together to gain a better understanding of how to work with — and ultimately use — big data. 
* [Partnership on AI](http://www.partnershiponai.org/)
    Amazon, DeepMind/Google, Facebook, IBM, and Microsoft announced that they will create a non-profit organization that will work to advance public understanding of artificial intelligence technologies (AI) and formulate best practices on the challenges and opportunities within the field. Academics, non-profits, and specialists in policy and ethics will be invited to join the Board of the organization, named the Partnership on Artificial Intelligence to Benefit People and Society (Partnership on AI).
* [爱可可-爱生活](http://weibo.com/fly51fly?from=profile&wvr=6&is_all=1) 老师的推荐十分值得一看
* [Guide to deploying deep-learning inference networks and realtime object recognition tutorial for NVIDIA Jetson TX1](https://github.com/dusty-nv/jetson-inference)
* [A Return to Machine Learning](https://medium.com/@kcimc/a-return-to-machine-learning-2de3728558eb)
    This post is aimed at artists and other creative people who are interested in a survey of recent developments in machine learning research that intersect with art and culture. If you’ve been following ML research recently, you might find some of the experiments interesting but will want to skip most of the explanations.
* [ResNets, HighwayNets, and DenseNets, Oh My!](https://medium.com/@awjuliani/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32#.1d3mpy4hd)
	This post walks through the logic behind three recent deep learning architectures: ResNet, HighwayNet, and DenseNet. Each make it more possible to successfully trainable deep networks by overcoming the limitations of traditional network design.
* [How to build a robot that “sees” with $100 and TensorFlow](https://www.oreilly.com/learning/how-to-build-a-robot-that-sees-with-100-and-tensorflow)
	
	>I wanted to build a robot that could recognize objects. Years of experience building computer programs and doing test-driven development have turned me into a menace working on physical projects. In the real world, testing your buggy device can burn down your house, or at least fry your motor and force you to wait a couple of days for replacement parts to arrive.

* [Navigating the unsupervised learning landscape](https://culurciello.github.io//tech/2016/06/10/unsup.html)
	Unsupervised learning is the Holy Grail of Deep Learning. The goal of unsupervised learning is to create general systems that can be trained with little data. Very little data.
* [Deconvolution and Checkerboard Artifacts](http://distill.pub/2016/deconv-checkerboard/)
* [Facial Recognition on a Jetson TX1 in Tensorflow](http://www.mattkrzus.com/face.html)
	Here's a way to hack facial recognition system together in relatively short time on NVIDIA's Jetson TX1.
* [Deep Learning with Generative and Generative Adverserial Networks – ICLR 2017 Discoveries](https://amundtveit.com/2016/11/12/deep-learning-with-generative-and-generative-adverserial-networks-iclr-2017-discoveries/)
	This blog post gives an overview of Deep Learning with Generative and Adverserial Networks related papers submitted to ICLR 2017.
* [Unsupervised Deep Learning – ICLR 2017 Discoveries](https://amundtveit.com/2016/11/12/unsupervised-deep-learning-iclr-2017-discoveries/)
	This blog post gives an overview of papers related to Unsupervised Deep Learning submitted to ICLR 2017.
	
### Benchmark/Leaderboard/Dataset ###
* [Visual Tracker Benchmark](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html)
    This website contains data and code of the benchmark evaluation of online visual tracking algorithms. Join visual-tracking Google groups for further updates, discussions, or QnAs.
* [Multiple Object Tracking Benchmark](https://motchallenge.net/)
	With this benchmark we would like to pave the way for a unified framework towards more meaningful quantification of multi-target tracking.
* [Leaderboards for the Evaluations on PASCAL VOC Data](http://host.robots.ox.ac.uk:8080/leaderboard/main_bootstrap.php)
* [Open Images dataset](https://github.com/openimages/dataset)
    Open Images is a dataset of ~9 million URLs to images that have been annotated with labels spanning over 6000 categories.
* [Open Sourcing 223GB of Driving Data](https://github.com/udacity/self-driving-car)
	223GB of image frames and log data from 70 minutes of driving in Mountain View on two separate days, with one day being sunny, and the other overcast. 
* [MS COCO](http://mscoco.org)
* [UMDFaces Dataset](http://umdfaces.io/)
	UMDFaces is a face dataset which has 367,920 faces of 8,501 subjects. 

From this page you can download the entire dataset and the trained model for predicting the localization of the 21 keypoints. 

### Toolkits ###
* [Caffe](http://caffe.berkeleyvision.org/)
    Caffe is a deep learning framework made with expression, speed, and modularity in mind. It is developed by the Berkeley Vision and Learning Center (BVLC) and by community contributors. Yangqing Jia created the project during his PhD at UC Berkeley. Caffe is released under the BSD 2-Clause license.
* [Caffe on Intel](https://github.com/intel/caffe)
    This fork of BVLC/Caffe is dedicated to improving performance of this deep learning framework when running on CPU, in particular Intel® Xeon processors (HSW+) and Intel® Xeon Phi processors
* [TensorFlow](https://github.com/tensorflow/tensorflow)
    TensorFlow is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) that flow between them. This flexible architecture lets you deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device without rewriting code. TensorFlow also includes TensorBoard, a data visualization toolkit.
* [MXNet](http://mxnet.io/)
    MXNet is a deep learning framework designed for both efficiency and flexibility. It allows you to mix the flavours of symbolic programming and imperative programming to maximize efficiency and productivity. In its core, a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly. A graph optimization layer on top of that makes symbolic execution fast and memory efficient. The library is portable and lightweight, and it scales to multiple GPUs and multiple machines.
* [neon](https://github.com/NervanaSystems/neon)
    neon is Nervana's Python based Deep Learning framework and achieves the fastest performance on modern deep neural networks such as AlexNet, VGG and GoogLeNet. Designed for ease-of-use and extensibility.
* [Piotr's Computer Vision Matlab Toolbox](https://pdollar.github.io/toolbox/)
    This toolbox is meant to facilitate the manipulation of images and video in Matlab. Its purpose is to complement, not replace, Matlab's Image Processing Toolbox, and in fact it requires that the Matlab Image Toolbox be installed. Emphasis has been placed on code efficiency and code reuse. Thanks to everyone who has given me feedback - you've helped make this toolbox more useful and easier to use. 
* [NVIDIA Developer](https://developer.nvidia.com/)
* [nvCaffe](https://github.com/NVIDIA/caffe/tree/experimental/fp16)
    A special branch of caffe is used on TX1 which includes support for FP16.
* [dlib](http://dlib.net/)
    Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems. It is used in both industry and academia in a wide range of domains including robotics, embedded devices, mobile phones, and large high performance computing environments. Dlib's open source licensing allows you to use it in any application, free of charge. 
* [OpenCV](http://opencv.org/)
    OpenCV is released under a BSD license and hence it’s free for both academic and commercial use. It has C++, C, Python and Java interfaces and supports Windows, Linux, Mac OS, iOS and Android. OpenCV was designed for computational efficiency and with a strong focus on real-time applications.
* [CNNdroid](https://github.com/ENCP/CNNdroid)
	CNNdroid is an open source library for execution of trained convolutional neural networks on Android devices. 

### Learning/Tricks ###
* [Backpropagation Algorithm](http://deeplearning.stanford.edu/wiki/index.php/Backpropagation_Algorithm)
    A website that explain how Backpropagation Algorithm works.
* [Deep Learning ( textbook authored by Ian Goodfellow and Yoshua Bengio and Aaron Courville)](http://www.deeplearningbook.org/)
    The Deep Learning textbook is a resource intended to help students and practitioners enter the field of machine learning in general and deep learning in particular. 
* [Neural Networks and Deep Learning (online book authored by Michael Nielsen)](http://neuralnetworksanddeeplearning.com/index.html)
    Neural Networks and Deep Learning is a free online book. The book will teach you about 1) Neural networks, a beautiful biologically-inspired programming paradigm which enables a computer to learn from observational data and 2) Deep learning, a powerful set of techniques for learning in neural networks. Neural networks and deep learning currently provide the best solutions to many problems in image recognition, speech recognition, and natural language processing. This book will teach you many of the core concepts behind neural networks and deep learning.
* [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/)
    This book is largely based on the computer vision courses that Richard Szeliski has co-taught at the University of Washington (2008, 2005, 2001) and Stanford (2003) with Steve Seitz and David Fleet.
* [Must Know Tips/Tricks in Deep Neural Networks ](http://210.28.132.67/weixs/project/CNNTricks/CNNTricks.html)
    Many implementation details for DCNNs are collected and concluded. Extensive implementation details are introduced, i.e., tricks or tips, for building and training your own deep networks.
* [The zen of gradient descent](http://blog.mrtz.org/2013/09/07/the-zen-of-gradient-descent.html)
* [Deriving the Gradient for the Backward Pass of Batch Normalization](https://kevinzakka.github.io/2016/09/14/batch_normalization/)
* [Reinforcement Learning: An Introduction](https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html)
* [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/)
* [Regularizing neural networks by penalizing confident predictions](https://pan.baidu.com/s/1kUUtxdl)
* [What you need to know about data augmentation for machine learning](https://cartesianfaith.com/2016/10/06/what-you-need-to-know-about-data-augmentation-for-machine-learning/)
    Plentiful high-quality data is the key to great machine learning models. But good data doesn’t grow on trees, and that scarcity can impede the development of a model. One way to get around a lack of data is to augment your dataset. Smart approaches to programmatic data augmentation can increase the size of your training set 10-fold or more. Even better, your model will often be more robust (and prevent overfitting) and can even be simpler due to a better training set.
* [Guide to deploying deep-learning inference networks and realtime object recognition tutorial for NVIDIA Jetson TX1]
* [The Effect of Resolution on Deep Neural Network Image Classification Accuracy](https://medium.com/the-downlinq/the-effect-of-resolution-on-deep-neural-network-image-classification-accuracy-d1338e2782c5#.5iwdz2te8)
	The author explored the impact of both spatial resolution and training dataset size on the classification performance of deep neural networks in this post.
* [深度学习调参的技巧](https://www.zhihu.com/question/25097993)
* [CNN怎么调参数](https://www.zhihu.com/question/27962483)
* [视频多目标跟踪当前（2014,2015,2016）比较好的算法有哪些](https://www.zhihu.com/question/40545681)
* [5 algorithms to train a neural network](https://www.neuraldesigner.com/blog/5_algorithms_to_train_a_neural_network.html)
* [Towards Good Practices for Recognition & Detection](http://mp.weixin.qq.com/s?__biz=MzI1NTE4NTUwOQ==&mid=2650325586&idx=1&sn=69a7e8482d884dda869290581a50a6d6&chksm=f235a558c5422c4eabe75f6db92fd13a3ca662d648676461914c9bfd1f4e22affe12430afce3&mpshare=1&scene=2&srcid=1024IdcoRukOMBvDA5dPbRoV&from=timeline&isappinstalled=0#wechat_redirect)
	海康威视研究院ImageNet2016竞赛经验分享
* [What are the differences between Random Forest and Gradient Tree Boosting algorithms](https://www.quora.com/What-are-the-differences-between-Random-Forest-and-Gradient-Tree-Boosting-algorithms)
* [为什么现在的CNN模型都是在GoogleNet、VGGNet或者AlexNet上调整的](https://www.zhihu.com/question/43370067)

## Skills ##

### About Caffe ###
* [Set Up Caffe on Ubuntu14.04 64bit+NVIDIA GTX970M+CUDA7.0](http://joshua881228.webfactional.com/blog_set-up-caffe-on-ubuntu1404-64bitnvidia-gtx970mcuda70_55/)
* [VS2013配置Caffe卷积神经网络工具（64位Windows 7）——建立工程](http://blog.csdn.net/joshua_1988/article/details/45048871)
* [VS2013配置Caffe卷积神经网络工具（64位Windows 7）——准备依赖库](http://blog.csdn.net/joshua_1988/article/details/45036993)

### Setting Up ###
* [Installation of NVIDIA GPU Driver and CUDA Toolkit](http://joshua881228.webfactional.com/blog_installation-of-nvidia-gpu-driver-and-cuda-toolkit_54/)
* [Tensorflow v0.10 installed from scratch on Ubuntu 16.04, CUDA 8.0RC+Patch, cuDNN v5.1 with a 1080GTX](https://marcnu.github.io/2016-08-17/Tensorflow-v0.10-installed-from-scratch-Ubuntu-16.04-CUDA8.0RC-cuDNN5.1-1080GTX/)
