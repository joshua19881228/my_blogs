**TITLE**: Optimizing Deep CNN-Based Queries over Video Streams at Scale

**AUTHOR**: Daniel Kang, John Emmons, Firas Abuzaid, Peter Bailis, Matei Zaharia

**ASSOCIATION**: Stanford InfoLab

**FROM**: [arXiv:1703.02529](https://arxiv.org/abs/1703.02529)

## CONTRIBUTIONS ##

1. NOSCOPE, the first data management system that accelerates CNN-based classification queries over video streams at scale.
2. CNN-specific techniques for difference detection across frames and model specialization for a given stream and query, as well as a cost-based optimizer that can automatically identify the best combination of these filters for a given accuracy target.
3. An evaluation of NOSCOPE on fixed-angle binary classification showing up to 3,200x speedups on real-world data.

## METHOD ##

The work flow of NoScope can be viewed in the following figure. Brefiely, it can be explained that NoScope's optimizer selects a different configuration of difference detectors and specialized models for each video stream to perform binary classification as quickly as possible without calling the full target CNN, which will be called only when necessary.

![Overall Framework of NoScope](https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/Reading_Note_20170710_NoScope.png "Overall Framework of NoScope =480")

There are mainly three compoments in this system, Difference Detectors, Specialized Models and Cost-based Optimizer.

1. *Difference Detectors* consider attempts to detect differences between images. They are used to determine whether the considered frame is significantly different from another image with known labels. There are two forms of difference detectors supported: difference detection against a fixed reference image for the video stream that is known to contain no objects and difference detection against an earlier frame, some configured time into the past.
2. *Specialized Models* are small CNNs specified for each video and query. They are designed using different combinations of numbers of channels and layers. This can be thought as expert classifiers or detectors for different videos. For static cameras, one specifialized model does not need to consider samples that would only appear in other camers.
3. *Cost-based Optimizer* brings difference detectors and model specialization together that maximizes the throughput subject to a certain condition, e.g. FP and FN rate.

## DISADVANTAGES ##

1. This scheme is suitable for fixed views, but if the input changes frequently, this scheme may work less efficiently or effectively.