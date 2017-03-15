**TITLE**: A Pursuit of Temporal Accuracy in General Activity Detection

**AUTHOR**: Yuanjun Xiong, Yue Zhao, Limin Wang, Dahua Lin, Xiaoou Tang

**ASSOCIATION**: The Chinese University of Hong Kong, ETH

**FROM**: [arXiv:1703.02716](https://arxiv.org/abs/1703.02716)

## CONTRIBUTIONS ##

1. A novel proposal scheme is proposed that can efficiently generate candidates with accurate temporal boundaries.
2. A cascaded classification pipeline is introduced that explicitly distinguishes between relevance and completeness of a candidate instance. 

## METHOD ##

The proposed action detection framework starts with evaluating the actionness of the snippets of the video. A set of temporal action proposals (in orange color) are generated with temporal actionness grouping (TAG). The proposals are evaluated against the cascaded classifiers to verify their relevance and completeness. Only proposals being complete instances are produced by the framework. Non-complete proposals and background proposals are rejected by a cascaded classification pipeline. The framework is illustrated in the following figure.

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/Reading_Note_20170315_TAG_0.jpg" alt="" width="640"/>

###Temporal Region proposals###

The temporal region proposals are generated with a bottom-up procedure, which consists of three steps: extract snippets, evaluate snippet-wise actionness, and finally group them into region proposals. 

1. To evaluate the actionness, a binary classifier is learnt based on the Temporal Segment Network proposed in *Temporal segment networks: Towards good practices for deep action recognition*.
2. To generate temporal region proposals, the basic idea is to group consecutive snippets with high actionness scores. The scheme first obtains a number of action fragments by thresholding – a fragment here is a consecutive sub-sequence of snippets whose actionness scores are above a certain threshold, referred to as **actionness threshold**. 
3. Then, to generate a region proposal, a fragment is picked as a starting point and expanded recursively by absorbing succeeding fragments. The expansion terminates when the portion of low-actionness snippets goes beyond a threshold, a positive value which is referred to as the **tolerance threshold**. Beginning with different fragments, we can obtain a collection of different region proposals.

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/Reading_Note_20170315_TAG_1.jpg" alt="" width="480"/>

Note that this scheme is controlled by two design parameters: the **actionness threshold** and the **tolerance threshold**. The final proposal set is the union of those derived from individual combination of the two values. This scheme is called *Temporal Actionness Grouping*, illustrated in the above figure, which has several advantages:

1. Thanks to the actionness classifier, the generated proposals are mostly focused on action-related contents, which greatly reduce the number of needed proposals. 
2. Action fragments are sensitive to temporal transitions. Hence, as a bottom-up method that relies on merging action fragments, it often yields proposals with more accurate temporal boundaries.
3. With the multi-threshold design, it can cover a broad range of actions without the need of case-specific parameter tuning. With these properties, the proposed method can achieve high recall with just a moderate number of proposals. This also benefits the training of the classifiers in the next stage.

###Detecting Action Instances###

this is accomplished by a cascaded pipeline with two steps: *activity classification* and *completeness filtering*.

**Activity Classification**

A classifier is trained based on TSN. During training, region proposals that overlap with a ground-truth instance with an IOU above 0.7 will be used as positive samples. A proposal is considered as a negative sample only when less than 5% of its time span overlaps with any annotated instances. Only the proposals classified as non-background classes will be retained for completeness filtering. The probability from the activity classifier is denoted as $P_{a}$.

**Completeness Filtering**

To evaluate the completeness, a simple feature representation is extracted and used to train class-specific SVMs. The feature comprises three parts: (1) A temporal pyramid of two levels. The first level pools the snippet scores within the proposed region. The second level split the segment into two parts and pool the snippet scores inside each part. (2) The average classification scores of two short periods – the ones before and after the proposed region. The method is illustrated in the following figure.

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/Reading_Note_20170315_TAG_2.jpg" alt="" width="480"/>

The output of the SVMs for one class is denoted as $S_{c}$.

Then final detection confidence for each proposal is 

$$ S_{Det} = P_{a} \times S_{c} $$