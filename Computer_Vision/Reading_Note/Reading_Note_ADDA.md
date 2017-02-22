**TITLE**: Adversarial Discriminative Domain Adaptation

**AUTHOR**: Eric Tzeng, Judy Hoffman, Kate Saenko, Trevor Darrell

**ASSOCIATION**: UC Berkeley, Stanford University, Boston University

**FROM**: [arXiv:1702.05464](https://arxiv.org/abs/1702.05464)

## CONTRIBUTIONS ##

1. A novel unified framework for adversarial domain adaptation is proposed, which is called Adversarial Discriminative Domain Adaptation (ADDA).
2. Design choices such as weight-sharing, base models, and adversarial losses and subsumes previous work, are unified.


## METHOD ##

The main idea of this work is to find a map function project target data, the data for testing, to the source data domain, that are used for training. The training procedure is illustrated in the following figure.

<img class="img-responsive center-block" src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/ADDA_1.jpg" alt="" width="640"/>

As the figure shows, first pre-train a source encoder CNN using labeled source image examples. Next, perform adversarial adaptation by learning a target encoder CNN such that a discriminator that sees encoded source and target examples cannot reliably predict their domain label. During testing, target images are mapped with the target encoder to the shared feature space and classified by the source classifier. Dashed lines indicate fixed network parameters.

The ADDA method can be formalized as:

$$ \min \limits_{M_{s}, C} \mathcal{L}_{cls}(\mathbf{X_{s}}, \mathbf{Y_{s}}) = 
-\mathbb{E}_{(\mathbf{x}_{s}, y_{s})\sim(\mathbf{X}_{s}, \mathbf{Y}_{s})}[\sum_{k=1}^{K} \mathbf{1}_{[k=y_{s}]} \log C(M_s(\mathbf{x_s}))]$$

$$ \min \limits_{D} \mathcal{L}_{ \mathbf{adv}_D}(\mathbf{X_{s}}, \mathbf{X_{t}}, \mathbf{M_{s}}, \mathbf{M_{t}}) = 
-\mathbb{E}_{\mathbf{x}_{s}\sim \mathbf{X}_{s}} [\log D(M_s(\mathbf{x_s}))]
-\mathbb{E}_{\mathbf{x}_{t}\sim \mathbf{X}_{t}} [\log (1-D(M_t(\mathbf{x_t})))]$$

$$ \min \limits_{M_s,M_t} \mathcal{L}_{ \mathbf{adv}_M}(\mathbf{X_{s}}, \mathbf{X_{t}}, D) = 
-\mathbb{E}_{\mathbf{x}_{s}\sim \mathbf{X}_{t}} [\log D(M_t(\mathbf{x_t}))] $$

The first formula is a typical supervised learning. The second formula is just like what has been proposed in GAN. It is used to learn a discriminator to tell target data from source data. The first term can be ignored because $M_s$ is fixed. The third formula is used to learn a $M_t$ mapping data from targe domain to source domain.
