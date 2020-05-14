## Landscape

Causality spans a broad area of topics, including using causal insights to improve machine learning methods, adapting it for high-dimensional datasets and applying them for better data-driven decision making in real-world contexts. We also discussed in Chapter 3 how the collected data is rarely an accurate reflection of the population, and hence may fail to generalize in different environments or new datasets. Methods based on invariance show promise in addressing out-of-distribution generalization.

### Use Cases

As we demonstrated in the [Prototype](#prototype) Chapter, Invariant Risk Minimization is particularly well suited to image problems in diverse physical environments. However, an environment need not mean only the scenery in an image, and when it does, it need not be fixed to a single value. Here we suggest some applications in and beyond computer vision.

#### Healthcare

In clinical imaging settings, radiologists manually annotate tissues, abnormalities, and pathologies of patients. Biomedical engineers then use these annotations to train systems to perform automatic tissue segmentation or pathology detection in medical images. Suppose a hospital installs a new MRI scanner. Unfortunately, due to the mechanical configuration, calibration, vendor and acquisition protocol of the scanner, the images it produces will differ from images produced by other scanners. Consequently, systems trained on data from other scanners would fail to perform well on the new scanner.

An algorithmic system based on invariant prediction that treats scanners as environments could find correspondences in images between scanners, and change its decisions accordingly. This could save the time, funds and energy needed to annotate images from the new scanner. ^[[Transfer Learning Improves Supervised Image Segmentation Across Imaging Protocols](https://ieeexplore.ieee.org/document/6945865)]

#### Robotics

[figure: robot]

Autonomous systems need to detect and adapt to different environments. For instance, autonomously following a man-made trail such as those normally traversed by hikers or mountain-bikers is a challenging and mostly unsolved task for robotics.^[[A Machine Learning Approach to Visual Perception of Forest Trails for Mobile Robots](http://rpg.ifi.uzh.ch/docs/RAL16_Giusti.pdf)] Solving such a problem is important for many applications, including wilderness mapping and search and rescue. Moreover, following a trail would be the most efficient and safest way for a ground robot to travel medium and long distances in a forested environment: by their nature, trails avoid excessive slopes and impassable ground due to excessive vegetation or wetlands. Many robot types, including wheeled, tracked and legged vehicles, are capable of locomotion along realworld trails.

In order to successfully follow a forest trail, a robot has to perceive where the trail is, then react in order to stay on the trail. Perceiving real-world trails in natural conditions is an extremely difficult and interesting pattern recognition problem, which is often challenging even for humans. Unpaved roads are normally much less structured than paved ones: their appearance is very variable based on the wilderness area and often boundaries are not well defined. It would be impossible to capture comprehensive data about all trails. Casting the trail perception problem as an image classification task and adopting a method based on invariance that operates directly on the image’s raw pixel values would allow for out-of-distribution generalization to new trails. Naturally, similar ideas are relevant for autonomous vehicles in urban areas.

#### Activity recognition systems

The diversity of sensors on personal devices, such as smartphones and smartwatches, has driven the development of novel activity recognition technologies, which help capture a person’s daily lifestyle activities and gestures in the physical world. These systems utilize features defined over data from sensors, such as accelerometer or gyroscope, and are trained using data labeled explicitly with tags such ‘sitting’, ‘walking’ and ‘climbing’. Such capabilities enable many new applications from wellness tracking to immersive gaming.

Unfortunately, it is hard to satisfactorily model this data due to the diversity exhibited in the real world. At times the users can exhibit significant differences in the way they perform the same daily lifestyle activities, while at other times these devices may be unusually placed or a user may switch from one device to another which might degrade the system’s performance due to instance-specific variations. This also means that we may either need a labelled dataset that captures the activity for each user and device (which is prohibitively expensive) or another way of identifying attributes that generalize better. Methods based on invariance could be particularly useful and well suited for this.

#### Natural language processing

[figure: different types of text]

Invariant prediction approaches are of course not restricted exclusively to image problems. In natural language processing, texts from different publication platforms are tricky to analyze due to different contexts, vocabularies and differences between how authors express themselves. For instance, financial news articles use a vocabulary and tone that differs from that used biomedical research abstracts. Similarly, online movie reviews are linguistically different from tweets. Sentiment classification also relies heavily on context; different words are used to express whether someone likes a book versus an electronic gadget.
Two recent papers, [An Empirical Study of Invariant Risk Minimization](https://arxiv.org/abs/2004.05007) and [Invariant Rationalization](https://arxiv.org/abs/2003.09772), apply the idea of IRM to sentiment classification task, and find it improves out of distribution generalization. In particular, invariance acts to remove spurious reliance on single words which correlate highly with the target. Like images, text corpora form very high dimensional datasets (there are many possible words!), making spurious correlations extremely unlikely to be noticed “manually”. As such, invariance based approaches are especially promising here.

#### Recommender systems

[figure: recommender system]

Making good recommendations is an important problem on the web. In the recommendation problem, we observe how a set of users interact with a set of items, and our goal is to show each user a set of previously unseen items that s/he will like. Broadly speaking, recommendation systems use historical data to infer users’ preferences, and then use the inferred preferences to suggest items.

Traditionally, they use click (or ratings) data alone to infer user preferences. Click data expresses a binary decision about items - for example this can be clicking, purchasing, viewing—and we aim to predict unclicked items that she would want to click on. But this inference is biased by the exposure data: users do not consider each item independently at random. At times this assumption is mistaken, and overestimates the effect of the unclicked items. Some of these items—many of them, in large-scale settings—are unclicked because the user didn’t see them, rather than because she chose not to click them.

ReySys are a classic application for causality, which allows us to correct for this exposure bias by treating the selection of items to present to a user as an intervention. Applying causal approaches to recommendation naturally improves generalization to new data,^[See [Causal Inference for Recommendation](http://www.its.caltech.edu/~fehardt/UAI2016WS/papers/Liang.pdf) and [The Deconfounded Recommender: A Causal Inference Approach to Recommendation](https://arxiv.org/abs/1808.06581).] and it seems likely that methods seeking invariant prediction could enhance this.

### Tools

The invariance-based approaches to causality we have discussed do not require dedicated tooling - ICP and IRM are procedures that could be implemented with general purpose machine learning frameworks.

Nonetheless, the authors of the ICP papers ^[[Causal inference using invariant prediction: identification and confidence intervals](https://arxiv.org/abs/1501.01332) and [Invariant Causal Prediction for Nonlinear Models](https://arxiv.org/abs/1706.08576).] provide corresponding R packages: [InvariantCausalPrediction](https://cran.r-project.org/web/packages/InvariantCausalPrediction/index.html) and [nonlinearICP](https://cran.r-project.org/web/packages/nonlinearICP/index.html). The packages make the techniques easy to use, and include additional utilities, such as dedicated plots for confidence intervals on causal coefficients. We are not aware of a package for IRM, but the authors have provided a [code repository](http://github.com/facebookresearch/InvariantRiskMinimization/) which reproduces the paper results.

Below, we list a handful of open source projects that aid in traditional, SCM-based causal inference.

#### DoWhy

Microsoft Research is developing the [DoWhy](https://microsoft.github.io/dowhy/) python library for causal inference, incorporating elements of both causal graphical models and potential outcomes. The library is oriented around pandas DataFrames, and as such fits easily into a Python data analysis workflow. In particular DoWhy makes a separation between four stages of causal inference:

1. Modeling - defining a causal graph, or else the assumptions necessary for a potential outcomes approach (the common causes of the treatment and the outcome variable).
2. Identification - identifying the expression it is necessary to evaluate, in terms of conditional probability distributions.
3. Estimation - estimating the treatment effect. There are many estimation methods available in DoWhy, including machine learning based methods from another of Microsoft's causal libraries: [EconML](https://github.com/microsoft/EconML)
4. Refutation - assessing the robustness of the conclusion. Given the reliance of causal inference on modeling assumptions, it is especially important to find ways to test our conclusions. DoWhy provides several methods for this, such as introducing a dummy common cause or replacing the treatment with a random placebo.

In addition to the above, DoWhy includes a novel algorithm, the "do-sampler." In much of causal inference, the quantity of interest is a single number, for instance, the difference in the outcome variable when a binary treatment variable is applied ("what is the average causal effect of smoking on cancer incidence?"). The do-sampler extends the pandas DataFrame API directly, and moves beyond calculating causal effects to allow sampling from the full interventional distribution. Having done so, we can then compute arbitrary statistics under this intervention. The do-sampler is new, but provides a very promising direction for further research, and a potential avenue to making causal inference accessible to many more data science practitioners.

#### CausalDiscoveryToolbox

The [Causal Discovery Toolbox](https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html) provides implementations of many algorithms designed for causal discovery - attempting to recover the full causal graph from observational data alone. There are many approaches to causal discovery, and the library is relatively comprehensive, including both algorithms pairwise causal discovery (inferring the direction of causation between a pair of variables), graph skeleton creation (creating an undirected graph of potential causal relationships), and full graphical causal model discovery.

Discovery of entire causal graphs does not yet appear mature enough that we can naively trust it’s conclusions about the causal structure of a problem. This makes sense given the difficulty of the task! Inferring the whole causal structure from only observational data is about the hardest imaginable problem we could face with data!

#### CausalNex

[CausalNex](https://causalnex.readthedocs.io/en/latest/) is a very recently released (at time of writing) toolkit to help data scientists do causal reasoning, by QuantumBlack. It provides both a graph structure learning component, to help build the causal graph, and tools to fit that graph as a Bayesian network.

The structure learning component is an implementation of [DAGs with NOTEARS](https://arxiv.org/abs/1803.01422), an algorithm that casts structure learning as a continuous optimization problem. In its simplest form, it assumes linear relationships between variables (but unlike some causal discovery methods, does not assume Gaussian noise). Further, the algorithm assumes that all variables are observed (ie. there is data for all variables). Unfortunately, this is rarely the case in causal problems.

Within these limitations, the algorithm is performant, and allows the user to specify hard constraints (such as "these variables cannot be child nodes", or "there is no causal relationship between these two variables"). This facilitates directly encoding domain knowledge into the graph, and using the structure learning component as an aid in places where the causal connection is not known.

#### Pyro

Uber's [Pyro](http://pyro.ai/) probabilistic programming library is primarily intended for implementing deep probabilistic models and fitting them with variational inference. However, in addition to tools for conditioning on observed data, the library implements a do operation to force a variable to take a certain distribution. This allows simulating from interventional distributions, provided the structural causal model (including equations) is known. The intersection of probabilistic programming with causal inference is nascent, but promising!