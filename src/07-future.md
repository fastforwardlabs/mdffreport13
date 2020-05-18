## Future

At the outset, causal reasoning provides a conceptual and technical framework for addressing questions about the effect of real or hypothetical actions or _interventions_. Once we understand what the effect of an action is, we can turn the question around and ask what action plausibly caused an event. This gives us a formal language to talk about cause-and-effect. That said, not every question about cause is easy to answer. Further, it may not be a trivial task to find an answer or even to interpret it. Causal graphs that we discuss in the [Background: Causal Inference](#background%3A-causal-inference) chapter provide a convenient way to discuss these notions, and allow us to reason about statistical dependencies in observed data.

Structural causal models take a step further to this intuitive way of reasoning by making formal assumptions about the parametric form of how the variables interact.

However, causal graphs and SCMs become difficult to construct as the number of variables increases. Some systems are hard to model in this way. How do we draw a causal graph for pixels of an image? Or words in text? The problem gets out of hand quickly.

Fortunately, not all problems require the entire causal graph. Often, we are interested only in the causal relations associated with one particular target variable. This is where methods based on invariance (like IRM) step in to allow the model to capture stable features across environments (that is, different data generating processes). This paradigm enables out-of-distribution generalization. As opposed to causal graphs or structural causal models, where the only way to validate assumptions of the variable interactions is through experimentation, IRM allows us to test them on an unseen test set!

### Comparable approaches

So, at this point we probably agree that methods based on invariance are promising. How else might we approach out-of-distribution generalization? In general, there are two families of approaches; those that learn to match the feature distributions (or estimate a data representation) and those that employ some kind of optimization technique.

#### Domain adaptation

Domain adaptation is a special case of transfer learning. In domain adaptation, the model learns a task in a source domain, which has some feature distribution, and we would like it to be able to perform the same task well in a target domain, where the feature distribution is different. Domains play the same role as environments in invariance-based approaches - a source domain is an environment that was trained in, and a target domain is any environment that was not trained in.

Domain adaptation also enforces a kind of invariance - it seeks a representation that is distributed the same across source and target domain (so, across environments).^[[Domain adversarial training of neural networks](https://arxiv.org/abs/1505.07818)] However, truly invariant, causal features need not follow the same distribution in different environments. A snowy cow will not generate quite the same pixel distribution as a sandy cow, and the causal feature we wish to represent is the cow itself.

#### Robust learning

The idea of learning across multiple environments is not novel to invariance based approaches. [Robust Supervised Learning](https://www.aaai.org/Library/AAAI/2005/aaai05-112.php) is a family of techniques that uses the same multi-environment setup as IRM (but much predate it), with a similar goal of enabling or enhancing out-of-distribution generalization. Said differently, the goal is a predictor that is robust to distributional shifts of the inputs.

The difference from the IRM setup we have covered is the loss function. The key idea is to add environment-specific “baseline” terms to the loss, and try to fix these terms such that particularly noisy environments where the loss may be high do not dominate. Then minimizing the loss should guarantee good performance across all the known environments. Further, a robust predictor will perform well in new environments that are interpolations of those seen in training. This certainly improves out-of-distribution generalization, but does not allow _extrapolation_ outside of what was seen in training, whereas IRM can extrapolate, thanks to relying on an invariant predictor.

#### Meta-learning

Approaches like domain adaptation, robust learning, and in general transfer learning try to alleviate the problem of out-of-distribution generalization to some extent. Unfortunately, learning invariant features with varying distributions across environments is still challenging. These approaches are good at interpolation, but not extrapolation.

This is where meta-learning approaches like Model Agnostic Meta Learning (MAML)^[[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)] come into play. The underlying idea for meta-learners generally is to attempt to learn tasks with a small number of labeled examples. Training meta-learners is a two-step process involving a _learner_ and a _trainer_. The goal of the learner (model) is to quickly learn new tasks from a small amount of new data; hence, it is sometimes called a _fast learner_. (A task here refers to any supervised machine learning problem - e.g., predicting a class given a small number of examples.) This learner is trained, by the meta-learner, to be able to learn from a large number of different tasks. The meta-learner accomplishes this by repeatedly showing the learner hundreds and thousands of different tasks.

Learning then, happens at two levels. The first level focuses on quick acquisition of knowledge within each task with a few examples. The second level slowly pulls and digests information across all tasks. In case of MAML (which is optimization based), the learner (or the first level) can achieve an optimal fast learning on a new task with only a small number of gradient steps because the meta-learner provides a good initialization of a model’s parameters. This approach is close to the problem of learning an optimal classifier in multiple environments, and could be explored further to learn invariant features within the data.

Some recent works have made the connection between causality and meta-learning explicitly, see [A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms](https://arxiv.org/abs/1901.10912).

### Looking ahead

In this section, we discuss future possibilities with causality in general as well as with methods based on invariance.

#### Causal Reinforcement Learning

Reinforcement learning is the study of how an agent can learn to choose actions that maximize its future rewards in an interactive and uncertain environment. These agents rely on plenty of simulations (and sometimes real data) to learn which actions lead to high reward in a particular context. Causality is also about calculating the effect of actions, and allows us to transfer knowledge to new, unfamiliar situations. These two disciplines have evolved independently with little interaction between them until recently. Integrating them is likely to be a fruitful area of research, and may extend the reach of both causality and reinforcement learning (RL).^[There is a nice introduction to causal reinforcement learning in the paper [Reinforcement learning and causal models](http://gershmanlab.webfactional.com/pubs/RL_causal.pdf). The blog post [Introduction to Causal RL](https://causallu.com/2018/12/31/introduction-to-causalrl/) contains a shorter description, and also suggests some medical applications.]

There is a natural mapping between the concept of intervention in causal inference and actions taken in RL. Throughout an episode of reinforcement learning (an episode is formed of one run of the system, for example, a complete game of chess, or go), an agent takes actions. This defines a data generating process for the reward that the agent ultimately cares about; different sequences of actions will generate different rewards. Since the agent can choose its actions, each of them is an intervention in this data generating process. In making this connection, we can leverage the mathematics of causal inference. For instance, we could use counterfactuals, the third level of the [The ladder of causation](#the-ladder-of-causation), to reason about actions not taken. Applying such causal techniques may reduce the state space the agent needs to consider, or help account for confounders.

Methods based on invariance, like IRM, in principle, learn to discover unknown invariances from multiple environments. We could leverage this attribute in reinforcement learning. An episode of RL consists of all the states that fall in between an initial state and a terminal state. Since each episode is independent of another, in IRM terminology they could be viewed as different environments.  An agent could then learn robust policies from each of these episodes that leverage the invariant part of behaviour or actions that lead to reward.

While reinforcement learning itself is still in nascent stages when it comes to commercial applications, combining it with causality offers a great potential.^[We are grateful to David Lopez-Paz (one of the [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893) authors) for sharing his thoughts and ideas about possible extensions and applications of IRM with us, including applications to RL.] But prior to that, we need to address some questions. For example, how do we combine programming abstractions in causal modeling with reinforcement learning that can help find best decisions? What tools and libraries are necessary to enable commercial applications in this space?

#### IRM and environments

IRM uses the idea of training in multiple environments to achieve out-of-distribution generalization. Unfortunately, few datasets come with existing environment annotations. There are at least two ways we can try to address this problem.

The first is to be mindful of the environment when collecting data, and collect metadata alongside. This may be easy (for example, collecting the geo-location of photos in settings where this is possible and does not violate a user’s privacy), or extremely hard, and require much post-collection manual labelling.

Another compelling but untested option is to try combining IRM with some sort of clustering to segment a single dataset into environments.^[This idea was also suggested to us by David Lopez-Paz.] The question would be how to cluster in such a way that meaningful and diverse environments are defined. Since existing clustering approaches are purely correlative, and as such vulnerable to spurious correlations, this could prove challenging.

Studying the impact of environment selection, and how to create or curate datasets with multiple environments would be a valuable contribution to making invariance-based methods more widely applicable. The authors of [An Empirical Study of Invariant Risk Minimization](https://deepai.org/publication/an-empirical-study-of-invariant-risk-minimization) reach the same conclusion.

#### Causal reasoning for algorithmic fairness

In the [Ethics](#ethics) chapter we reviewed some notions of fairness in prediction problems and shared how tools of causal reasoning can be leveraged to address fairness. They depart in the traditional way of wholly relying on data-driven approaches and emphasize the need to require additional knowledge of the structure of the world, in the form of a causal model. This additional knowledge is particularly valuable as it informs us how changes in variables propagate in a system, be it natural, engineered or social. Explicit causal assumptions remove ambiguity from methods that just depend upon statistical correlations. Avoiding discrimination through causal reasoning is an active area of research. As efforts to aid more transparency and fairness in machine learning systems grow, causal reasoning will continue to gain significant momentum in guiding algorithms towards fairness.
