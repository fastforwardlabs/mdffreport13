## Ethics

Machine learning is playing an increasingly critical role in our society. Decisions that were previously exclusively made by humans are more frequently being made algorithmically. These algorithmic systems govern everything from which emails reach our inboxes, to whether we are approved for credit, to whom we get the opportunity to date – and their impact on our experience of the world is growing. Furthermore, we still lack the understanding of how these machines work. We can neither explain nor correct them when they are acting unfairly or reinforcing biases. Causal reasoning gives us a framework for thinking about these problems. It provides a principled approach to assist the data scientist making an informed decision about the metric used to ascertain fairness while being fully aware of the tradeoffs involved with his/her choice.

### Causal graphs make assumptions explicit

Even without employing the full machinery of causal inference, when one approaches a new problem, it can be informative to try to write down the causal graph. This forces us to confront our assumptions about a system. It also allows someone else to understand our assumptions, and gives a precise framework to debate.

[figure: people arguing over which direction an arrow goes]

Making our assumptions explicit aids transparency, which is a win. However, it doesn’t protect against bad assumptions. Establishing causal relationships is hard. Unless we are able to perform sufficient experiments to validate our hypotheses, causal reasoning from observational data is subject to untested (sometimes untestable) assumptions.

We should make any causal claim with humility. As ever, we should be careful of dressing up a bad analysis with additional formalism.

### Omitting protected attributes is not enough

Discrimination can be broadly grouped into two categories: direct and indirect. Direct discrimination is when indiviudals receive less favorable treatments on the basis of a protected attribute such as age, race, or disability. Some extreme cases include voting rights or unequal pay based on race or gender. Appropriately, these protected attributes are frequently omitted from machine learning systems. Using a protected attribute as a feature directly is inviting discrimination based on that attribute.

Indirect discrimination on the other hand is concerned with individuals who receive treatments on the basis of inadequately justified factors that are somewhat related to (but not the same as) the protected attribute. These cases are arguably more complex to characterize, and require more refined reasoning. One such scenario is wherein an insurance company denies coverage to residents in certain geographic locations. This isn’t a problem in itself, unless the majority of the residents belong to a certain ethnicity. In this case, the geographic location serves as a proxy for race. This is indirect _causal discrimination_.

Another sub-category of indirect discrimination is _spurious discrimination_. These are instances when there are no pathways from causal attributes to the outcome, but as we saw in [From correlation to causation](#from-correlation-to-causation), correlations can arise from numerous causal structures. As such, merely omitting the protected attribute does not omit its effects. A system is not guaranteed to be non-discriminatory on a protected attribute simply because it does not include that attribute directly. More simply, just because a feature does not cause the target does not mean that it will not be predictive of the target. This presents a particular challenge to algorithmic systems that are designed to find subtle correlations, especially since much historical data on which algorithms are trained is subject to selection bias (and other biases).

Since removing protected attributes is not enough, we must evaluate the resulting model for its discrimination and fairness properties. There are many possible measures of fairness, and it is generally impossible to optimize for all of them (see [Inherent Trade-Offs in the Fair Determination of Risk Scores](https://arxiv.org/abs/1609.05807)).

Several recent papers (see [Causal Reasoning for Algorithmic Fairness](https://arxiv.org/abs/1805.05859) and [Avoiding Discrimination through Causal Reasoning](https://arxiv.org/abs/1706.02744), for instance) have proposed causality as a route to understanding and defining fairness and discrimination. In particular, if we have a causal graphical model of a system, we can see which paths are impacted by protected attributes, and correctly account for that impact. There have also been contributions in non-parametric structural causal models that allow one to detect and distinguish the three main discriminations, namely, direct, indirect and spurious (see [Fairness in Decision-Making – The Causal Explanation Formula](https://www.cs.purdue.edu/homes/eb/r30.pdf)).

That said, the difficulty lies in constructing the causal graph. A causal graph could of course be used to embed all kinds of biases and prejudices, but at least provides a basis for argument.

### Invariance as a route to fairness

An interesting idea is proposed in the final section of the IRM paper: treating groups over which we want fairness as the environments. When we seek to learn an invariant model (be that by ICP or IRM), we are explicitly trying to learn a model that performs optimally in different environments. We could construct those environments by separating out groups having different values for protected attributes. Then, by learning a model that seeks to perform optimally in each environment, we are explicitly trying to guarantee the best performance for each protected attribute.

Said differently, invariant features are exactly those that are consistent across groups. Consider again a bank granting loans, this time directly to individuals. The bank does not wish to discriminate on the basis of protected attributes. By treating the protected attributes as the groups, they are looking to learn what impacts loan defaulting invariantly across those groups.

[figure: different types of cow]

The idea of learning an invariant predictor across environments is that the representation used is capturing something true about the generative process of the data. This representation would be, to some degree, _disentangled_, in the sense that each dimension of the representation (a vector) should correspond to something meaningful. [On the Fairness of Disentangled Representations](https://arxiv.org/abs/1905.13662) shows experimentally that disentangled representations improve fairness in downstream uses.