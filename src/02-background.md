## Background: Causal Inference

In this chapter, we aim to familiarize the reader with the essentials of causal reasoning, especially in how it differs from supervised learning. In particular, we give an informal introduction to structural causal models. Grasping the basic notions of causal modeling allows for a much richer understanding of invariance and generalization, which we discuss in the next Chapter.

### Why are we interested in causal inference?

Imagine a bank that would like to reduce the number of business loans which default. Historical data and sophisticated supervised learning techniques may be able to accurately identify which loans are likely to default, and interpretability techniques may tell us some features that are correlated with (or predictive of) defaulting. However, to reduce the default rate, we must understand what changes to make, which requires understanding not only _which_ loans default, but _why_ the loans default.

![A bank would like to decide which business loans to grant based on true, causal relationships.](figures/ff13-01.png)

It may be that we find small loans are more likely to default than larger loans. One might naively assume that the bank ought to stop making small loans. However, perhaps it is really the case that smaller businesses are more likely to fail than large businesses, and _also_ more likely to apply for small loans. In this case, the true causal relationship is between the size of the _business_ and defaulting, not between the size of the _loan_ and defaulting. If so, we could have made a poor policy decision about loan size, rather than business size.

Unfortunately, supervised learning alone cannot tell us which is true. If we include both loan size and business size as features in our model, we will simply find that they are both related to loan defaulting, to some extent. While that insight is true - as they are both statistically related to defaulting - which is _causing_ defaulting is a separate question, and the one to which we want the answer.

Causality gives us a framework to reason about such questions, and recent developments at the intersection of causality and machine learning are making the discovery of such causal relationships easier.


#### The shortcomings of supervised learning

Supervised machine learning has proved enormously successful at some tasks - particularly in dealing with tasks that require high-dimensional inputs, such as computer vision and natural language processing. There has been truly remarkable progress over the past two decades, and it should be noted that acknowledgment of the shortcomings of supervised learning does not diminish that progress.

With success have come inflated expectations of autonomous systems capable of independent decision-making, and even human-like intelligence. Current machine learning approaches are unable to meet those expectations, owing to fundamental limitations of pattern recognition.

One such limitation is **generalizability** (also called _robustness_ or _adaptability_), that is, the ability to apply a model learned in one context in a new environment. Many current state-of-the-art machine learning approaches assume that the trained model will be applied to data that looks the same as the training data. These models are trained on highly specific tasks, like recognizing dogs in images or identifying fraud in banking transactions. In real life, though, the data on which we predict is often different from the data on which we train, even when the task is the same. For example, training data is often subject to some form of selection bias, and simply collecting more of it does not mitigate that.

![The real world is often distributed differently than our training data.](figures/ff13-02.png)

Another limitation is **explainability**, that is, machine learning models remain mostly “black boxes” unable to explain the reasons behind their predictions or recommendations, thus eroding users' trust and impeding diagnosis and repair. For example, a deep learning system can be trained to recognize cancer in medical images with high accuracy, provided it is given plenty of images and compute power, but - unlike a real doctor - it cannot explain why or how a particular image suggests disease. Several methods for understanding model predictions have been developed, and while these are necessary and welcome, understanding the interpretation and limitations of their outputs is a science in itself. While model interpretation methods like [LIME](https://arxiv.org/abs/1602.04938) and [SHAP](https://arxiv.org/abs/1705.07874) are useful, they provide insight only into how the model works, not how the world works.

![figure: explainability](figures/ff13-03.png)

And finally, the understanding of **cause-and-effect** connections - a key element of human intelligence - is absent from pattern recognition systems. Humans have the ability to answer “what if” kinds of questions. What if I change something? What if I had acted differently? Such interventional, counterfactual or retrospective questions are the forté of human intelligence. While imbuing machines with this kind of intelligence is still far-fetched, researchers in deep learning are increasingly recognizing the importance of these questions, and using them to inform their research.^[See for instance, recent works by Yoshua Bengio, like [A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms](https://arxiv.org/abs/1901.10912).]

![figure: cause and effect](figures/ff13-04.png)

All of this means that supervised machine learning systems must be used cautiously in certain situations. And if we want to mitigate these restrictions effectively, causation is key.

#### What does causality bring to the table?

Causal inference provides us with tools that allow us to answer the question of _why_ something happens. This takes us a step further than traditional statistical or machine learning approaches that are focused on predicting outcomes and concerned with identifying associations.

Causality has long been of interest to humanity on a philosophical level, but it has only been in the latter half of the 20th century (thanks to the work of pioneering methodologists such as Donald Rubin and Judea Pearl), that a mathematical framework for causality has been introduced. In recent years, the boom of machine learning has enhanced the development of causal inference and attracted new researchers to the area.

Identifying causal effects helps us understand a variety of things: for example, user behavior in online systems^[See, for instance [Using Causal Inference to Improve the Uber User Experience](https://eng.uber.com/causal-inference-at-uber/).], effect of social policies, risk factors of diseases. Questions of cause-and-effect are also critical for the design of data-driven applications. For instance, how do algorithmic recommendations affect our purchasing decisions? How do they affect a student’s learning outcome or a doctor’s efficacy? All of these are hard questions and require thinking about the counterfactual: what would have happened in a world with a different system, policy, or intervention?  Without causal reasoning, correlation-based methods can lead us astray.

That said, learning causality is a challenging problem. There are broadly two situations in which we could find ourselves: in one case, we are able to actively intervene in the system we are modeling and get experimental data; in the other, we have only observational data.

The gold standard in establishing causal effects is a Randomised Controlled Trial (RCT) and this falls under the experimental data category. In an RCT, we try to engineer similar populations using random assignment (as choosing the populations manually could introduce selection effects that destroy our ability to learn causal relations) and apply an intervention to one population and not the other. From this, we measure the causal effect of changing one variable as a simple difference in the quantity of interest between the two populations.

![figure: rct](figures/ff13-05.png)

We can use RCTs to establish whether a particular causal relation holds. However, trials are not always physically possible, and even when they are, they are not always ethical (for instance, it would not be ethical to deny a patient a treatment that is reasonably believed to work, or trial a news aggregation algorithm designed to influence a person’s mood without informed consent).^[Facebook performed [such an experiment](https://www.pnas.org/content/111/24/8788) in 2012, and received [much criticism](https://www.theatlantic.com/technology/archive/2014/06/everything-we-know-about-facebooks-secret-mood-manipulation-experiment/373648/) as a result. The ethical problem is not so much with the experiment itself, but rather that the subjects had not given informed consent, in violation of basic ethical guidelines for psychological research.] In some cases, we can find naturally occurring experiments. In the worst case, we're left trying to infer causality from observational data alone.

In general, this is not possible, and we must at least impose some modeling assumptions. There are several formal frameworks for doing so. For our purpose of building intuition, we’ll introduce the [Structural Causal Model](http://bayes.cs.ucla.edu/BOOK-2K/) (SCM) framework of Pearl in this chapter.^[An alternative popular framework is the Neyman-Reuben causal model, also known as [Potential Outcomes](https://www.cambridge.org/core/books/causal-inference-for-statistics-social-and-biomedical-sciences/71126BE90C58F1A431FE9B2DD07938AB). The frameworks are equivalent in that they can compute the same things, though some causal queries may be easier to reason about in one or the other.]


### The ladder of causation

In [The Book of Why](http://bayes.cs.ucla.edu/WHY/), Judea Pearl, author of much foundational work in causality, describes three kinds of reasoning we can perform as the rungs on a ladder. These rungs describe when we need causality, and what it buys us.^[See also Pearl’s article: [The Seven Tools of Causal Inference, with Reflections on Machine Learning](https://cacm.acm.org/magazines/2019/3/234929-the-seven-tools-of-causal-inference-with-reflections-on-machine-learning/fulltext).]

![The ladder of causation, as described in [The Book of Why](http://bayes.cs.ucla.edu/WHY/).](figures/ff13-06.png)

On the **first rung**, we can do **statistical and predictive reasoning**. This covers most (but not all) of what we do in machine learning. We may make very sophisticated forecasts, infer latent variables in complex deep generative models, or cluster data according to subtle relations - all of these things sit on rung one.

_Example: a bank wishes to predict which of its current business loans are likely to default, so it can make financial forecasts accounting for likely losses._

The **second rung** is **interventional reasoning**. Interventional reasoning allows us to predict what will happen when a system is changed. This enables us to describe what characteristics are particular to the exact observations we’ve made, and what should be invariant across new circumstances. This kind of reasoning requires a _causal_ model. Intervening is a fundamental operation in causality, and we’ll discuss both interventions and causal models in this chapter.

_Example: a bank would like to reduce the number of loans which default, and considers changing its policy. Predicting what will happen as a result of this intervention requires that the bank understand the causal relations which affect loan defaulting._

The **third rung** is **counterfactual reasoning**. On this rung, we can talk not only about what has happened, but also what would have happened if circumstances were different. Counterfactual reasoning requires a more precisely specified causal model than intervention. This form of reasoning is very powerful, providing a mathematical formulation of computing in alternate worlds where events were different.

_Example: a bank would like to know what the likely return on a loan would have been, had they offered different terms to what they did._

![figure: ladder with bank](figures/ff13-07.png)

By now, we hopefully agree that there is something to causality, and it has much to offer. We have yet to really define causality. We must begin with that most familiar refrain: correlation is not causation.


### From correlation to causation

#### Spurious correlations

Very many things display correlation. The rooster crows when the sun rises.^[Some farm-experienced members of the CFF team are keen to point out that roosters crow pretty much _all the time_.] The lights turn off when you flick a switch. Global temperatures have risen alarmingly since the 1800s, and meanwhile pirate numbers have dwindled to almost nothing ([Forbes](https://www.forbes.com/sites/erikaandersen/2012/03/23/true-fact-the-lack-of-pirates-is-causing-global-warming/#5cb710453a67)). These examples show us that while correlation can _appear_ as a result of causation, as in the case of the light switch, correlation certainly does not always _imply_ causation, as in the case of the pirates.

Correlated things are not always related.^[On a technical note, correlation measures only _linear_ association. For instance, `x` squared is uncorrelated with `x`, despite being completely dependent on it. When we say “correlation is not causation,” we really mean “statistical dependence is not causation”.] It’s possible to find many correlations with no readily imaginable causal interaction. The internet treasure [Spurious Correlations](https://www.tylervigen.com/spurious-correlations) collects many amusing examples of this. These spurious correlations most likely arise as a result of small sample size and coincidences that are bound to happen when making many comparisons. We should not be surprised if we find something that has low probability if we try many combinations.

[figure: spurious correlation]

In real world systems, spurious correlations can be cause for serious ethical concerns. For instance, certain characteristics may be associated with individuals or minority groups which make superficial features powerful at a learning task. This can easily embed bias and unfairness into an algorithm based on spurious correlations in a given dataset.

#### The Principle of Common Cause

In a posthumous 1956 book, [The Direction of Time](https://www.goodreads.com/book/show/848892.The_Direction_of_Time), Hans Reichenbach outlined the principle of common cause. He states the principle this way:

> “If an improbable coincidence has occurred, there must exist a common cause.”

Our understanding of causality has evolved, but this language is remarkably similar to what we use now. Let’s discuss how correlation may arise from causation.

We will do this in the framework of Structural Causal Models (SCMs). An SCM is a directed acyclic graph of relationships between variables. The nodes represent variables, and the edges between them point from cause to effect. The value of each variable depends only on its direct parents in the graph (the other variables which point directly into it) and a noise variable encapsulating any environmental interactions we are not modeling. We will examine three fundamental causal structures.

::: info
##### Causal Terminology

A causal graph is a directed acyclic graph denoting the dependency between variables.

A structural causal model carries more information than a causal graph alone - it also specifies what the functional form of dependencies between variables.

Remarkably, it’s possible to do much causal reasoning, including calculating the size of causal effects, via the graph alone, without specifying a parametric form for the relationships between causes and effects.
:::

##### 1. Direct causation

The simplest way in which correlation between two variables arises is when one variable is a direct cause of the other. We say that one thing causes another when a change in the first thing, while holding everything else constant, results in a change in the second. In the business loan defaulting example discussed earlier, this could mean we could create a two node graph with one of the nodes being whether or not a business is small (say “small business” with values 0 or 1) and the other node being “default” indicating whether or not the business defaulted on the loan. In this case, we would expect that a small business increases the chances of it defaulting.

![figure: direct causation](figures/ff13-08.png)

This setup is immediately reminiscent of supervised learning, where we have a dataset of features, X, and targets, Y, and want to learn a mapping between them. However, in machine learning, we typically start with all available features and select those that are most informative about the target. When drawing a causal relationship, only those features we believe have an actual causal effect on the target should be included as direct causes. As we will see below, there are other diagrams that can lead to a predictive statistical relationship between X and Y in which neither directly causes the other.

##### 2. Common cause

A common pattern is for a single variable to be the cause of multiple other variables. If a variable Z is a direct cause of both X and Y, we say that Z is a common cause and call the structure a “fork.” For example, unemployment could potentially cause both loan default and reduced consumer spend.

![figure: fork](figures/ff13-09.png)

Because both consumer spend and loan default depend on unemployment , they will appear correlated. A given value of unemployment will generate some values of consumer spend and loan default, and when unemployment changes, both consumer spend and loan default will change. As such, in the joint distribution of the SCM, the two dependent variables, consumer spend and loan default, will appear statistically related to one another.

However, if we were to _condition_ on unemployment (for instance, by selecting data corresponding to a fixed unemployment rate), we would see that consumer spend and loan default are independent from one another.

The common cause unemployment _confounds_ the relationship between consumer spend and loan default. We are unable to correctly calculate the relationship between consumer spend and loan default without accounting for unemployment (by conditioning). This is especially dangerous if unnoticed.

Unfortunately, confounders can be tricky or impossible to detect from observational data alone. In fact, if we look only at consumer spend and loan default, we could see the same joint distribution as in the case where consumer spend and loan default are directly causally related. As such, we should think of causal graphs as encoding our _assumptions_ about the system we are studying. We return to this point in [How do we know which graph to use?](#how-do-we-know-which-graph-to-use%3F)

##### 3. Common effect

The opposite common pattern is for one effect to have multiple direct causes. A node that has multiple causal parents is called a “collider” with respect to those nodes.

![figure: collider](figures/ff13-10.png)

A collider is a node that depends on more than one cause. In this example, loan defaulting depends on both commercial credit score and number of liens (a “lien” refers to the right to keep possession of property belonging to another entity until a debt owed by that entity is discharged), so we call loan default a _collider_.

Colliders are different to chains of direct causation and forks because the conditioning behaviour works oppositely. Before any conditioning, commercial credit score and number of liens are unconditionally independent. There is no variable with causal arrows going into both commercial credit score and number of liens, and no arrow linking them directly, so we should not expect a statistical dependency. However, if we condition on the collider, we will induce a conditional dependence between commercial credit score and number of liens.

This may seem a little unintuitive, but we can make sense of it with a little thought experiment. Loan default depends on both commercial credit score and number of liens, so if either of those changes value, the chance of loan default changes. We fix the value of loan default (say, we look only at those loans that did default). Now, if we were to learn anything about the value of commercial credit score, we would know something about the number of liens too; only certain values of number of liens are compatible with the conditioned value of loan defaulting and observed value of commercial credit score. As such, conditioning on a collider induces a spurious correlation between the parent nodes. Conditioning on a collider is exactly selection bias!


#### Structural Causal Models, in code

::: info

The small causal graphs shown above are an intuitive way to reason about causality. Remarkably, we can do much causal reasoning (and calculate causal effects) simply by specifying qualitatively which variables causally influence others with these graphs. In the real world, causal graphs can be large and complex.

Of course, there are other ways to encode the information. Given the graph, we can easily write down an expression for the joint distribution: it’s the product of probability distributions for each node conditioned on its direct causal parents. In the case of a collider structure, `x` &rarr; `z` &larr; `y`, the joint distribution is simply `p(x,y,z) = p(x) p(y) p(z|x,y)`. The conditional probability `p(z|x,y)` is exactly what we’re used to estimating in supervised learning!

If we know more about the system, we can move from this causal graph to a full structural causal model. An example SCM compatible with this graph would be:

```python
from numpy.random import randn

def x():
  return -5 + randn()

def y():
  return 5 + randn()

def z(x, y):
  return x + y + randn()

def sample():
  x_ = x()
  y_ = y()
  z_ = z(x_, y_)
  return x_, y_, z_
```

Each of the variables has an independent random noise associated with it, arising from factors not modeled by the graph. These distributions need not be identical, but must be independent. Notice that the structure of the graph encodes the dependencies between variables, which we see as the function signatures. The values of `x` and `y` are independent, but `z` depends on both. We can also see clearly that the model defines a generative process for the data, since we can easily sample from the joint distribution by calling the `sample` function. Doing so repeatedly allows us to chart the joint distribution, and see that `x` and `y` are indeed independent; there’s no apparent correlation in the scatter chart.

[figure: histogram and scatter plot]
Fig. Histogram of the distribution of x, y and z, and scatter plot of the joint distribution of x and y.

Now that we have a model in code, we can see a selection bias effect. If we condition the data to only values of `z` (the collider node) greater than a cutoff (which we can do easily, if inefficiently, by filtering the samples to those where `z > 2.5`), the previously independent `x` and `y` become negatively correlated.

:::

### From prediction to intervention

Now that we have some understanding of what a causal model is, we can get to the heart of causality: the difference between an observation and an _intervention_.

When we introduced the ladder of causation, we mentioned the notion of _intervention_, something that changes the system. This is a fundamental operation, and it is important to understand the difference between intervention and observation. It may not at first seem natural to consider intervening as a fundamental action, evoking a similar sense of confusion as when one first encounters priors in Bayesian statistics. Is an intervention subjective? Who gets to define what an intervention is?

Simply, an intervention is a change to the data generating process. The joint distribution of the variables in the graph may be sampled from by simply "running the graph forward." For each cause, we sample from its noise distribution and propagate that value through the SCM to calculate the resulting effects. To compute an _interventional_ distribution, we force particular causes (on which we are intervening) to some value, and propagate those values through the equations of the SCM. This introduces a different distribution to the observational distribution we usually work with.

There is sometimes confusion between an interventional distribution and a conditional distribution. A conditional distribution is generated by filtering an observed distribution to meet some criteria. For instance, we might want to know the loan default rate among the businesses we have granted a loan to at a particular interest rate. This interest rate would itself likely have been determined by some model, and as such, the businesses with that rate will likely share statistical similarities.

The interventional distribution (when we intervene on interest rate) is fundamentally different. It is the distribution of loan defaulting if we _fix_ the interest rate to a particular value, regardless of other features of the business that may warrant a different rate. This corresponds to removing all the inbound arrows to the interest rate in the causal graph - we’re forcing the value, so it no longer depends on its causal parents.

Clearly, not all interventions are physically possible! While we could intervene to set the interest rate, we would not be able to make every business a large one.

#### Interventions in code

::: info

It is easy to make interventions concrete with code. Returning to the collider example, to compute an interventional distribution, we could define a new sampling function where instead of drawing all variables at random, we intervene to set `x` to a particular value. Because this is an intervention, not simply conditioning (as earlier), we must make the change, then run the data generating process again.

```python
def sample_intervened():
  x_ = -3
  y_ = y()
  z_ = z(x_, y_)
  return x_, y_, z_
```

Performing this intervention results in a new distribution for `z`, which is different from the observational distribution that we saw earlier.

[figure: interventional histogram]

Further, the relationship between x and y has changed; the joint distribution is now simply the marginal distribution of `y`, since `x` is fixed. This is a strikingly different relationship than when we simply conditioned the observational distribution.

[figure: interventional scatterplot]

:::

#### Interventions in customer churn

In our [interpretability report](https://ff06-2020.fastforwardlabs.com/), we present a customer churn modeling use case. Briefly, given 20 features of customers of a telco - things like tenure, demographic attributes, whether they have phone and internet services and whether they have tech support - we must model their likelihood of churning within a fixed time. To do this, a dataset of customers and whether they churned in the time period is provided. This can be modeled as straightforward binary classification, and we can use the resulting output scores as a measure of how likely a customer is to churn.

The model used to calculate the churn score is an ensemble of a linear model, a random forest, and a simple feed forward neural network. With appropriate hyperparameters and training procedure, such an ensemble is capable of good predictive performance. That performance is gained by exploiting subtle correlations in the data.

To understand the predictions made, we apply [LIME](https://arxiv.org/abs/1602.04938). This returns a feature importance at the local level: which features contributed to each individual prediction. To accompany the analysis, we built Refractor, an interface for exploring the feature importances. Examining these is interesting, and highlights the factors that are _correlated_ with a customer being likely to churn. [Refractor](https://refractor.fastforwardlabs.com/) suggests which features most affect the churn prediction, and allows an analyst to change customer features and see the resulting churn prediction.

[figure: refractor]

Because we have a model that provides new predictions when we change the features, it is tempting to believe we can infer from this alone how to reduce churn probability. Aside from the fact that often the most important features cannot be changed by intervention (tenure, for instance), this is an incorrect interpretation of what LIME and our model provide. The correct interpretation of the prediction is the probability of churn for someone who _naturally_ occurred in our dataset with those features, or, for instance, what this same customer's churn probability will look like next year (when tenure will have naturally increased by one year), assuming none of their other features change.

Of course, there are some features that can be changed in reality. For instance:
- the telco could reduce the monthly fee for a customer,
- or try to convince them to change contract type from monthly to yearly (one does not have to think too hard about why this changes the short-term churn probability), or
- upgrade the service from DSL to fiber optic.

Which of these interventions would most decrease the probability that the customer churns? We don't know. Our model alone, for all its excellent predictive accuracy, can't tell us that, precisely because it is entirely correlative. Even a perfect model, that 100% accurately predicts which customers will churn, cannot tell us that.

With some common sense, we can see that a causal interpretation is not appropriate here. LIME often reports that having a faster fiber optic broadband connection increases churn probability, relative to slower DSL. It seems unlikely that faster internet has this effect. In reality, LIME is correctly reporting that there is a _correlation_ between having fiber optic and churning, likely because of some latent factors - perhaps people who prefer faster internet are also intrinsically more willing to switch providers. This distinction of interpretation is crucial.

The model can only tell us what **statistical dependencies** exist in the dataset we trained it on. The training dataset was purely observational - a snapshot of a window of time with observations about those customers in it. If we select "give the customer access to tech support" in the app, the model can tell us that similar customers who also had access to tech support were less likely to churn. Our model only captures information about customers who happened to have some combination of features. It does not capture information about what happens when we _change_ a customer's features. This is an important distinction.

To know what would happen when we intervene to change a feature, we must compute the interventional distribution (or a point prediction), which can be very different from the observational distribution. In the case of churn, it’s likely the true causal graph is rather complex.
Interpretability techniques such as LIME provide important insights into models, but they are not causal insights. To make good decisions using the output of any interpretability method, we need to combine it with causal knowledge.

Often, this causal knowledge is not formally specified in a graph, and we simply call it “domain knowledge,” or expertise. We have emphasized what the _model_ cannot do, in order to make the technical point clear, but in reality, anyone working with the model would naturally apply their own expertise. The move from that to a causal model requires formally encoding the assumptions we make all the time and verifying that the expected statistical relationships hold in our observed data (and if possible, experimenting). Doing so would give us an understanding of the cause-effect relationships in our system, and the ability to reason quantitatively about the effect of interventions.

![figure: people making assumptions](figures/ff13-11.png)

Constructing a useful causal model of churn is a complex undertaking, requiring both deep domain knowledge and a detailed technical understanding of causal inference.^[ Alas, it requires a far more detailed technical knowledge than we can provide in this report. We recommend the textbook [Causal Inference in Statistics: A Primer](http://bayes.cs.ucla.edu/PRIMER/) for a succinct introduction to Structural Causal Models. An abbreviated overview ([Causal Inference in Statistics: An Overview](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf)) is freely available as a PDF. The textbook [Elements of Causal Inference](https://mitpress.mit.edu/books/elements-causal-inference) (available through Open Access) also covers structural causal models, with additional links to machine learning.] In Chapter 3, we will discuss some techniques that are bridging the gap between a full causal model, and the supervised learning setup we use in problems like churn prediction.

#### When do we need interventions?

When do we need to concern ourselves with intervention and causality? If all we want to do is predict, and to do so with high accuracy (or whatever model performance metric we care about), then we should use everything at our disposal to do so. That means making use of all the variables that may correlate with the outcome we’re trying to predict, and it doesn’t matter that they don’t cause the outcome. Correlation is not causation, but correlation is still predictive,^[We will examine the nuances of this statement in the next Chapter. Correlation is predictive in distribution.] and supervised learning excels at discovering subtle correlations.

Some situations where this pure supervised learning approach is useful:
- We want to predict when a machine in our factory will fail.
- We want to forecast next quarter’s sales.
- We want to identify named entities in some text.

Conversely, if we want to predict the effect of an intervention, we need causal reasoning. For example:
- We want to know what to change about our machines to reduce the likelihood of failures.
- We want to know how we can increase next quarter’s sales.
- We want to know whether longer or shorter article headlines generate more clicks.^[Adam Kelleher and Amit Sharma have an excellent [blog post](https://medium.com/@akelleh/introducing-the-do-sampler-for-causal-inference-a3296ea9e78d) describing this problem, and introducing a new causal sampling technology to make solving it easier.]

### How do we know which graph to use?

Knowing the true causal structure of a problem is immensely powerful. Earlier in the chapter we discussed three building blocks of causal graphs - direct causation, forks and colliders - but for real problems, a graph can be arbitrarily complex.

The graph structure allows us to reason qualitatively about what statistical dependencies ought to hold in our data. In the absence of abundant randomized controlled trials or other experiments, qualitative thinking is necessary for causal inference. We must use our domain knowledge to construct a plausible graph to test against the data we have. It is possible to refute a causal graph by considering the statistical independence relations it implies, and matching those against the expected relations from the causal structure. For example, if two variables are connected by a common cause we have not conditioned on, we should expect a statistical dependence between them.

::: info

##### Causal Discovery

The independence relationships implied by a graph can be used for causal discovery. Causal discovery is the process of attempting to recover causal graphs from observational data. There are many approaches appropriate for different sets of assumptions about the graph. However, since many causal graphs can imply the same joint distribution, the best we should hope for from causal discovery is a set of plausible graphs, which, if we are fortunate, may contain the true graph. In reality, inferring the direction of causation in even a two variable system is not always possible from data alone. See [Distinguishing cause from effect using observational data: methods and benchmarks](http://jmlr.org/papers/v17/14-518.html).

:::

It is not, in general, possible to _prove_ a causal graph, since different graphs can result in the same observed and even interventional distributions. The difficulty of confirming a causal relationship means that we should always proceed with caution when making causal claims. It is best to think of causal models as giving results _conditional on a set of causal assumptions_. Two nodes that are not directly connected in the causal graph are assumed to be independent in the data generating process, except insofar as the causal relations described above (or combinations of them) induce a statistical dependence.

The validity of the results depends on the validity of the assumptions. Of course, we face the same situation in all machine learning work, and it is to be expected that stronger, causal claims require stronger assumptions than merely observational claims.

One case where we may be able to write down the true causal graph is when we have ourselves created the system. For instance, a manufacturing line may have a sufficiently deterministic process that it is possible to write down a precise graph encoding which parts move from which machine to another. If we were to model the production of faulty parts, that graph would be a good basis for the causal graph, since a machine that has not processed a given faulty part is unlikely to be responsible for the fault, and causal graphs encode exactly these independences.


### Recap

Causal graphical models present an intuitive and powerful means of reasoning about systems. If an application requires only pure prediction, this reasoning is not necessary, and we may apply supervised learning to exploit subtle correlations between variables and our predicted quantity of interest. However, when a prediction will be used to inform a decision that changes the system, or we want to predict for the system under intervention, we must reason causally  - or else likely draw incorrect conclusions. That said, behind every causal conclusion there is always a causal assumption that cannot be tested or verified by mere observation.

Even without a formal education in causal inference, there are advantages to the qualitative reasoning enabled by causal graphical models. Trying to write down a causal graph forces us to confront our mental model of a system, and helps to highlight potential statistical and interpretational errors. Further, it precisely encodes the independence assumptions we are making. However, these graphs could be complex and high dimensional and require close collaboration between practitioners and domain experts who have substantive knowledge of the problem.

In many domains, problems such as the large numbers of predictors, small sample sizes, and possible presence of unmeasured causes, remain serious impediments to practical applications of causal inference. In such cases, there is often limited background knowledge to reduce the space of alternative causal hypotheses. Even when experimental interventions are possible, performing the many thousands of experiments that would be required to discover causal relationships between thousands or tens of thousands of predictors is often not practical.

Given these challenges, how do we combine causal inference and machine learning? Many of the researched approaches at the intersection of ML and causal inference are motivated by the ability to apply causal inference techniques to high dimensional data, and in domains where specifying casual relationships could be difficult. In the next chapter, we will bridge this gap between structural causal models and supervised machine learning.
