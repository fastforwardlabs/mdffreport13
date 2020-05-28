## Causality and invariance

Supervised machine learning is very good at prediction, but there are useful lessons we can take from causal models even for purely predictive problems.

Relative to recent advancements made in the broader field of machine learning, the intersection of machine learning and causal reasoning is still in its infancy. Nonetheless, there are several emerging research directions. Here, we focus on one particularly promising path: the link between causality and invariance. Invariance is a desirable property for many machine learning systems: a model that is invariant is one that performs well in new circumstances, particularly when the underlying data distribution changes. As we will see in this chapter, invariance also provides a route to some causal insights, even when working only with observational data.

### The great lie of machine learning

In supervised learning, we wish to predict something that we don’t know, based on only the information that we do have. Usually, this boils down to learning a mapping between input and output.

To create that map, we require a dataset of input features and output targets; the number of examples required scales with the complexity of the problem. We can then fit the parameters of a learning algorithm to the dataset to minimize some loss function that we choose. For instance, if we are predicting a continuous number, like temperature, we might seek to minimize the mean squared difference between the prediction and the true measurements.

If we are not careful, we will _over_ fit the parameters of the ML algorithm to the dataset we train on. In this context, an overfit model is one that has learned the idiosyncrasies (the spurious correlations!) of our dataset. The result is that when the model is applied to any other dataset (even one with the same data generating process), the model’s performance is poor, because it is relying on superficial features that are no longer present.

To avoid overfitting, we employ various regularization schemes and adjust the capacity of the model to an appropriate level. When we fit the model, we shuffle and split our data, so we may learn the parameters from one portion of the data, and validate the resulting model’s performance on another portion. This gives us confidence that the learned parameters are capturing something about all the data we have, and not merely a portion of it.

Whatever procedure we use (be it cross-validation, forward chaining for time series, or simpler train-test-validation splits), we are relying on a crucial assumption. The assumption is that the data points are _independent and identically distributed_ ( i.i.d.). By _independent_, we mean that each data point was generated without reference to any of the others, and by _identically distributed_, we mean that the underlying distributions in the data generating process are the same for all the data points.

Paraphrasing [Zoubin Ghahramani](https://www.youtube.com/watch?v=x1UByHT60mQ&feature=youtu.be&t=37m34s),

> the i.i.d. assumption is the great lie of machine learning.

Rarely are data truly independent and identically distributed. What are the ramifications of this misassumption for machine learning systems?

### Dangers of spurious correlations

When we train a machine learning system with the i.i.d. assumption, we are implicitly assuming an underlying data generating process for that data. This data generating process defines an _environment_. Different data generating processes will result in different environments, with different underlying distributions of features and targets.

When the environment in which we predict differs from the environment in which our machine learning system was trained, we should expect it to perform poorly. The correlations between features and the target are different - and, as such, the model we created to map from features to target in one environment will output incorrect values of the target for the features in another environment.

Unfortunately, it’s rarely possible to know whether the data generating process for data at predict time (in a deployed ML system, for instance) will be the same as during training time. Even once the system is predicting in the wild, if we do not or cannot collect ground truth labels to match to the input features on which the prediction was based, we may never know.

This problem is not academic. [Recognition in Terra Incognita](https://arxiv.org/abs/1807.04975) points this out in humorous fashion (see also [Unbiased Look at Dataset Bias](http://people.csail.mit.edu/torralba/publications/datasets_cvpr11.pdf)). Both of these papers highlight that computer vision systems trained for visual recognition of objects, animals, and people can utterly fail to recognise the same objects in different contexts. A cow on the slopes of an alpine field is easily recognised, but a cow on a beach is not noticed at all, or poorly classified as a generic “mammal.”

![Figure from [Recognition in Terra Incognita](https://arxiv.org/abs/1807.04975), where annotations were provided by [ClarifAI.com](https://www.clarifai.com/).](figures/terra-incognita.png)

These failures should not come as a surprise to us! Supervised machine learning is _designed_ to exploit correlations between features to gain predictive performance, and cows and alpine pastures are highly correlated. Neural networks are a very flexible class of models that encode the invariants of the dataset on which they’re trained. If cows dominantly appear on grass, we should expect this to be learned.

::: info

##### When is a correlation spurious?

In supervised learning, we learn to use subtle correlations, possibly in high dimensional spaces like natural images, to make predictions. What distinguishes a genuine correlation from a spurious one? The answer depends on the intended use of the resulting model.

If we intend for our algorithm to work in only one environment, with very similar images, then we should use all the correlations at our disposal, including those that are very specific to our environment. However, if - as is almost always the case - we intend the algorithm to be used on new data outside of the training environment, we should consider any correlation that only holds in the training environment to be spurious. A spurious correlation is a correlation that only appears to be true due to a selection effect (such as selecting a training set!).

In the previous chapter, we saw that correlation can arise from several causal structures. In the strictest interpretation, any correlation that does not arise from direct causation could be considered spurious.

Unfortunately, given only a finite set of training data, it is often not possible to know which correlations are spurious. The methods in this section are intended to address precisely that problem.

:::

When a machine learning algorithm relies heavily on spurious correlations for predictive performance, its performance will be poor on data from outside the dataset on which it was trained. However, that is not the only problem with spurious correlations. 

There is an important and growing emphasis on interpretability in machine learning. A machine learning system should not only make predictions, but also provide a means of inspecting how those predictions were made. If a model is relying on spurious correlations, the feature importances (such as those calculated by [LIME](https://arxiv.org/abs/1602.04938) or [SHAP](https://arxiv.org/abs/1705.07874)) will be similarly spurious. No one wants to make decisions based on spurious explanations!

### Invariance

To be confident of our predictions outside of our training and testing datasets, we need a model that is robust to distributional shifts away from the training set. Such a model would have learned a representation which ignores dataset-specific correlations, and instead relies upon features that affect the target in all environments.

How can we go about creating such a model? We could simply train our model with data from multiple environments, as we often do in machine learning (playing fast and loose with the i.i.d. assumption). However, doing so naively would provide us with a model that can only generalize to the environments it has seen (and interpolations of them, if we use a robust objective).^[See [Robust Supervised Learning](https://www.aaai.org/Library/AAAI/2005/aaai05-112.php).] We wish our model to generalize beyond the limited set of environments we can access for training, and indeed extrapolate to new and unseen (perhaps unforeseen) environments. The property we are looking for - performing optimally in all environments - is called invariance.

The connection between causality and invariance is well established. In fact, causal relationships are - by their nature - invariant. The way many intuitive causal relationships are established is by observing that the relationship holds all the time, in all circumstances. 

Consider how physical laws are discovered. They are found by performing a series of experiments in different conditions, and monitoring which relationships hold, and what their functional form is. In the process of discovering nature’s laws, we will perform some tests that do not show the expected result. In cases where a law does not hold, this gives us information to refine the law to something that is invariant across environments.^[The scientific process of iterated hypothesis and experimentation can also be applied to constructing a causal model for business purposes. The popular George Edward Box quote is pertinent here: “all models are wrong, but some are useful”.]

![We learn causal relationships by observing under different experimental conditions. Causal relationships are those that are invariant across the environments created by these conditions.](figures/ff13-12.png)

For example, water boils at 100&deg; Celsius (212&deg; Fahrenheit). We could observe that everywhere, and write a simple causal graph: temperature &rarr; water boiling. We have learned a relationship that is invariant across all the environments we have observed.

Then, a new experiment conducted on top of a tall mountain reveals that on the mountain, water boils at a slightly lower temperature. After some more experimentation, we improve our causal model, by realising that in fact, both temperature and pressure affect the boiling point of water, and the true invariant relationship is more complicated.

The mathematics of causality make the notion of invariance and environments precise. Environments are defined by interventions in the causal graph. Each intervention changes the data generating process, such that the correlations between variables in the graph may be different (see [From prediction to intervention](#from-prediction-to-intervention)). However, direct causal relationships are invariant relationships: if a node in the causal graph depends only on three variables, and our causal model is correct, it will depend on those three variables, and in the same way, regardless of any interventions. It may be that an intervention restricts the values that the causal variables take, but the relationship itself is not changed. Changing the arguments to a function does not change the function itself.

#### Invariance and machine learning

In the machine learning setting, we are mostly concerned with using features to predict a target. As such, we tend to select features for their predictive performance. In contrast, causal graphs are constructed based on domain knowledge and statistical independence relations, and thus encode a much richer dependency structure. However, we are not always interested in the entire causal graph. We may be interested only in the causes of a particular target variable. This puts us closer to familiar machine learning territory.

![In supervised learning, we often use all available variables (or a subset selected for predictive performance) to predict an outcome. With structural causal models, we encode a much richer dependency structure between variables.](figures/ff13-14.png)

We will now examine two approaches to combining causal invariance and machine learning. The first, invariant causal prediction, uses the notion of invariance to infer the direct causes of a variable of interest. This restricted form of causal discovery (working out the structure of a small part of the graph in which we are interested) is appropriate for problems with well defined variables where a structural causal model (or at least causal graph) could be created - in principle, if not in practice.

Not all problems are amenable to SCMs. In the following section, we describe invariant risk minimization, where we forego the causal graph and seek to find a predictor that is invariant across multiple environments. We don’t learn anything about the graph structure from this procedure, but we do get a predictor with greatly improved out-of-distribution generalization.

### Invariant Causal Prediction

[Invariant causal prediction](https://arxiv.org/abs/1501.01332) (ICP) addresses the task of invariant prediction explicitly in the framework of structural causal models.

Often, the quantity we are ultimately concerned with in a causal analysis is the causal effect of an intervention: what is the difference in the target quantity when another variable is changed.^[Judea Pearl’s do-calculus is a set of rules to calculate exactly which variables we must account for, and how, to answer a given causal query in a potentially complicated graph. This is not trivial - often there are unobserved variables in a graph, and we must try to express the query only in terms of those variables for which we have data.] To calculate that, we either need to hold some other variables constant, or else account for the fact that they have changed. If we are only interested in the causes that affect a particular target, we do not need to construct the whole graph, but rather only determine which factors are the true direct causes of the target. Once we know that, we can answer causal questions, like how strongly each variable contributes to the effect, or the causal effect of changing one of the input variables.

The key insight offered by ICP is that because direct causal relationships are invariant, we can use that to determine the causal parents (the direct causes). The set up is similar to machine learning; we have some input features, and we’d like a model of an output target. The difference from supervised learning is that the goal is not performance at predicting the target variable. In ICP, we aim to discover the direct causes of a given variable - the variables that point directly into the target in the causal graph.

![We are not always interested in the full causal graph, and instead only seek to find the direct causes of a given target variable. This brings some of the advantages of a causal model into the supervised learning paradigm.](figures/ff13-15.png)

To use ICP, we take a target variable of interest, and construct a plausible list of the potential direct causes of that variable. Then we must define environments for the problem: each environment is a dataset. In the language of SCMs, each environment corresponds to data observed when a particular intervention somewhere in the graph was active. We can reason about this even without specifying the whole graph, or even which particular intervention was active, so long as we can separate the data into environments. In practice, we often take an observed variable to be the environment variable, when it could plausibly be so.

For instance, perhaps we are predicting sales volume in retail, and want to discern what store features causally impact sales. The target is sales volume, and the potential causes would be features like store size, number of nearby competitors, level of staffing, and so on.

Environments might be different counties (or even countries) - something that is unlikely to impact the sales directly, but which may impact the features that themselves impact the sales. For instance, different places will have different populations, and population density is a possible cause of sales volume. Importantly, the environment cannot be a descendent of the target variable.^[There is a subtlety here. We said environments were defined by interventions. Naturally, it is impossible to intervene on the country a store is built in once the store is built. This turns out not to matter for the purposes of inferring the direct causal parents of the sales volume, so long as the country is further up the graph, and changing country alters the data generating process.]

![We fit a model in multiple environments, and monitor which features are consistently predictive.](figures/ff13-16.png)

To apply ICP, we first consider a subset of features. We then fit a linear (Gaussian) regression from this subset to the target in each environment we have defined. If the model does not change between environments (which can be assessed either via the coefficients or a check on residuals), we have found a set of features that appear to result in an invariant predictor. We iterate over subsets of features combinatorially. Features that appear in a model that is invariant are plausible causes of the target variable. The intersection of these sets of plausible causes (i.e. the features which are predictive in all environments) is then a subset of the true direct causes.

![The features that are consistently predictive of a target are likely the causal parents in the (unknown!) causal graph.](figures/ff13-17.png)

In machine learning terms, ICP is essentially a feature selection method, where the features selected are very likely to be the direct causes of the target. The model built atop those features can be interpreted causally: a high coefficient for a feature means that feature has a high causal effect on the target, and changes in those features should result in the predicted change in the target.

Naturally, there are some caveats and assumptions. In particular, we must assume that there is no unobserved confounding between the features and the target (recall, a confounder is a common cause of the feature and target). If there are known confounders, we must make some adjustments to account for them, detailed in the paper. The authors provide an R package, [InvariantCausalPrediction](https://cran.r-project.org/web/packages/InvariantCausalPrediction/index.html), implementing the methods.

The restriction of using a linear Gaussian model, and that environments be discrete, rather than defined by the value of a continuous variable, are removed by nonlinear ICP (see [Invariant Causal Prediction for Nonlinear Models](https://arxiv.org/abs/1706.08576)). In the nonlinear case, we replace comparing residuals or coefficients with conditional independence tests.^[Nonparametric conditional independence testing is an area of active research, and is generally hard, and made more so by having finite data. The nonlinear ICP paper also introduces the notion of defining sets - sometimes no single set of variables is accepted as the set of causal parents, but there are similar sets differing by only one or two variables that may be related. While the algorithm has failed to find a single consistent model, it is nonetheless conveying useful causal information.]

### Invariant Risk Minimization

When using Invariant Causal Prediction, we avoid writing the full structural causal model, or even the full graph of the system we are modeling, but we must still think about it.

For many problems, it’s difficult to even attempt drawing a causal graph. While structural causal models provide a complete framework for causal inference, it is often hard to encode known physical laws (such as Newton's gravitation, or the ideal gas law) as causal graphs. In familiar machine learning territory, how does one model the causal relationships between individual pixels and a target prediction? This is one of the motivating questions behind the paper [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893) (IRM). In place of structured graphs, the authors elevate invariance to the defining feature of causality.

They also make the connection between invariance and causality well:

> “If both Newton’s apple and the planets obey the same equations, chances are that gravitation is a thing.”
> -- [IRM](https://arxiv.org/abs/1907.02893) authors

Like ICP, IRM uses the idea of training in multiple environments. However, unlike ICP, IRM is not concerned with retrieving the causal parents of the target in a causal graph. Rather, IRM focusses on out-of-distribution generalization - the performance of a predictive model when faced with a new environment. The technique proposed aims to create a data representation, on which a classifier or regressor can perform optimally in all environments. The paper itself describes the IRM principle:

> “To learn invariances across environments, find a data representation such that the optimal classifier on top of that representation matches for all environments.”
> -- [IRM](https://arxiv.org/abs/1907.02893) authors

Said differently, the idea is that there is a latent causal structure behind the problem we’re learning, and the task is to recover a representation that encodes the part of that structure that affects the target. This is different from selecting features like in Invariant Causal Prediction. In particular, it provides a bridge from very low level features, such as individual pixels, to a representation encoding high level concepts, such as cows.

#### The causal direction

The idea of a latent causal system generating observed features is particularly useful as a view of computer vision problems. Computer vision researchers have long studied the generative processes involved in moving from real world objects to pixel representations.^[Longer than you may think! See, for instance, [Machine perception of three-dimensional solids](https://dspace.mit.edu/handle/1721.1/11589), published in 1963.] It’s instructive to inspect the causal structure of a dataset of cow pictures.

![When the features are the causes of the target, we say we are learning in the causal direction. When effects are the features, we are learning in the anti-causal direction.](figures/ff13-13.png)

In nature, cows exist in fields and on beaches, and we have an intuitive understanding that the cow itself and the ground are different things. A neural network trying to predict the presence of a cow in an image could be called an “anti-causal” learning problem, because the direction of causation is the opposite of the direction of prediction. The presence of a cow causes certain pixel patterns, but pixels are the input to the network, and the presence of a cow is the output.

However, a further sophistication can be added: the dataset on which we train a neural network is not learning from nature, but rather from human provided annotations. This changes the causal direction - we are now learning the effect from the cause - since those annotations are caused by the pixels of the image. This is the view taken by IRM^[The final section of the IRM paper includes a charming socratic dialogue that discusses this distinction, as well as the reason that regular supervised learning is so successful, from an invariance standpoint.], which thus interprets supervised learning from images as being a causal (rather than anticausal) problem.^[See [On Causal and Anticausal Learning](https://arxiv.org/abs/1206.6471) for a description of the insight considering the causal direction of a problem brings to machine learning.]

Not all supervised learning problems are causal. Anticausal supervised learning problems arise when the label is not provided based on the features, but by some other mechanism that causes the features. For example, in medical imaging, we could obtain a label without reference to the image itself by observing the case over time (this is not a recommended approach for treatment, of course).

Learning in the causal direction explains some of the success of supervised learning - there is a chance that it can recover invariant representations without modification. Any supervised learning algorithm is learning how to combine features to predict the target. If the learning direction is causal, each input is a potential cause of the output, and it’s possible that the features learned will be the true causes. The modifications that invariant risk minimization makes to the learning procedure improve the chance by specifically promoting invariance.

### How IRM works

To learn an invariant predictor, we must provide the IRM algorithm with data from multiple environments. As in ICP, these environments take the form of datasets, and as such the environments must be discrete. We need not specify the graphical or interventional structure associated with the environments. The motivating example of the IRM paper asks us to consider a machine learning system to distinguish cows from camels, highlighting a similar problem to that which [Recognition in Terra Incognita](https://arxiv.org/abs/1807.04975) does - animals being classified based on their environment, rather than the animal. In this case, cows on sand may be misclassified as camels, due to the spurious correlations absorbed by computer vision systems.

![In the IRM setup, we feed the algorithm data from multiple environments, and we must be explicit about which environment a data point belongs to.](figures/ff13-18.png)

Simply providing data from multiple environments is not enough. The problem of learning the optimal classifier in multiple environments is a bi-level constrained optimization problem, in which we must simultaneously find the optimal data representation and optimal classifier across multiple separate datasets. IRM reduces the problem to a single optimization loop, with the trick of using a constant classifier and introducing a new penalty term to the loss function.

```
IRM loss = sum over environments (error + penalty)
```

The `error` is the usual error we would use for the problem at hand - for example, the cross entropy for a classification problem - calculated on each environment. The technical definition of the new `penalty` term is the squared gradient norm with respect to a constant classifier, but it has an intuitive explanation. While the error measures how well the model is performing in each environment, the penalty measures how much the performance could be improved in each environment with one gradient step.

By including the penalty term in the loss, we punish high gradients - situations where a large improvement in an environment would be possible with one more epoch of learning. The result is a model with optimal performance in all environments. Without the IRM penalty, a model could minimize the loss by performing extremely well in just one environment, and poorly in others. Adding a term to account for the model having a low gradient (roughly, it has converged) in each environment ensures that the learning is balanced between environments.

To understand the IRM paradigm, we can perform a thought experiment. Imagine we have a dataset of cows and camels, and we’d like to learn to classify them as such. We separate out the dataset by the geolocation of photos - those taken in grassy areas from one environment, and those taken in deserts form another.

As a baseline, we perform regular supervised learning to learn a binary classifier between cows and camels. The learning principle at work in supervised learning is referred to as _empirical risk minimization_, or ERM - we’re just seeking to minimize the usual cross entropy loss.^[Technically, loss is the error on the training set, and risk is the error across the whole data distribution. With finite training data, minimizing the loss on the training set is a proxy for minimizing the risk.] We’ll surely find that we can get excellent predictive performance on these two environments, because we have explicitly provided data from both.

The trouble arises when we want to identify a cow on snow, and find that our classifier did not really learn to identify a cow. It learned to identify grass. The holdout performance of our model in any new environment we haven’t trained on will be poor.

![If we rely on empirical risk minimization, we learn spurious correlations between animals and their environments.](figures/ff13-19.png)

With IRM, we perform the training across (at least) two environments, and include the penalty term for each in the loss. We’ll almost certainly find that our performance in the training environments is reduced. However, because we have encouraged the learning of invariant features that transfer across environments, we’re more likely to be able to identify cows on snow. In fact, the very reason our performance in training is reduced is that we’ve not absorbed so many spurious correlations that would hurt prediction in new environments.

It is impossible to guarantee that a model trained with IRM learns _no_ spurious correlations. That depends entirely on the environments provided. If a particular feature is a useful discriminator in all environments, it may well be learned as an invariant feature, even if in reality it is spurious. As such, access to sufficiently diverse environments is paramount for IRM to succeed.

However, we should not be reckless in labeling something as an environment. Both ICP and IRM note that splitting on arbitrary variables in observational data can create diverse environments while destroying the very invariances we wish to learn. While IRM promotes invariance as the primary feature of causality, it pays to hold a structural model in the back of one’s mind, and ask if an environment definition makes sense as something that would alter the data generating process.

#### Considerations for applying IRM

IRM buys us extrapolation powers to new datasets, where independent and identically distributed supervised learning can at best interpolate between them. Using it to construct models improves their generalization properties by explicitly promoting performance across multiple environments, and leaves us with a new, closer-to-causal representation of the input features. Of course, this representation may not be perfect - IRM is an optimization-based procedure, and we will never know if we have found the true minimum risk across all environments - but it should be a step towards latent causal structure. This means that we can use our model to predict based on true, causal correlations, rather than spurious, environment-specific correlations.

However, there is no panacea, and IRM comes with some challenges.

Often, the dataset that we use in a machine learning project is collected well ahead of time, and may have been collected for an entirely different purpose. Even when a well-labeled dataset that is amenable to the problem exists, it is seldom accompanied by detailed metadata (by which we mean information about the information). As such, we often do not have information about the environment in which the data was collected.

![Most datasets are collected in a variety of environments, and without the metadata necessary to separate them. This presents a challenge for invariance-based approaches.](figures/ff13-20.png)

Another challenge is finding data from sufficiently diverse environments. If the environments are similar, IRM will be unlikely to learn features that generalize to environments that are different. This is both a blessing and a curse - on the one hand, we do not need to have perfectly separated environments to benefit from IRM, but on the other, we are limited by the diversity of environments. If a feature appears to be a good predictor in all the environments we have, IRM will not be able to distinguish that from a true causal feature. In general, the more environments we have, and the more diverse they are, the better IRM will do at learning an invariant predictor, and the closer we get to a causal representation.

![IRM relies on representative data from diverse environments. If we cannot collect enough data from sufficiently diverse environments, we may still learn spurious correlations.](figures/ff13-21.png)

No model is perfect, and whether or not one is appropriate to use depends on the objective. IRM is more likely to produce an invariant predictor, with good out-of-distribution performance, than empirical risk minimization (regular supervised learning), but doing so will come at the expense of predictive performance in the training environment. It’s entirely possible that for a given application, we are very sure that the data in the eventual test distribution (“in the wild”) will be distributed in the same way as our training data. Further, we may know that all we want to do with the resulting model is predict, not intervene. If both these things are true, we should  stick to supervised learning with empirical risk minimization and exploit all the spurious correlations we can.
