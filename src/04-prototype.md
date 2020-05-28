## Prototype

The promise of Invariant Risk Minimization (greatly improved out-of-distribution generalization using a representation that is closer-to-causal) is tempting. The IRM paper performs some experiments that clearly show the method works when applied to an artificial structural causal model. Further, an experiment in which an artificial spurious correlation is injected into the MNIST dataset (by coloring the images) is detailed, and works.

In order to gain a better understanding of the algorithm and investigate further, we wanted to test the same technique in a less artificial scenario: on a natural image dataset.

### The Wildcam dataset

The [iWildCam 2019 dataset](https://www.kaggle.com/c/iwildcam-2019-fgvc6) (from The iWildCam 2019 Challenge Dataset) consists of wildlife images taken using camera traps. In particular, the dataset contains the Caltech Camera Traps (CCT) dataset, on which we focus. The CCT dataset contains 292,732 images, with each image labeled as containing one of 13 animals, or none. The images are collected from 143 locations, and feature a variety of weather conditions and all times of day. The challenge is to identify the animal present in the image.

![Left, a coyote in its natural environment. Right, a raccoon in the same location at night. Image credit: The [iWildCam 2019 Challenge Dataset](https://arxiv.org/abs/1907.07617), used under the [Community Data License Agreement](https://cdla.io/permissive-1-0/).](figures/wildcam-coyote-raccoon.png)

### Experimental setup

This setup maps naturally to the environmental splits used in IRM. Each camera trap location is a distinct physical environment which is roughly consistent, allowing for seasonal, weather, and day/night patterns. No two environments are the same, though the camera locations are spread around roughly the same geographic region (the American Southwest).

The objects of interest in the dataset are animals, which are basically invariant across environments: a raccoon looks like a raccoon in the mountains and in your backyard (though the particular raccoon may be different). The images are not split evenly between environments, since there is more animal activity in some places than others. Nor are the animal species evenly distributed among cameras. Some cameras will primarily produce images of one species or another, depending on the animals active in the area.

If we were to naively train a model using empirical risk on a subset of cameras, we could well end up learning exactly those class imbalances. If 99% of the images from camera 1 are labeled as deer, then we could have a 99% accurate classifier by learning to recognize the fallen tree that is present only in camera 1, rather than the deer themselves. Clearly such a classifier has not really learned to recognize deer, and would be useless for predicting in another environment.

We want to learn to recognize the animals themselves. The IRM setup seems ideally suited to address this challenge.

To validate the approach, we restricted our experiment to only three cameras and two animal species, which were randomly chosen. Of the three cameras, two were used as training environments, and one as a held-out environment for testing. The task was binary classification: distinguish coyotes from raccoons. We used [ResNet18](https://arxiv.org/abs/1512.03385), a pretrained classifier trained on the much larger ImageNet dataset, as a feature extractor with a final fully connected layer with sigmoid output, which we tuned to the problem at hand.

Each of the environments contained images of both coyotes and racoons. Even this reduced dataset exhibited several challenges typical to real world computer vision: some images were dark, some were blurred, some were labeled as containing an animal when only the foot of the animal was visible, and some featured nothing but a patch of fur covering the lens. We saw some success simply ignoring these problems, but ultimately manually selected only those images clearly showing an identifiable coyote or raccoon.

### Results

When tackling any supervised learning problem, it’s a good idea to set up a simple baseline against which to compare performance. In the case of a binary classifier, an appropriate baseline model is to always predict the majority class of the training set. The three environments had a class balance as shown in the table below. The majority class in both train environments is coyote, so our baseline accuracy is the accuracy if we always predict the animal is a coyote, regardless of environment or input image.

|                   | Train environment 1 | Train environment 2 | Test     |
|-------------------|:-------------------:|:-------------------:|:--------:|
| Coyotes           | 582                 | 512                 | 144      |
| Raccoons          | 276                 | 241                 | 378      |
| Baseline accuracy | 68%                 | 68%                 | 28%      |

When we treated the problem with empirical risk minimization (minimizing the cross-entropy between classes), we found good performance in the train environments, but very poor performance in the test environment. We report the metrics over 120 epochs of training in the table below. The best test accuracy is achieved at epoch 40, after which ERM (empirical risk minimization) begins to overfit. In the case of IRM (invariant risk minimization), we paid a small price in train set accuracy, but achieved much better test results - again, reporting the highest test accuracy achieved in 120 epochs (at epoch 120).

![Table comparing metrics on the combined train set and test set for empirical risk minimization (ERM) and invariant risk minimization (IRM).](figures/erm-vs-irm-table.png)

ERM outperforms the baseline in all environments, but not by too much in the new test environment. This can be attributed to the learning of spurious correlations. The network was able to effectively distinguish between raccoons and coyotes in the training environments, but the features it relied upon to do so were not general enough to help prediction much in the test environment.

In contrast, IRM loses a single percentage point of accuracy in the train environments, but performs almost as well in the test environment. The feature representation IRM constructs has translated between different environments effectively, and proves an effective discriminator.

As a practical point, we found that IRM worked best when the additional IRM penalty term was not added to the loss until the point at which ERM had reached its best performance - in this case the 40th training epoch. As such, ERM and IRM had identical training routines and performance until this point. When we introduced the IRM penalty, the IRM procedure continued to learn and gain out-of-distribution generalization capability, whereas ERM began to overfit. By the 120th epoch, IRM had the accuracy reported above, whereas ERM had achieved 91% in the combined training environments, at the cost of reducing its test accuracy by a few percentage points to 33%.

#### Interpretability

IRM yields impressive results, especially considering how hard it is to learn from these images. It has a clear and significant improvement in when compared to ERM in a new environment. In this section, we examine a few concrete examples of successes and failures of our prototype model and our speculations as to why they may be.

It would be nice to have a better sense of whether IRM has learned invariant features. By that we mean, whether it has learned to spot a raccoon’s long bushy tail or a coyote’s slender head, instead of the terrain or foliage in the image. Understanding which parts of the image contribute towards IRM’s performance is a powerful proposition. The classification task itself is hard: if you closely look at some of the images in the Wildcam dataset, at a first glance it’s even hard for us, humans, to point out where exactly the animal is. An interpretability technique like Local Interpretable Model-agnostic Explanations ([LIME](https://arxiv.org/abs/1602.04938)) provides valuable insights into how that classification is working.

LIME is an explanation technique that can be applied to almost any type of classification model — our report [FF06: Interpretability](https://ff06-2020.fastforwardlabs.com/) discusses these possibilities — but here we will consider its application to image data. LIME is a way to understand how different parts of an input affect the output of a model. This is accomplished, essentially, by turning the dials of the input and observing the effect on the output.

Let’s first try and understand how LIME works at a high level - including what inputs we need to provide, and what to expect as output - through a sample image in the test set. The image on the left of the figure below is a sample raw image of a coyote with dimensions height=747 and width=1024, as were all images in the dataset.

![Left: a raw Wildcam image. Right: Having been cropped and scaled to the input dimensions required by ResNet18.](figures/coyote-resized.png)

To use the IRM model, we must first perform some image transformations like resizing, cropping, and normalization - using the same transformations that we did when training the model. The input image then appears as shown on the right of the figure above, a normalized, 224 * 224 image. The transformed image when scored by the IRM model outputs a probability of 98% (0.98) for the coyote class! So yes, our model is pretty confident of its prediction.

Now, let’s see how LIME works on this image. First, LIME constructs a local linear model, and makes a prediction for the image. For the example image, the predicted score is 0.95, pretty close to the IRM model. When trying to explain the prediction, LIME uses interpretable representations. For images, interpretable representations are basically contiguous patches of similar pixels called superpixels. The superpixels for an image are generated by a standard algorithm, QuickShift, in the LIME implementation. The left panel in the figure below shows all of the 34 superpixels generated by LIME for the example image.

![LIME masks random combinations of superpixels, generated by QuickShift, to build a local linear model.](figures/lime-masks.png)

It then creates versions of the original image by randomly masking different combinations of the superpixels as shown in the middle and right panes of the above figure. Each random set of masked superpixels is one perturbation of the image. The modeler chooses the number of perturbations; in our case, we used 1000 perturbations of the original image. LIME then builds a regression model on all these perturbed images and determines the superpixels that contributed most towards the prediction, based on their weights.

The figure below shows the superpixel explanations (with the rest of the image grayed out) for the top 12 features that contribute towards the prediction of the coyote classification. While there are quite a few features that are mostly spurious covering the foliage or terrain, one of them covers the entire body of the coyote. Looking at these explanations provides an alternative way of assessing the IRM model and can enhance our trust that the model is learning to rely on sensible features.

![The non-grayed-out pixels correspond to the top 12 superpixels that contribute positively to the Coyote classification for the IRM model.](figures/irm-top-12.png)

Now when we generate the top 12 LIME explanations for the same image but based on the ERM model, they seem to capture more of the surroundings, rather than any of the coyote’s body parts.

![The non-grayed-out pixels correspond to the top 12 superpixels that contribute positively to the Coyote classification for the ERM model. In this case, they didn't catch much of the coyote.](figures/erm-top-12.png)

And then there are instances where LIME explanations seem to rely on spurious features. For example, in the figure below, the original image is classified as a coyote by the IRM model with a probability of 72% (0.72), whereas the LIME score is close to 0.53. The superpixels contributing towards the classification for both the IRM and ERM models usually cover the terrain or foliage, though some outline the coyote’s body.

![In this instance, both models seem to be relying on environmental features to predict Coyote.](figures/spurious-coyote.png)

We observe that the explanations make more intuitive sense when the LIME score is close to the model score.

IRM can only learn to be invariant with respect to the invariants that the environments encode. If there are spurious correlations that are the same across environments, IRM will not distinguish them from invariant features.

One feature that appears invariant in this dataset is the day or night cycle. Raccoons appear exclusively at night, and IRM could well learn that night means raccoon, and rely on it heavily. This correlation is spurious; a raccoon is still a raccoon during the day! However, we would need more environments, including images of raccoons in the daytime, to disentangle that.

The representation that IRM extracts from an environment should theoretically be closer to encoding the latent causal structure of the problem than that which ERM extracts. In our scenario, we might expect that IRM learns to focus more on the actual animal in the picture, since the presence of the animal is the cause of a given annotation. The animals change little between environments, whereas environmental features (like foliage) are completely different at different camera trap locations. Thus, the causal features ought to be invariant between environments.

That said, although the IRM results appear promising for some samples, it is hard to confirm that there is an obvious pattern, and this can be attributed to both the model and the interpretability technique. We chose to train only the last layer of ResNet18 to come up with the IRM model. This choice has an inherent drawback: the capacity for feature learning is low. As such, we wouldn’t expect perfect results, since it’s unlikely that the pretrained ResNet representations map perfectly to raccoons and coyotes.^[Imperfect interpretability results notwithstanding, using ResNet as a feature extractor is representative of how CV systems are used in the real world, and the resulting out-of-distribution performance improvements are impressive.]

Further, although an explanation of an image provides some reassurance of the quality of the model, it’s probably still insufficient to provide an overall picture of the _kind_ of features a given model is using, aggregated from all the individual explanations. And even though explanations for multiple images are insightful, these have to be judiciously selected. When it comes to text or tabular data, there are ways to determine the global feature importances, because the features in tabular data or vocabulary stay consistent across all the data points. The superpixels of an image cannot be consistent across all the images, which makes it really hard to assess whether the explanations make sense. Developing tools to understand large image datasets is a worthy endeavour!

### Product: Scene

![The [Scene prototype](https://scene.fastforwardlabs.com)](figures/scene.png)

To accompany this report, we built a prototype called [Scene](https://scene.fastforwardlabs.com) that takes you on a guided tour through the dataset, models, and results of our experiment. With Scene, we really wanted to give people a feel for the images that make up the dataset. Each panel of the tour features 16 images from the dataset, cropped and resized to the same dimensions of the images that the model is trained on. Many of the images featured are randomly sampled from the dataset when we generate the page, while others we specifically selected to use as examples. We hope that the amount and variety of images shown helps people get an intuitive feel for the dataset.

![View all the images in the dataset on the [all](https://scene.fastforwardlabs.com/all) page.](figures/scene-all.png)

If you want to go even deeper, we included an [all](https://scene.fastforwardlabs.com/all) page, which shows all 2,133 images in the dataset, along with the predictions and interpretability visualizations for each model. It's nice to be able to use these visualizations to check intuitions (like which features are important to each model) with your own eyes. Of course, even having access to all the images doesn't mean you can see "the big picture." It's difficult to hold everything you've seen in your head as you scroll through. If you're not careful, you'll end up generalizing the patterns you've seen most recently to the entire dataset. This is the challenge of visualizing the scale of the data that machine learning systems take in. Other techniques, like embeddings (as seen in our [Active Learner](https://activelearner.fastforwardlabs.com/) prototype) can help you visualize patterns, but then you lose some of the detail gained by being able to see the images up close. No one technique can give you the whole picture; data visualization requires a variety of techniques.

Generating such a large number of images, complete with text labels and interpretability overlays, was an interesting technical challenge. Originally, we'd planned to have Scene animate transitions between the original image and the interpretability overlays. To do this efficiently in a browser, you generate a "sprite sheet" - a large image that contains all the different animation states you'll transition through (a technique borrowed from video games). It was while we were generating the sprite sheets that we decided that, rather than transitioning through them one at a time, it would be more effective to show the entire sheet. Having more images visible together made comparisons easier and the scale of the dataset more clear. We ended up using the [node-canvas](https://activelearner.fastforwardlabs.com/) package to crop and place the images, overlay the interpretability layers, and apply the labels through a node script. Since we do all the work of generating images locally, we guarantee the user as snappy an experience as possible. Static site generation has seen renewed interest as a web-development strategy, and could be especially useful for large-scale data-visualization.
