## Prototype

The promise of Invariant Risk Minimization (IRM), namely greatly improved out-of-distribution generalization using a representation that is closer-to-causal, is tempting. The paper performs some experiments that clearly show the method works when applied to an artificial structural causal model. Further an experiment in which an artificial spurious correlation is injected into the MNIST dataset (by coloring the images) is detailed, and works.

In order to gain a better understanding of the algorithm and investigate further we wanted to test the same technique in a less artificial scenario, on a natural image dataset.

### The Wildcam dataset

The iWildCam 2019 dataset (from The iWildCam 2019 Challenge Dataset) consists of wildlife images taken using camera traps. In particular, the dataset contains the Caltech Camera Traps (CCT) dataset, on which we focus. The CCT dataset contains 292,732 images, with each image labeled as containing one of 13 animals, or none. The images are collected from 143 locations, and feature a variety of weather conditions and all times of day. The challenge is to identify the animal present in the image.

![Left, a coyote in its natural environment. Right, a raccoon in the same location at night. Image credit: The [iWildCam 2019 Challenge Dataset](https://arxiv.org/abs/1907.07617), used under the [Community Data License Agreement](https://cdla.io/permissive-1-0/).](figures/wildcam-coyote-raccoon.png)

### Experimental set-up

This set-up maps naturally to the environmental splits used in IRM. Each camera trap location is a distinct physical environment, which is roughly consistent, allowing for seasonal, weather and day/night patterns. No two environments are the same, though the camera locations are spread around roughly the same geographic region (the America Southwest).

The objects of interest in the dataset are animals, which are basically invariant across environments: a raccoon looks like a raccoon in the mountains and in your backyard (though the particular raccoon may be different). The images are not split evenly between environments, since there is more animal activity in some places than others. Nor are the animal species evenly distributed among cameras. Some cameras will primarily produce images of one species or another, depending on the animals active in the area.

If we were to naively train a model using empirical risk on a subset of cameras, we could well end up learning exactly those class imbalances. If 99% of the images from camera 1 are labeled as deer, then we could have a 99% accurate classifier by learning to recognize the fallen tree that is present only in camera 1, rather than the deer themselves. Clearly such a classifier has not really learned to recognize deer, and would be useless for predicting in another environment.

We want to learn to recognize the animals themselves. The IRM setup seems ideally suited to address this challenge.

To validate the approach, we restricted our experiment to only three cameras and two animal species, which were randomly chosen. Of the three cameras, two were used as training environments, and one as a held-out environment for testing. The task was binary classification: distinguish coyotes from raccoons. We used Resnet18, a pretrained classifier trained on the much larger ImageNet dataset as a feature extractor with a final fully connected layer with sigmoid output, which we tuned to the problem at hand.

Each of the environments contained images of both coyotes and racoons. Even this reduced dataset exhibits several challenges typical to real world computer vision: some images are dark, some are blurred, some are labelled as containing an animal when only the foot of the animal is visible, and some feature nothing but a patch of fur covering the lens. We saw some success simply ignoring these problems, but ultimately manually selected only those images clearly showing an identifiable coyote or raccoon.

### Results

When tackling any supervised learning problem, it’s a good idea to set up a simple baseline against which to compare performance. In the case of a binary classifier, an appropriate baseline model is to always predict the majority class of the training set. The three environments had a class balance as shown in the table below. The majority class in both train environments is coyote, so our baseline accuracy is the accuracy if we always predict the animal is a coyote, regardless of environment or input image.

|                   | Train environment 1 | Train environment 2 | Test     |
|-------------------|:-------------------:|:-------------------:|:--------:|
| Coyotes           | 582                 | 512                 | 144      |
| Raccoons          | 276                 | 241                 | 378      |
| Baseline accuracy | 68%                 | 68%                 | 28%      |

When we treat the problem with empirical risk minimization - minimizing the cross entropy between classes - we found good performance in the train environments, but very poor performance in the test environment. We report the metrics over 120 epochs of training in the table below.

|        | ERM            |               |                |             | IRM            |               |                |             |
|:------:|:--------------:|:-------------:|:--------------:|:-----------:|:--------------:|:-------------:|:--------------:|:-----------:|
| epoch  | train accuracy | test accuracy | test precision | test recall | train accuracy | test accuracy | test precision | test recall |
| 0      | 30.8%          | 27.0%         | 57.7%          | 4.0%        | 30.8%          | 26.9%         | 56.0%          | 3.7%        |
| 10     | 68.2%          | 28.1%         | 0.0%           | 0.0%        | 68.2%          | 28.0%         | 0.0%           | 0.0%        |
| 20     | 84.0%          | 35.5%         | 73.4%          | 18.3%       | 79.6%          | 26.4%         | 64.3%          | 2.4%        |
| 30     | 84.5%          | 31.5%         | 65.9%          | 7.1%        | 81.3%          | 29.7%         | 60.0%          | 1.6%        |
| 40     | 86.2%          | 36.1%         | 74.0%          | 19.6%       | 85.5%          | 34.0%         | 72.4%          | 16.7%       |
| 50     | 86.3%          | 31.7%         | 73.6%          | 10.3%       | 85.6%          | 39.2%         | 75.2%          | 24.9%       |
| 60     | 87.9%          | 35.0%         | 73.6%          | 14.0%       | 86.7%          | 41.2%         | 76.0%          | 27.5%       |
| 70     | 88.4%          | 30.5%         | 74.5%          | 10.1%       | 85.8%          | 43.6%         | 76.2%          | 28.8%       |
| 90     | 89.6%          | 29.5%         | 74.4%          | 7.7%        | 85.8%          | 59.4%         | 81.2%          | 56.1%       |
| 80     | 89.3%          | 30.6%         | 72.7%          | 6.3%        | 86.5%          | 51.9%         | 80.3%          | 39.9%       |
| 100    | 90.6%          | 29.9%         | 71.9%          | 6.1%        | 84.9%          | 75.1%         | 83.1%          | 79.4%       |
| 110    | 90.0%          | 29.3%         | 75.0%          | 5.6%        | 84.7%          | 75.2%         | 83.2%          | 84.1%       |
| 120    | 91.3%          | 33.2%         | 75.6%          | 9.0%        | 85.0%          | 78.7%         | 83.5%          | 88.4%       |

