# Udacity-DeepLearning-Nanodegree
Udacity DeepLearning Nanodegree Projects
# Deep Learning (PyTorch)

This repository contains material related to Udacity's [Deep Learning Nanodegree program](https://www.udacity.com/course/deep-learning-nanodegree--nd101). It consists of a bunch of tutorial notebooks for various deep learning topics. In most cases, the notebooks lead you through implementing models such as convolutional networks, recurrent networks, and GANs. There are other topics covered such as weight initialization and batch normalization.

There are also notebooks used as projects for the Nanodegree program. In the program itself, the projects are reviewed by real people (Udacity reviewers), but the starting code is available here, as well.



## Projects

Project 1 : [Predicting Bike-Sharing Patterns](https://github.com/mhmohona/DeepLearning-NanodegreeProgram-Phase-2/tree/master/Project%201_Predicting%20Bike-Sharing%20Patterns%C2%A0%C2%A0)
-
![Project 1_Predicting Bike-Sharing Patterns](https://user-images.githubusercontent.com/14244685/55161620-2c301580-5190-11e9-8947-4535a4993201.PNG)

In this project, I have implemented a neural network in Numpy to predict bike rentals.

Project 2 :  [Dog Breed Classifier](https://github.com/mhmohona/DeepLearning-NanodegreeProgram-Phase-2/tree/master/Project%202_Dog%20Breed%20Classifier):
-
![dog breed](https://user-images.githubusercontent.com/14244685/56671322-40efc280-66d6-11e9-80b6-3a2fd9657d10.PNG)

In this project, I have built a convolutional neural network with PyTorch to classify any image (even an image of a face) as a specific dog breed.


Project 3 :  [TV Script Generation](https://github.com/mhmohona/Udacity-DeepLearning-NanodegreeProgram-Phase-2/tree/master/Project%203_Generate%20TV%20Scripts): 
-
![Generating TV script](https://user-images.githubusercontent.com/14244685/56514538-5b4f6200-6557-11e9-9235-084928059a5a.PNG)

In this project, I have trained a recurrent neural network to generate scripts in the style of dialogue from Seinfeld.

Project 4 :  [Face Generation](https://github.com/mhmohona/Udacity-DeepLearning-NanodegreeProgram-Phase-2/tree/master/Project%204_Generate%20Faces): 
-
![Face Generation](https://user-images.githubusercontent.com/14244685/56671622-cbd0bd00-66d6-11e9-9a8d-d8b8725a054e.PNG)

In this project, I have used a DCGAN on the CelebA dataset to generate images of new and realistic human faces.


## Table Of Contents

### Tutorials

### Introduction to Neural Networks

* [Introduction to Neural Networks](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-neural-networks): Learn how to implement gradient descent and apply it to predicting patterns in student admissions data.
* [Sentiment Analysis with NumPy](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-analysis-network): [Andrew Trask](http://iamtrask.github.io/) leads you through building a sentiment analysis model, predicting if some text is positive or negative.
* [Introduction to PyTorch](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-to-pytorch): Learn how to build neural networks in PyTorch and use pre-trained networks for state-of-the-art image classifiers.

### Convolutional Neural Networks

* [Convolutional Neural Networks](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/convolutional-neural-networks): Visualize the output of layers that make up a CNN. Learn how to define and train a CNN for classifying [MNIST data](https://en.wikipedia.org/wiki/MNIST_database), a handwritten digit database that is notorious in the fields of machine and deep learning. Also, define and train a CNN for classifying images in the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
* [Transfer Learning](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/transfer-learning). In practice, most people don't train their own networks on huge datasets; they use **pre-trained** networks such as VGGnet. Here you'll use VGGnet to help classify images of flowers without training an end-to-end network from scratch.
* [Weight Initialization](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/weight-initialization): Explore how initializing network weights affects performance.
* [Autoencoders](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/autoencoder): Build models for image compression and de-noising, using feedforward and convolutional networks in PyTorch.
* [Style Transfer](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/style-transfer): Extract style and content features from images, using a pre-trained network. Implement style transfer according to the paper, [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) by Gatys et. al. Define appropriate losses for iteratively creating a target, style-transferred image of your own design!

### Recurrent Neural Networks

* [Intro to Recurrent Networks (Time series & Character-level RNN)](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/recurrent-neural-networks): Recurrent neural networks are able to use information about the sequence of data, such as the sequence of characters in text; learn how tom implement these in PyTorch for a variety of tasks.
* [Embeddings (Word2Vec)](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/word2vec-embeddings): Implement the Word2Vec model to find semantic representations of words for use in natural language processing.
* [Sentiment Analysis RNN](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-rnn): Implement a recurrent neural network that can predict if the text of a moview review is positive or negative.
* [Attention](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/attention): Implement attention and apply it to annotation vectors.

### Generative Adversarial Networks

* [Generative Adversarial Network on MNIST](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/gan-mnist): Train a simple generative adversarial network on the MNIST dataset.
* [Batch Normalization](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/batch-norm): Learn how to improve training rates and network stability with batch normalizations.
* [Deep Convolutional GAN (DCGAN)](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/dcgan-svhn): Implement a DCGAN to generate new images based on the Street View House Numbers (SVHN) dataset.
* [CycleGAN](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/cycle-gan): Implement a CycleGAN that is designed to learn from unpaired and unlabeled data; use trained generators to transform images from summer to winter and vice versa.

### Deploying a Model (with AWS SageMaker)

* [All exercise and project notebooks](https://github.com/udacity/sagemaker-deployment) for the lessons on model deployment can be found in the linked, Github repo. Learn to deploy pre-trained models using AWS SageMaker.



### Elective Material

* [Intro to TensorFlow](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/tensorflow/intro-to-tensorflow): Starting building neural networks with TensorFlow.
* [Keras](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/keras): Learn to build neural networks and convolutional neural networks with Keras.

---