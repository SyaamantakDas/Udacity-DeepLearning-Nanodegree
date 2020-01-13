# Udacity-DeepLearning-Nanodegree
Udacity DeepLearning Nanodegree Projects
# Deep Learning (PyTorch)

This repository contains material related to Udacity's [Deep Learning Nanodegree program](https://www.udacity.com/course/deep-learning-nanodegree--nd101). It consists of a bunch of tutorial notebooks for various deep learning topics. In most cases, the notebooks lead you through implementing models such as convolutional networks, recurrent networks, and GANs. There are other topics covered such as weight initialization and batch normalization.

You can get the code used in this nanodegree from the official Udacity repo on Github. You can clone this [link](https://github.com/udacity/deep-learning)

## Program Structure

### 1. Introduction 

The first part is an introduction to the program as well as a couple lessons covering tools you'll be using. You'll also get a chance to apply some deep learning models to do cool things like transferring the style of artwork to another image.

We’ll start off with a simple introduction to linear regression and machine learning. This will give you the vocabulary you need to understand recent advancements, and make clear where deep learning fits into the broader picture of Machine Learning techniques.



### 2. Neural Networks

In this part, you'll learn how to build a simple neural network from scratch using python. We'll cover the algorithms used to train networks such as gradient descent and backpropagation. The first project is also available this week. In this project, you'll predict bike ridership using a simple neural network.

You'll also learn about model evaluation and validation, an important technique for training and assessing neural networks. We also have guest instructor Andrew Trask, author of [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning), developing a neural network for processing text and predicting sentiment.



### 3. Convolutional Networks

Convolutional networks have achieved state of the art results in computer vision. These types of networks can detect and identify objects in images. You'll learn how to build convolutional networks in TensorFlow. You'll also get the second project, where you'll build a convolutional network to classify dog breeds in pictures.

You'll also use convolutional networks to build an *autoencoder*, a network architecture used for image compression and denoising. Then, you'll use a pretrained neural network ([VGGnet](https://arxiv.org/pdf/1409.1556.pdf)), to classify images of flowers the network has never seen before, a technique known as *transfer learning*.



### 4. Recurrent Neural Networks

In this part, you’ll learn about Recurrent Neural Networks (RNNs) — a type of network architecture particularly well suited to data that forms sequences like text, music, and time series data. You'll build a recurrent neural network that can generate new text character by character.

Then, you'll learn about word embeddings and implement the [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) model, a network that can learn about semantic relationships between words. These are used to increase the efficiency of networks when you're processing text.

You'll combine embeddings and an RNN to predict the sentiment of movie reviews, an example of common tasks in natural language processing.

In the third project, you'll use what you've learned here to generate new TV scripts from episodes of The Simpson's.



### 5. Generative Adversarial Networks

Generative adversarial networks (GANs) are one of the newest and most exciting deep learning architectures, showing incredible capacity for understanding real-world data. The networks can be used for generating images such as the [CycleGAN](https://github.com/junyanz/CycleGAN) project.

The inventor of GANs, Ian Goodfellow, will show you how GANs work and how to implement them. You'll also learn about semi-supervised learning, a technique for training classifiers with data mostly missing labels.

In the fourth project, you'll use a deep convolutional GAN to generate completely new images of human faces.



## Projects

Project 1 : [Predicting Bike-Sharing Patterns]()
-
![Project 1_Predicting Bike-Sharing Patterns](https://user-images.githubusercontent.com/14244685/55161620-2c301580-5190-11e9-8947-4535a4993201.PNG)

In this project, I have implemented a neural network in Numpy to predict bike rentals.

Project 2 :  [Dog Breed Classifier]():
-
![dog breed](https://user-images.githubusercontent.com/14244685/56671322-40efc280-66d6-11e9-80b6-3a2fd9657d10.PNG)

In this project, I have built a convolutional neural network with PyTorch to classify any image (even an image of a face) as a specific dog breed.


Project 3 :  [TV Script Generation](): 
-
![Generating TV script](https://user-images.githubusercontent.com/14244685/56514538-5b4f6200-6557-11e9-9235-084928059a5a.PNG)

In this project, I have trained a recurrent neural network to generate scripts in the style of dialogue from Seinfeld.

Project 4 :  [Face Generation](): 
-
![Face Generation](https://user-images.githubusercontent.com/14244685/56671622-cbd0bd00-66d6-11e9-9a8d-d8b8725a054e.PNG)

In this project, I have used a DCGAN on the CelebA dataset to generate images of new and realistic human faces.


Project 5 :  [Sentiment Analysis](): 
[Sentiment Analysis Web App](https://github.com/udacity/sagemaker-deployment/tree/master/Project) is a notebook and collection of Python files to be completed. The result is a deployed RNN performing sentiment analysis on movie reviews complete with publicly accessible API and a simple web page which interacts with the deployed endpoint. This project assumes that you have some familiarity with SageMaker. Completing the XGBoost Sentiment Analysis notebook should suffice.


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
