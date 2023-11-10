### Table of Contents

- [What is this project?](#what-is-this-project)
- [Definitions](#definitions)
- [The History of Machine Learning](#the-history-of-machine-learning)

<br>

# Machine Learning Basics

## What is this project?

This project is a collection of machine learning models organized by the following subcategories:
* supervised / unsupervised / semisupervised / reinforcement
* classification / regression / clustering / dimensionality reduction / anomaly detection / ensemble models / generative AI
* linear / nonlinear

It is still a work in progress, so not all subcategories are covered.

## Definitions

* The field of *Artificial Intelligence* seeks to automate intellectual tasks normally performed by humans.

* *Machine Learning*: A machine learning model transforms its input data into meaningful outputs, a process that is "learned" from exposure to known examples of inputs and outputs. Therefore, the central problem in machine learning is to meaningfully transform data. That is, to learn useful representations of the input data. They are useful in the sense that they allow simple rules to solve the problem in that representation space. "Learning" describes an automatic search process for useful data transformations – ones that produce representations of the input data that are amenabele to simpler rules solving the task at hand – guided by some feedback signal. This search for useful transformations is over a predefined space of possibilities.

## The History of Machine Learning

Being familiar with the history of machine learning and understanding the context in which models were developed is both helpful and informative. Not only does the historical context provide a solid foundation for understanding today's cutting-edge models, but it demonstrates the remarkable speed at which this technology is developing.

### Timeline

**1944**: *Logistic Regression* is a machine learning classification algorithm introduced in 1944.


**1950s**: 

   * Alan Turing published his landmark paper, "Computing Machinery and Intelligence". Turing was of the opinion – highly provocative at the time – that computers could in principle be made to emulate all aspects of human intelligence.
     
   * The core ideas of neutral networks were first investigated in toy form, but the approach took decades to get started. A *neural network* is a multi-stage data transformation pipeline, where each layer performs some transformation. The specification of what a layer does to its input data is stored in the layer's weights (parameters), which in essence are a bunch of numbers. So in this context, "learning" means finding a set of values for the weights of all layers in a network such that the network will correctly map example inputs to their associated targets. Neural networks can contain tens of millions of parameters. Initially, the weights of the network are assigned random values. A *loss function* is a function that computes a distance score between a prediction of the network and the true target, capturing how well the network has done on this specific example. The fundamental trick in deep learning is to use that score as a feedback signal to adjust the value of the weights a little, in a direction that will lower the loss score for the current example. That adjustment is the job of the *optimizer*, which implements what's called the *Backpropogation algorithm*: the central algorithm in deep learning. The weights are continually adjusted in this training loop until the loss score is minimized, and the network is trained.


* **1956**: John McCarthy, then a young Assistant Professor of Mathematics at Dartmouth College organized a summer workshop under the following proposal:


    *The study is to proceed on the basis of the conjecture that every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to understand it. An attempt will be made to find how to make machines use language, form abstractios and concepts, solve kinds of problems now reserved for humans, and improve themselves.*


* **1950s - 1980s**: *Symbolic AI*, where programmers handcrafted explicit rules for manipulating data was the dominant paradigm in AI. Suitable to solve well-defined, logical problems, such as playing chess. Unable to solve more complex, fuzzy problems, such as image classification, speech recognition, or natural language translation.


* **1960s**: 

    * The *Naive Bayes algorithm* is a machine learning classifier introduced in the 1960s that applies Bayes' theorem while assuming that the features in the input data are all indepdenent.
    
    * 1963: An older linear formulation of *Support Vector Machines (SVMs)* was published. SVM is a classification algorithm that works by finding "decision boundaries" separating two classes by 1) mapping to a new high-dimensional representation (using a *kernel function* - a function that maps any two points in your initial space to the distance between these points in your target representation space, bypassing the explicit computation of the new representation) where the decision boundary can be expressed as a hyperplane and 2) maximizing  the distance between the hyperplane and the closest data points from each class (*maximizing the margin*).


* **1980s**: The *Backpropagation algorithm*, a way to train chains of parametric operatioins using gradient-descent optimization, was rediscovered and applied to neural networks.

    * 1989: Yann LeCun at Bell Labs introduced LeNet for classifying handwritten digits. It was used by the United States Postal Service in the 1990s to automate the the reading of ZIP codes on mail envelopes.

* **1990s**: *Machine Learning*, started to flourish. It has quickly become the most popular and most successful subfield of AI, a trend driven by the availability of faster hardware and larger datasets. 

    * 1995: Bell Labs published the modern formulation of *Support Vector Machines (SVMs)*. They were well understood and easily intrepretable, so they became extremely popular in the field for a long time. But they were hard to scale and didn't provide good results for preceptual problems such as image classification because of the *feature engineering* required (extracting useful representations manually)

    * 1997: *The Long Short-Term Memory (LSTM) algorithm*, fundamental to deep learning for time series data, was developed.

* **Early 2000s**: *Decision Trees*, flowchart-like structures that let you classify input data, began to receive significant research interest. By the end of the decade they were often preferred to kernel methods. The *Random Forest* algorithm was introduced, which involves building a large number of specialized decision trees and then ensembling their outputs.


* **Early 2010s**: *Deep Learning*, a new take on learning representations from data that puts an emphasis on learning successive layers of increasingly meaningful representations, started to take off. Modern deep learning often involves tens or even hundreds of successive layers of representations, and they're all learned automatically from exposure to training data. Meanwhile, other approaches to machine learning tend to focus on learning only one or two layers of represenatations. In deep learning, these layered representations are learned via models called neural networks. 

    * 2006: The ImageNet dataset was released. If there's one dataset that has been a catalyst for the rise of deep learning, it's this one. It consisted of 1.4 million images that have been hand annotated with 1000 image categories.

    * 2007: NVIDIA invested billions of dollars in developing fast, massively parallel chips (GPU) to power the graphics of video games to render complex 3D scenes on your screen in real time. THey laucnhed CUDA, a programming interface for its line of GPUs.

    * 2010: 
        * Off-the-shelf CPUs were now faster than in 1990 by a factor of 5,000.
        * Several algorithmic inventions allowed for better gradient propogation to train deep neural networks with 10+ layers. Deep learning started to shine.
            * Better activation functions for neural layers
            * Better weight-initialization schemes
            * Better optimizers (RMSProp and Adam)

    * 2011: 

        * Researchers bgan to write CUDA implelmentations of neural nets (since they mostly consist of many small matrix multiplications, they are highly parallelizable)

        * Dan Ciresan from IDSIA began to win academic image-classification competitions with GPU-trained deep neural networks. This was the first practical success of modern deep learning.

    * 2012: The watershed moment when Geofrey Hinton's group from the University of Toronto entered into the yearly image-classification challenge ImageNet (required classifying high resolution color images into 1000 different categories after training on 1.4 million images). They archieved a top-five accuracy of 83.6%. Deep CNNs became the go-to algorithm for all perceptual tasks.

    * Deep learning started to completely replace SVMs and decision trees in a wide range of applications. 

* **2014**: *Gradient Boosting Machines* took over as the best algorithm for any shallow machine learning task. A gradient boosting machine is like random forest, but it uses gradient boosting, as a way to improve the model by iteratively training new models that specialize in addressing the weak points of the previous models. It is still the best algorithm for dealing with nonperceptual data today.


* **2015**: The Keras library was released and quickly became the go-to deep learning solution.

* **2016**: 

    * Google revealed its Tensor Processing Unit (TPU) project – a new chip design developed from the ground up to run deep neural networks significationly faster and far more energy efficiently than top-of-the-line GPUs. TPU cards are designed for be assembled into large-scale configureations called "pods".

    * Even more advanced ways to improve gradient propagation were discovered. Large-scale model architectures brought critical advances in computer vision and natural language processing (e.g. 
        * Batch normalization
        * Residual connections
        * Depthwise separable convolutions)



* **2016-2020**: The entire machine learning and data science industry has been dominated by deep learning (for percentual problems) and gradient boosted trees (when structured data is available).

* **2017**: Transformer-based deep learning models for natural language processing (such as BERT, GPT-3) were developed and revolutionized the field.
