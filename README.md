# Machine Learning Basics

- [Project Overview](#project-overview)
- [Definitions](#definitions)
- [The History of Machine Learning](#the-history-of-machine-learning)
- [The Reason Machine Learning Works: The Manifold Hypothesis](#the-reason-machine-learning-works-the-manifold-hypothesis)
- [The Goal of Machine Learning: Generalization](#the-goal-of-machine-learning-generalization)
- [The Workflow of Machine Learning](#the-workflow-of-machine-learning)
  - [1. Define the task](#1-define-the-task)
  - [2. Develop the model](#2-develop-the-model)
    - [Prepare the data](#prepare-the-data)
    - [Choose an evaluation protocol](#choose-an-evaluation-protocol)
    - [Train a model that has generalization power](#train-a-model-that-has-generalization-power)
        - [Feature selection](#feature-selection)
        - [Feature engineering](#feature-engineering)
        - [Early stopping](#early-stopping)
    - [Train a model that can overfit](#train-a-model-that-can-overfit)
    - [Regularize and tune your model](#regularize-and-tune-your-model)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
      - [What To Do If Training Won't Start](#what-to-do-if-training-wont-start)
      - [What To Do If Your Model Won't Start To Overfit](#what-to-do-if-your-model-wont-start-to-overfit)
  - [3. Deploy the model](#3-deploy-the-model)
      - [How to deploy your model via an API (self-hosting)](#how-to-deploy-your-model-via-an-api-self-hosting)
      - [How to deploy your model via an API (third-party hosting)](#how-to-deploy-your-model-via-an-api-third-party-hosting)
      - [How to deploy a model on a website](#how-to-deploy-a-model-on-a-website)
      - [How to deploy a model on a personal device](#how-to-deploy-a-model-on-a-personal-device)
      - [How to optimize your model](#how-to-optimize-your-model)
  - [4. Monitor your model in the wild](#4-monitor-your-model-in-the-wild)
- [Reference](#reference)
  - [How To Improve Generalization](#how-to-improve-generalization)
  - [Last-layer activation and loss functions for different tasks](#last-layer-activation-and-loss-functions-for-different-tasks)




## Project Overview

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

    * 2013: Surprisingly, these early successes weren't enough make make deep learning mainstream at the time. It still faced intense skepticism from many senior computer vision researchers.

* **2014**: *Gradient Boosting Machines* took over as the best algorithm for any shallow machine learning task. A gradient boosting machine is like random forest, but it uses gradient boosting, as a way to improve the model by iteratively training new models that specialize in addressing the weak points of the previous models. It is still the best algorithm for dealing with nonperceptual data today.


* **2015**: The Keras library was released and quickly became the go-to deep learning solution.

* **2016**: 

    * Deep learning became dominant.

    * Google revealed its Tensor Processing Unit (TPU) project – a new chip design developed from the ground up to run deep neural networks significationly faster and far more energy efficiently than top-of-the-line GPUs. TPU cards are designed for be assembled into large-scale configureations called "pods".

    * Even more advanced ways to improve gradient propagation were discovered. Large-scale model architectures brought critical advances in computer vision and natural language processing (e.g. 
        * Batch normalization
        * Residual connections
        * Depthwise separable convolutions)



* **2016-2020**: The entire machine learning and data science industry has been dominated by deep learning (for percentual problems) and gradient boosted trees (when structured data is available).

* **2017**: Transformer-based deep learning models for natural language processing (such as BERT, GPT-3) were developed and revolutionized the field.

## The Reason Machine Learning Works: The Manifold Hypothesis

- A remarkable fact about deep learning models is that they can be trained to fit anything, as long as they have enough representational power. If all the labels were random, it would just end up memorizing mappings of specific inputs to targets, much like a dictionary. If this is the case, then how come deep learning models generalize at all?
- A **manifold** is a lower-dimensional subspace of some parent space that is locally similar to a linear space. For instance, a smooth surface within a 3D space is a 2D manifold, and so on. More generally, **the manifold hypothesis** *posits that all natural data lies on a low-dimensional manifold within the high-dimensional space where it is encoded.* **That's a pretty strong statement about the structure of the universe. As far as we know, it's accurate, and it's the reason why deep learning works.** It's true for the handwritten digits in the MNIST dataset, human faces, the sounds of the human voice, etc. The manifold hypothesis implies that:
  - Machine learning models only have to fit relatively simple, low-dimensional, highly structures subspaces within their potential input space.
  - Within one of these manifolds, it's always possible to interpolate between two inputs, that is to say, morph one into another via a continuous path along which all points fall on the manifold.
- The ability to interpolate between samples is the key to understanding generalization in deep learning.
- Interpolation can be used to fill in the blanks (this is called "local generalization"). Humans are capable of "extreme generalization". (reason vs. pattern recognition)
- A deep learning model is basically a very high-dimensional continuous curve that is fitted to data points via gradient descent, smoothly and incrementally. There will be an intermediate point during training at which the model roughly approximates the natural manifold of the data.
- You'll only be able to generalize if your data forms a manifold where points can be interpolated. The more informative and the less noisy your features are, the better you will be able to gernalize. And the more dense the sampling, the more accurate the generalization.
- You should never expect a deep learning model to perform anything more than crude interpolation between its training samples. The only thing you will find in a deep learning model is what you put into it.

## The Goal of Machine Learning: Generalization

The fundamental issue in machine learning is the tension between optimization and generalization.
  * **Optimization** refers to the process of adjutsing a model to get the best performance possible on the training data.
  * **Generalization** refers to how well the trained model performs on data it has never seen before.

The goal is to get good generalization. But you don't control generalization. You only control optimization (fit the model to its training data). If you do that *too well*, overfitting kicks in and generalization suffers. At the beginning of training, optimization and generalization are corrlated: the lower the loss on training data, the lower the loss on test data. While this is happening your model is said to be **underfit**: there is still progress to be made. The network hasn't yet modeled all relevant patterns in the training data. But after a certain number of iterations on the training data, generalization stops improving, validation metrics stall and then begin to degrade. The model is starting to **overfit**. That is, it is beginning to learn patterns that are specific to the training data but that are misleading or irrelevant when it comes to new data.

## The Workflow of Machine Learning

### 1. Define the task

- What will your input data be? What are you trying to predict?
- What type of machine learning task are you facing?
  - binary classification
  - multiclass classification
  - scalar regression
  - vector regression
  - multiclass, multilabel classification
  - image segmentation
  - ranking
  - clustering
  - generation
  - reinforcement learning
- What do existing solutions look like? How do they work?
- Are there particular constraints you will need to deal with? (e.g. latency constraints)
- What will your inputs and targets be -- you are hypothesizing that your targets can be predicted given your inputs, and that the data available is sufficiently informative to learn the relationship between the inputs and targets (until you have a working model, these are merely hypotheses)
- Should you annotate the data yourself (inexpensive, but no control)? Should you use a crowdsourcing platform like Mechanical Turk to collect labels (inexpensive, scales well, but noisy)? Should you use the services of a specialized data-labeling company? Do the data labelers need to be subject matter experts? If annotating the data requires specialized knowledge, can you train people to do it? If not, how can you get access to relevant experts? Do you yourself understand the way experts come up with the annotations (if not, you won't be able to perform feature engineeering)?
- **Is the data used for training representative of the production data?**
- Will concept drift be an issue (occurs when the properties of the production data change over time causing model accuracy to gradually decay, e.g. a music recommendation engine trained in 2013)? Dealing with concept drift requires constant data collection, annoation, and model retraining. Using machine learning trained on past data to predict the future is making the assumption that the future will behave like the past
- If your data includes images or natural language text, take a look at a few samples and their labels directly. If your data contains numerical features, plot the histogram of feature values to get a feel for the range of values taken and the frequency of different values. If your data includes location information, plot it on a map. Do any clear patterns emerge? Are some samples missing values for some features? If your task is a classification problem, print the number of instances of each class in your data. Are the classes roughly equally represented? Is every feature in your data something that will be available in the same form in production? (otherwise, you risk 'target leaking')
- What is your measure of success? (accuracy, precision and recall, customer retention rate) Your metric for success will guide all of the technical choices you make through the project. It should directly align with your higher-level goals such as the business success for your customer.
  - Balanced classification: accuracy and the area under a receiver operating characteristic curve (ROC AUC) are common metrics
  - Class imbalance, ranking, and multilabel classification: precision/recall, weighted accuracy, ROC AUC
  - To get a sense of the diversity of machine learning success metrics and how they relate to different domains it's helpful to browse the data science competitions on Kaggle

### 2. Develop the model 

#### Prepare the data

Train on as much high quality data as possible. By the manifold hypothesis, a dense sampling means interpolations will be more accurate. Spending more effort and money on data collection almost always yields a much greater return on investment than spending the same on developing a better model.

Data preprocessing aims to make the raw data more amenable to neural networks. This includes:
  - Vectorization
  - Normalization
  - Handling missing values / Anomalous or mislabeled inputs
(Many preprocessing techniques are domain-specific,  but these ones are common to all data domains.)

##### Vectorization
Whatever data you need to process, you must first trun into tensors.

##### Normalization
In general, it isn't safe to feed into a neural network data that takes relatively large values or data that is heterogeneous (e.g. on feature is in range 0-1 and another is in range 100-200). Doing so can trigger large gradient descent updates that will prevent the network from converging. To make learning easier for your network, normalize your features.

`x -= x.mean(axis=0)`
`x /= x.std(axis=0)`

##### Handling missing values

1. If the feature is categorical, it's safe to create a new category that means "this value is missing"
2. If the feature is numerical, replace the missing value with the average or median value for the feature in the dataset

Note that if you're expecting missing categorical features in the test data, but the network was trained on data without any missing values, the network won't have learned to ignore missing values. You would need to artificially generate training samples with missing entries (copy some training samples and drop some of the categorical features you expect are likely to be missing in the test data)

#### Choose an evaluation protocol

1. Holdout validation set (when you have plenty of data)
2. K-fold cross validation (when you have too few samples for holdout validation)
3. Iterated k-fold validation (when you need a highly accurate model evaluation but little data is available)

Training a deep learning model is a bit like pressing a button that launches a rocket in a parallel world. You can't hear it or see it. You can't observe the manifold learning process - it's happening in a space with thousands of dimensions, and even if you projected it to 3D, you couldn't interpret it. The only feedback you have is your validation metrics - like an altitude meter on your invisible rocket.

*Simplest options*:

1. **Common sense baseline**:
    - Before you start working with a dataset, you should always pick a trivial baseline that you'll try to beat. This baseline could be the performance of a random classifier.
    - If you have a binary classification problem where 90% of samples belong to class A and 10% belong to class B, thena classifier that always predicts A already achieves 0.8 in validation accuracy, and you'll have to do better than that.
2. **Simple holdout validation**:
    - The simplest evaluation protocol, but if little data is available, then your validation and test sets may contain too few samples to be statistically representative of the data at hand. (This is happening if different random shuffling rounds of the data before splitting end up yielding very different measures of model performance.)

*Options for splitting your data when little data is available*:

3. **K-fold validation**
    - Split your data into K partitions of equal size.
    - For each partition i, train a model on the remaining k - 1 partitions, and evaluate it on partition i.
    - Your final score is then the averages of the k scores obtained
4. **Iterated k-fold validation with shuffling** (for when you need to evaluate your model as precisely as possible)
    - Apply k-fold validation multiple times, shuffling the data every time before splitting it k ways.
    - The score is the average of the scores obtained at each run of k-fold validation.
    - Note that you end up training and evaluating p * k modls (where p is the number of iterations you use), which can be very expensive

*Notes*:
- If you're not predicting the future, randomly shuffle your data before splitting.
- If you are predicting the future, don't shuffle, and make sure data in the training set is before data in the test set. (otherwise you will create a **temporal leak**.)
- Make sure your training set and validation set are disjoint (if some data points in your data appear twice, be careful to check for redundancy before shuffling.)


#### Train a model that has generalization power

- Feature selection / Feature engineering
- Select the correct architecture (densely connected network, convet, recurrent neural network, a Transformer, something other than deep learning)
- Select a good enough training configuration (loss function, batch size, learning rate)

##### Feature selection
  - Decrease noise by compute some usefulness score for each feature available – a measure of how informative the feature is with respect to the task – and only keep features that are above some threshold.
  - Gets rid of noisy, distracting, irrelevant channels

##### Feature engineering
- The process of using your own knowledge about the data to make the algorithm work better by applying hardcoded (non-learned) transformations to the data before it goes into the model.
- The essence of feature engineering is making a problem easier by expressing it in a simpler way (making the latent manifold smooth, simpler, better organized).  Doing so usually requires understanding the problem in depth.
- Before deep learning, feature engineering used to be the most important part of the machine learning workflow. Modern deep learning removes the need for most feature engineering because neural networks are capable of automatically extracting useful features from raw data. But even in deep learning, good features are beneficial because they allow you to solve problems more elegantly while using fewer resources, and they let you solve a problem with far less data.
- Feature engineering is very important if you don't have much data.

##### Early stopping
- Finding the exact boundry between an underfit curve and an overfit curve is one of the most effective things you can do to improve generalization
- Interrupts training as soon as validation metrics have stopped improving, while remembering the best model state

#### Train a model that can overfit
- Add layers
- Make the layers bigger
- Train for more epochs

#### Regularize and tune your model

- Try different architectures (add or remove layers)
- Add dropout
- If you model is small, add L1 or L2 regularization
- Try different hyperparameters (units per layer, learning rate of the optimizer) to find the optimal configuration
- Collect or annotate more data, develop better features, or remove features that don't seem to be informative

It's possible to automate a large chunk of this work by using automated hyperparameter tuning software such as KerasTuner. Remember not to tune too much to avoid information leaks.

##### Regularization
- Actively impede the model's ability to fit perfectly to the training data
- Decreasing the size of the model
- If a network can only afford to memorize a small number of patterns, the optimization process will force it to focus on the most prominent patterns, which have a better chance of generalizing well.
- At the same time, keep in mind that you should use models that have enough parameters that they don't underfit.
- (Note on finding the right size for your model: Start with relatively few layers and parameters and increase the size of the layers or add new layers until you see diminishing returns with regard to validation loss. You'll know your model is too large if it starts overfitting right away and if its validation loss curve looks choppy with high variance. The more capacity the model has, the more quickly it can model the training data, but the more susceptible it is to overfitting)
- **Weight Regularization** is a common way to mitigate overfitting by putting constraints on the complexity of a model by forcing its weights to take only small values (done by adding to the loss function of the model a cost associated with having large weights)
    - L1 regularization: The cost added is proportional to the absolute value of the weight coefficients (called the L1 norm of the weights)
    - L2 regularization ('weight decay'): The cost added is proportional to the square of the value of the weight coefficients (called the L2 norm of the weights)
    - In Keras you can use tensorflow.keras.regularizers.l1 or l1_l2 etc.
    - Weight regularization is more typically used for smaller deep learning models. Large deep learning models tend to be so overparameterized that imposing constraints on weight values hasn't much impact on model capacity. 
- **Dropout** is one of the most effective and most commonly used regularization techniques for neural networks.
  - Dropout, applied to a layer, consists of randomly dropping out (setting to zero) a number of output features of the layer during training.
  - The dropout rate is the fraction of the features that are zeroed out -- it's usually set between 0.2 and 0.5.
  - At test time, no units are dropped out, instead, the layer's output values are scaled down by a factor equal to the dropout rate. This process can be implemented by doing both operations at training time and leaving the output unchanged at test time, which is often the way it's implemented in practice.
  - In Keras, you can introduce dropout in a model via the Dropout layer, which is applied to the output of the layer right before it
  - Dropout is typically used for large deep learning models instead of weight regularization (which wouldn't have much impact on model capacity)


##### Hyperparameter tuning
- Every time you tune a hyperparameter (number of layers, size of layers) of your model based on your model's performance on the validation set, some information about the validation set leaks into the model.
- Tuning the configuration of the model based on its performance on the validation set can quickly result in overfitting to the validation set, even though your model is never directly trained on it.

###### What To Do If Training Won't Start
If your loss stops decreasing (or never started to), this is *always* something you can overcome. Remember that you can fit a model to random data (the model just memorizes the mappings).

1. Lower or increase the learning rate
  - A learning rate that is too high may lead to updates that vastly overshoot a proper fit
  - A learning rate that is too low may make training so slow that it appears to stall
2. Increase the batch size
  - A batch with more samples will lead to gradients that are more informative and less noisy

###### What To Do If Your Model Doesn't Generalize
If you have a model that fits, but your validation metrics aren't improving, this is perhaps the worst machine learning situation you can find yourself in. It indicates that *something is fundamentally wrong with your approach*

1. The input data may not contain sufficient information to predict your targets.
2. The kind of model you're using is not suited for the problem at hand (always make sure to read up on architecture best practices for the kind of task you're attacking)

###### What To Do If Your Model Won't Start To Overfit
If your validation loss goes down and just stays there, this is *always* something you can overcome. It should always be possible to overfit. It's likely a problem with the representational power of your model. You're going to need a bigger model, one with more capacity (able to store more information).

1. Add more layers
2. Use bigger layers
3. Use a kind of layer more appropriate for the problem at hand


Once you've developed a satisfactory model configuration, you can train your fianl production model on all the available data (training and validation) and evaluate it one last time on the test set. If the performance on the test set is significantly worse than the performance measured on the validation data, this may mean:
- Your validation procedure wasn't reliable
- You began overfitting to the validation data while tuning the parameters of the model

### 3. Deploy the model

Success is about consistently meeting or exceeding people's expectations. You need to set appropriate expectations before launch. The expectations of non-specialists towards AI systems are often unrealistic.

- Show some examples of the failure modes of your model (what incorrectly classified samples look like, especially those for which the misclassification seems surprising)
- Clearly relate the models' performance metrics to business goals. Don't say "the model has 98% accuracy". Say "With these settings, the fraud detection model would have a 5% false negative rate and a 2.5% false positive rate. Every day an average of 200 valid transactions would be flagged as fraudulent and sent for manual review, and an average of 14 fraudulent transactions would be missed. An average of 266 fraudulent transactions would be correctly caught.


#### How to deploy your model via an API (self-hosting)
A common way to turn a model into a product is to install TensorFlow on a server or cloud instance and query the models' predictions via a REST API. You could build your own serving app using something like Flask, or use TensorFlows own library for shipping models as APIs called TensorFlow Serving (can deploy a keras model in minutes). This deployment setup won't work if the application has strict latency requirement - the request, inference, and answer round trip will typically take around 500 ms.

#### How to deploy your model via an API (third-party hosting)
If you don't want to host the code yourself, Cloud AI Platform, a Google product, lets you simply upload your TensorFlow model to Google Cloud Storage and it gives you an API endpoint to query it. It takes care of batch predictions, load balancing, and scaling. 

#### How to deploy a model on a website
To deploy a keras model on a website (e.g. to offload compute to the end user which can dramatically reduce server costs), use TensorFlow.js.

#### How to deploy a model on a personal device
To deploy a keras model on a personal device (e.g. because the data is sensitive and can't be decrypted on a remote server), your go-to solution is TensorFlow Lite - it includes a converter that can straighforwardly turn your keras model into the TensorFlow Lite format. 

#### How to optimize your model
You can use the TensorFlow Model Optimization Toolkit to help optimize the model, or try weight pruning/quantization using the TensorFlow toolkit.

### 4. Monitor your model in the wild
- Use randomized A/B testing to isolate the impact of the model itself from other changes: a subset of cases should go through the new model, while another control subset should stick tot he old process. Once sufficiently many cases have been processed, the difference in outcomes between the two is likely attributable to the model.
- Do a regular manual audit of the model's predictions on production data: send some fraction of the production data to be manually annotated and compare the model's predictions to the new annotations. When manual audits aren't possible, try user surveys.
- As you get ready to train the next generation of the model:
  - Watch out for changes in the production data (new features? expand label set?)
  - Keep collecting an dannotating data (pay special attention to collecting samples that seem to be difficult for you current model to classify - such samples are the most likely to help improve performance.

# Reference

## How To Improve Generalization
- Decrease noise / Feature selection
- Train on more data / Clean your data (missing values, anomalous or mislabelled inputs)
- Regularization
    - Weight Regularization
        - L1 regularization
        - L2 regularization ('weight decay')
    - Dropout
- Feature engineering
- Using early stopping

## Last-layer activation and loss functions for different tasks

| Problem type      | Last-layer activation | Loss function | 
| ----------- | ----------- | ----------- |
| Binary classification      | sigmoid       | binary_crossentropy        |
| Multiclass, single-label classification    | softmax       | categorical cross-entropy        |
| Multiclass, multi-label classification    | sigmoid       | binary_crossentropy        |



