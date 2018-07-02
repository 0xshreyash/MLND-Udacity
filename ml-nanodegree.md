# Notes from Udacity's Machine Learning Nanodegree

## Section 1

What is Machine Learning? Humans learn from experience but machines need to be told what to do. Machine Learning teaches computers to perform tasks from past exepriences. Past experienes -> data.

### Decision Trees

Split data by the feature that best splits the data. Keep on subdividing the tree by the most divisive properties. Fidn the best fitting tree for a set of data. 

### Naive Bayes

Use probabilities given certain features and then classify things according to a combination of these features.

### Gradient Descent

Go in the direction that will get us to descend the most. We continue to go in the direction of maximum descent until we get to the correct solution.


### Linear Regression

Fit a line to the house in order to try to do well in predicting the house prices. Calculate the error to find how much we need to move the line to reduce the error. Use gradient descent to reduce the error the most and try to find the best answer possible to reduce the error the most until we find its minimum value until we find the best fit.

### Logistic Regression

Log loss function -> is what we need to minimise. Error function will assign a small penalty to correct points and large number to wrong points. Idea is to jiggle the line around to get the correct answer. Point of the process is to minimize the error function. We use gradient descent to minimize the error function. The goal here is to minimise the error and we use gradient descent to reduce the error by moving in the direction which reduces the error the most.

### Support Vecotr Machine

Find the best line to split the data. Take the smallest of the distances from the line and we want the minimum of all the distances must be as large as possible. We want to maximise the distance using Gradient Descent. The support vecotrs are the points closest to the line.

### Neural Network

Maybe separate data with two lines instead of just one. We have an input layer. Then input goes through a middle layer called a hidden layer. It helps find features that are important. Add more nodes in order to learn more complex set of features. The nodes resemble the neurons in the brain. 

### Kernel Trick

Use a lot in SVM. The curve and the plane trick are actually equivalent. The xy value is really important in the example given in the lectures. Change the number of planes or use a curve instead of a line to separate things.

Answer to challenge from Lesson 2 Lecture 21: Use XOR of the values to be able to use to kernal as XOR.

### K-means clustering

Find groups of points that are similar to each other. Start by choosing three random locations on the map. Each point goes into the location closest to it. Move each location (mean) as many times as you can change things around. We know how many clusters we want to end up with.

### Hierarchical Clustering

If two houses are close they should be served by the same pizza parlour. Start with a pizza parlour for each house. Then join closest houses to have the same pizza parlour. We do this until we want the two closest houses are too far apart. This is hierarchical clustering.


## Section 2

Simple implementation of a number of algorithms. This section mainly deals with model evaluation and testing.

Automatic tuning of parameters.

_Regression -_ Determines a value.
_Classification -_ Returns a state/class (the instructor enjoys calling this state).

Sklearn has an easy way to split data between training and test sets.

### Evaluation Matrix

Just make a matrix of the true positive, false negatives, false positives, true negatives.

Accuracy is not a good measure in cases where the majority class gives you a very high accuracy.

F1 score = (2 * precision * recall)/(x + y).

If one of precision and recall is bad then we kind of raise a red flag.

F1 score gives both precision and recall the same precedence. 

If our model cares about one of them a bit more than the other then 

F_beta score = ((1 + beta^2) * precision * recall)/((beta^2 * precision + recall)).

Finding a good value for beta is basically about intuition. Higher beta intutively means that the method will like recall more than precision.

ROC Curve -> Receiver Operating Characteristic Curve. Gives a perfect split a score of 1. A good split a score of 0.8 and a random split a score of 0.5.

True Positive Rate = No. of True Positive / Total positives.
FP rate = No. of false positves/Total positives

The closer the area under the ROC curve the better the model is. The area under the curve can be less than 0.5. READ ABOUT ROC AGAIN.

Mean Absolute Error or MSE are both used. R2 score -> 1 - Error for our model / Error for majority class/avg in case of regression. Larger R2 score is better since that means our model has small error as compared to the actual error from averaging/taking majority class.

#### Types of error

The error of over-simplifying a model or over-complicating a model are very easy to make. Over-simplifying -> Underfitting -> High bias. Over-complicating -> Overfitting -> High Variance. Underfitting does not do well on the training set. Overfitting does well on the training set but it is common for it to not do well on the test set.

**Detecting errors** - Use graphs to see the testing and training error. We pick the model with complexity such that the testing and training error are simialr to each other. Otherwise we are overfitting. Underfitting is less of a problem since most people will try to make their algorithm work well for the training data. Use a validation set to learn about the ideal compelxity of the model.

**K-fold cross validation** - Break the data into k buckets and then training the data on K - 1 sets. We choose buckets at random.

**Learning Curves** - The curves for training and testing move closer and closer to each other as we increase the size of the training set. If the model convergers to a low error point then the model is good. High bias models converge but to a high error point whereas high variance models do not converge at all. 

Train a model -> use cross-validation to validate our model -> test our model.



## Deep Learning

Deep Learning has many applications. The central application of deep learning is in neural networks. It tries to model a brain and how neurons in the brain function. 


