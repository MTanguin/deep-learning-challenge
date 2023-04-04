# deep-learning-challenge

## Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* The purpose of the analysis is to build a tool that can help nonprofit foundation, Alphabet Soup select the applicants for funding with the best   chance of success in their ventures. Specifically, to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup by using the features in the provided dataset, a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. These features are :

--EIN and NAME—Identification columns
--APPLICATION_TYPE—Alphabet Soup application type
--AFFILIATION—Affiliated sector of industry
--CLASSIFICATION—Government organization classification
--USE_CASE—Use case for funding
--ORGANIZATION—Organization type
--STATUS—Active status
--INCOME_AMT—Income classification
--SPECIAL_CONSIDERATIONS—Special considerations for application
--ASK_AMT—Funding amount requested
--IS_SUCCESSFUL—Was the money used effectively



## Results

Data Preprocessing

Several data preprocessing tasks, including dropping non-beneficial columns ('EIN' and 'NAME'), binning and replacing values in the 'APPLICATION_TYPE' and 'CLASSIFICATION' columns, and converting categorical data to numeric using one-hot encoding were performed.

The target variable for the model is IS_SUCCESSFUL, which indicates whether or not the money funded by Alphabet Soup was used effectively. The model will predict whether an applicant will be successful if funded by Alphabet Soup based on various features in the dataset.
The features for the model could include:

--APPLICATION_TYPE—Alphabet Soup application type
--AFFILIATION—Affiliated sector of industry
--CLASSIFICATION—Government organization classification
--USE_CASE—Use case for funding
--ORGANIZATION—Organization type
--STATUS—Active status
--INCOME_AMT—Income classification
--SPECIAL_CONSIDERATIONS—Special considerations for application
--ASK_AMT—Funding amount requested

EIN', 'NAME' , IS_SUCCESSFUL are variables that were removed


The neural network model includes eight hidden layers with various numbers of nodes and uses the 'relu' activation function for each layer except the output layer, which uses the 'sigmoid' activation function. The model is compiled using the 'adam' optimizer and the binary cross-entropy loss function.

The model is trained on the preprocessed data for 100 epochs using the training dataset, and the performance of the model is evaluated using the test dataset. The accuracy of the model is reported as a metric.

To optimize the model and achieve a target predictive accuracy higher than 75%, the following techniques is suggested:

Increase the number of epochs: The model may need more epochs to learn better patterns in the data. You can try increasing the number of epochs in the model.fit() function.

Change the optimizer: You can try using a different optimizer such as Adam or RMSprop instead of the stochastic gradient descent (SGD) optimizer used in the current code. These optimizers are known to work better for deep neural networks.

Adjust learning rate: You can try decreasing the learning rate to allow the model to converge slowly and improve the predictive accuracy.

Add more layers: You can add more layers to the model to make it more complex and better able to capture patterns in the data.

Increase the size of the layers: You can try increasing the number of neurons in the layers of the model to increase its capacity to learn complex patterns.




## Summary

In an attempt to build a model than can achieve target predictive accuracy higher than 75%, several experiment were done to build and train a deep neural network using TensorFlow and Keras to predict whether nonprofit organizations that apply for funding from Alphabet Soup will be successful. The experiment used data preprocessing techniques such as dropping columns, binning values, and one-hot encoding to clean the data and convert categorical variables to numerical values. Then splits the data into training and testing datasets, scales the features using StandardScaler, and defines a deep neural network with several hidden layers and an output layer with a sigmoid activation function. Finally, compiles, trains, and evaluates the model using binary cross-entropy loss and accuracy as the evaluation metric.


