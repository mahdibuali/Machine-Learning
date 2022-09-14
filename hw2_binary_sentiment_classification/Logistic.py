"""
You may need to import necessary modules like numpy and pandas. However, you can't use any external
libraries such as sci-kit learn, etc. to implement logistic regression and the training of the logistic function.
The implementation must be done completely by yourself.

We are allowing you to use two packages from nltk for text processing: nltk.stem and nltk.tokenize. You cannot import
nltk in general, but we are allowing the use of these two packages only. We will check the code in your programs to
make sure this is the case and if other packages in nltk are used then we will deduct points from your assignment.
"""
import nltk
import numpy as np
import string
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt


"""
This is a Python class meant to represent the logistic model and any sort of feature processing that you may do. You 
have a lot of flexibility on how you want to implement the training of the logistic function but below I have listed 
functionality that should not change:
    - Arguments to the __init__ function 
    - Arguments and return statement of the train function
    - Arguments and return statement of the predict function 


When you want the program (logistic) to train on a dataset, the train function will only take one input which is the 
raw copy of the data file as a pandas dataframe. Below, is example code of how this is done:

    data = pd.read_csv('data.csv', index_col=0)
    model = Logistic()
    model.train(data) # Train the model on data.csv


It is assumed when this program is evaluated, the predict function takes one input which is the raw copy of the
data file as a pandas dataframe and produces as output the list of predicted labels. Below is example code of how this 
is done:

    data = pd.read_csv('data.csv', index_col=0)
    model = Logistic()
    predicted_labels = model.predict(data) # Produce predictions using model on data.csv

I have added several optional helper methods for you to use in building the pipeline of training the logistic function. 
It is up to your discretion on if you want to use them or add your own methods.
"""


class Logistic():
    def __init__(self):
        """
        The __init__ function initializes the instance attributes for the class. There should be no inputs to this
        function at all. However, you can setup whatever instance attributes you would like to initialize for this
        class. Below, I have just placed as an example the weights and bias of the logistic function as instance
        attributes.
        """
        self.weights = None
        self.bias = None
        self.feature_dic = None

    def feature_extraction(self, text_arr):
        """
        Optional helper method to code the feature extraction function to transform the raw dataset into a processed
        dataset to be used in training.
        """
        if (self.feature_dic == None):
            voc = np.array([])
            ngram_arr = []
            for text in text_arr:
                text = text.translate(str.maketrans('', '', string.punctuation))
                tokenizer = word_tokenize(text)
                voc = np.concatenate((voc, tokenizer))
                ngram_arr.append(tokenizer)
            voc = np.unique(voc)
            self.feature_dic = {}
            for i in range(len(voc)):
                self.feature_dic[voc[i]] = i

            X = np.zeros((len(ngram_arr), len(voc)), dtype=int)
            for i in range(len(ngram_arr)):
                for ngram in ngram_arr[i]:
                    X[i][self.feature_dic[ngram]] = 1
        else:
            ngram_arr = []
            for text in text_arr:
                text = text.translate(str.maketrans('', '', string.punctuation))
                tokenizer = word_tokenize(text)
                ngram_arr.append(tokenizer)
            X = np.zeros((len(ngram_arr), len(self.feature_dic)), dtype=int)
            for i in range(len(ngram_arr)):
                for ngram in ngram_arr[i]:
                    if (ngram in self.feature_dic):
                        X[i][self.feature_dic[ngram]] = 1

        return X

    def logistic_loss(self, predicted_label, true_label):
        """
        Optional helper method to code the loss function.
        """
        return

    def stochastic_gradient_descent(self, data, labels):
        """
        Optional helper method to compute a gradient update for a single point.
        """
        return

    def update_weights(self, new_weights):
        """
        Optional helper method to update the weights during stochastic gradient descent.
        """
        self.weights = new_weights

    def update_bias(self, new_bias):
        """
        Optional helper method to update the bias during stochastic gradient descent.
        """
        self.bias = new_bias

    def predict_labels(self, data_point):
        """
        Optional helper method to produce predictions for a single data point
        """
        if np.dot(self.weights, data_point) + self.bias >= 0:
            return 1
        return 0
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))


    def train(self, labeled_data, learning_rate=None, max_epochs=None):
        """
        You must implement this function and it must take in as input data in the form of a pandas dataframe. This
        dataframe must have the label of the data points stored in a column called 'Label'. For example, the column
        labeled_data['Label'] must return the labels of every data point in the dataset. Additionally, this function
        should not return anything.

        The hyperparameters for training will be the learning rate and maximum number of epochs. Once you find the
        optimal values, update the default values for both the learning rate and max epochs keyword argument.

        The goal of this function is to train the logistic function on the labeled data. Feel free to code this
        however you want.
        """
        X = self.feature_extraction(np.array(labeled_data['Text']))
        Y = np.array(labeled_data['Label'])
        max_epochs = 15
        learning_rate = 0.0356
        n = len(X)
        r = np.array([i for i in range(n)])
        self.weights = np.zeros(len(X[0]))
        self.bias = 0.5
        for cycle in range(max_epochs):
            np.random.shuffle(r)
            for j in range(n):
                pred = np.dot(self.weights, X[r[j]]) + self.bias
                self.weights = self.weights - learning_rate * ((self.sigmoid(pred) - Y[r[j]]) * X[r[j]])
                self.bias = self.bias - learning_rate  * (self.sigmoid(pred) - Y[r[j]])
        return

    def evaluate(self, data):
        n = len(data[0])
        true = 0
        for i in range(n):

            if self.sigmoid(np.dot(data[0][i], self.weights) + self.bias) > 0.5:
                if data[1][i] == 1:
                    true += 1
            else:
                if data[1][i] == 0:
                    true += 1

        return true/n

    def predict(self, data):
        predicted_labels = []
        """
        This function is designed to produce labels on some data input. The only input is the data in the form of a 
        pandas dataframe. 

        Finally, you must return the variable predicted_labels which should contain a list of all the 
        predicted labels on the input dataset. This list should only contain integers  that are either 0 (negative) or 1
        (positive) for each data point.

        The rest of the implementation can be fully customized.
        """
        X = self.feature_extraction(np.array(data['Text']))

        n = len(X)
        for i in range(n):
            if self.sigmoid(np.dot(X[i], self.weights) + self.bias) > 0.5:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
        return predicted_labels