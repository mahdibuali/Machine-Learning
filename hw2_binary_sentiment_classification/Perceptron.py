"""
You may need to import necessary modules like numpy and pandas. However, you can't use any external
libraries such as sci-kit learn, etc. to implement the perceptron and the training of the perceptron.
The implementation must be done completely by yourself.

We are allowing you to use two packages from nltk for text processing: nltk.stem and nltk.tokenize. You cannot import
nltk in general, but we are allowing the use of these two packages only. We will check the code in your programs to
make sure this is the case and if other packages in nltk are used then we will deduct points from your assignment.
"""
import string
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
"""
This is a Python class meant to represent the perceptron model and any sort of feature processing that you may do. You 
have a lot of flexibility on how you want to implement the training of the perceptron but below I have listed 
functionality that should not change:
    - Arguments to the __init__ function 
    - Arguments and return statement of the train function
    - Arguments and return statement of the predict function 


When you want the program (perceptron) to train on a dataset, the train function will only take one input which is the 
raw copy of the data file as a pandas dataframe. Below, is example code of how this is done:

    data = pd.read_csv('data.csv', index_col=0)
    model = Perceptron()
    model.train(data) # Train the model on data.csv


It is assumed when this program is evaluated, the predict function takes one input which is the raw copy of the
data file as a pandas dataframe and produce as output the list of predicted labels. Below is example code of how this 
is done:

    data = pd.read_csv('data.csv', index_col=0)
    model = Perceptron()
    predicted_labels = model.predict(data) # Produce predictions using model on data.csv

I have added several optional helper methods for you to use in building the pipeline of training the perceptron. It is
up to your discretion on if you want to use them or add your own methods.
"""


class Perceptron():
    def __init__(self):
        """
        The __init__ function initializes the instance attributes for the class. There should be no inputs to this
        function at all. However, you can setup whatever instance attributes you would like to initialize for this
        class. Below, I have just placed as an example the weights and bias of the perceptron as instance attributes.
        """
        self.weights = None
        self.bias = None
        self.feature_dic = None

    def feature_extraction(self, text_arr):
        """
        Optional helper method to code the feature extraction function to transform the raw dataset into a processed
        dataset to be used in perceptron training.
        """
        if (self.feature_dic == None) :
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

    def sgn_function(self, perceptron_input):
        """
        Optional helper method to code the sign function for the perceptron.
        """
        if perceptron_input >= 0:
            return 1
        return -1

    def update_weights(self, new_weights):
        """
        Optional helper method to update the weights of the perceptron.
        """
        self.weights = new_weights

    def update_bias(self, new_bias):
        """
        Optional helper method to update the bias of the perceptron.
        """
        self.bias = new_bias

    def predict_labels(self, data_point):
        """
        Optional helper method to produce predictions for a single data point.
        """
        return self.sgn_function(np.dot(self.weights, data_point) + self.bias)

    def five_folds(self, X, Y):
        length = len(X) // 5
        fifth_X = (X[: length], X[length : 2 * length], X[2 * length : 3 * length], X[3 * length : 4 * length], X[4 * length :])
        fifth_Y = (Y[: length], Y[length : 2 * length], Y[2 * length : 3 * length], Y[3 * length : 4 * length], Y[4 * length :])
        train_sets, val_sets = [], []
        for i in range(5):
            val_sets.append((fifth_X[i], fifth_Y[i]))
            index = []
            for j in range(5):
                if j != i:
                    index.append(j)
            train_x = np.concatenate((fifth_X[index[0]], fifth_X[index[1]], fifth_X[index[2]], fifth_X[index[3]]))
            train_y = np.concatenate((fifth_Y[index[0]], fifth_Y[index[1]], fifth_Y[index[2]], fifth_Y[index[3]]))
            train_sets.append((train_x, train_y))
        return train_sets, val_sets


    def evaluate(self, data):
        n = len(data[0])
        true = 0
        for i in range(n):
            pred = self.predict_labels(data[0][i])
            if pred == data[1][i]:
                true += 1
        return true/n


    def train(self, labeled_data, learning_rate=None, max_iter=None):
        """
        You must implement this function and it must take in as input data in the form of a pandas dataframe. This
        dataframe must have the label of the data points stored in a column called 'Label'. For example, the column
        labeled_data['Label'] must return the labels of every data point in the dataset. Additionally, this function
        should not return anything.

        The hyperparameters for training will be the learning rate and max number of iterations. Once you find the
        optimal values of the hyperparameters, update the default values for each keyword argument to reflect those
        values.

        The goal of this function is to train the perceptron on the labeled data. Feel free to code this however you
        want.
        """

        X = self.feature_extraction(np.array(labeled_data['Text']))
        Y = np.array(labeled_data['Label'])
        Y = ((Y - 0.5) * 2).astype(int)

        max_iter = 15
        learning_rate = 0.0112
        n = len(X)
        self.weights = np.zeros(len(X[0]))
        self.bias = 0
        for cycle in range(max_iter):
            for j in range(n):
                pred = self.predict_labels(X[j])
                if pred != Y[j]:
                    self.weights = self.weights + learning_rate * Y[j] * X[j]
                    self.bias = self.bias + learning_rate * Y[j]








        return

    def predict(self, data):
        predicted_labels = []
        """
        This function is designed to produce labels on some data input. The first input is the data in the form of a 
        pandas dataframe. 
        
        Finally, you must return the variable predicted_labels which should contain a list of all the 
        predicted labels on the input dataset. This list should only contain integers that are either 0 (negative) or 1
        (positive) for each data point.
        
        The rest of the implementation can be fully customized.
        """
        X = self.feature_extraction(np.array(data['Text']))
        n = len(X)
        for i in range(n):
            predicted_labels.append(int(0.5 + self.predict_labels(X[i])/2))
        return predicted_labels

if __name__ == '__main__':
    d = {0.1 : 2, 0.2 : 5, 0.3: 3}
    s = 5
    plt.plot(d.keys(), d.values())
    plt.title('Learning rate and mean 5-folds validation\n accuracy for maximum iterations: ' + str(s))
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.show()