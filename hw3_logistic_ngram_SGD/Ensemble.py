"""
You only need to implement bagging.
"""
import matplotlib.pyplot as plt
import numpy as np
from string import punctuation
from nltk.tokenize import word_tokenize

class Ensemble():
    def __init__(self):
        """
        You may initialize the parameters that you want and remove the 'return'
        """
        self.weights = None
        self.biases = None
        self.feature_dic = None

    def feature_extraction(self, text_arr):
        """
        Use the same method as in Logistic.py
        """
        if (self.feature_dic == None):
            voc = np.array([])
            ngram_arr = []
            for text in text_arr:
                text = text.translate(str.maketrans('', '', punctuation))
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
                text = text.translate(str.maketrans('', '', punctuation))
                tokenizer = word_tokenize(text)
                ngram_arr.append(tokenizer)
            X = np.zeros((len(ngram_arr), len(self.feature_dic)), dtype=int)
            for i in range(len(ngram_arr)):
                for ngram in ngram_arr[i]:
                    if (ngram in self.feature_dic):
                        X[i][self.feature_dic[ngram]] = 1
        return X
    
    def predict_labels(self, data_point):
        """
        Optional helper method to produce predictions for a single data point
        """
        return

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def train(self, labeled_data, num_clf=100):
        """
        You must implement this function and it must take in as input data in the form of a pandas dataframe. 
        This dataframe must have the label of the data points stored in a column called 'Label'. For example, 
        the column labeled_data['Label'] must return the labels of every data point in the dataset. 
        Additionally, this function should not return anything.

        There is no limitation on how you implement the training process.
        """
        learning_rate = 0.01
        lam = 0.09
        X = self.feature_extraction(np.array(labeled_data['Text']))
        Y = np.array(labeled_data['Label'])
        n = len(X)
        self.weights = []
        self.biases = []
        for j in range(num_clf):
            weights = np.zeros(len(X[0]))
            bias = 0.5
            for cycle in range(5):
                for j in range(n):
                    r = np.random.randint(n)
                    pred = np.dot(weights, X[r]) + bias
                    weights = weights - learning_rate * ((self.sigmoid(pred) - Y[r]) * X[r] - lam * weights)
                    bias = bias - learning_rate * (self.sigmoid(pred) - Y[r] - lam * bias)
            self.weights.append(weights)
            self.biases.append(bias)

        return


    def predict(self, data):
        """
        This function is designed to produce labels on some data input. The only input is the data in the 
        form of a pandas dataframe. 

        Finally, you must return the variable predicted_labels which should contain a list of all the predicted 
        labels on the input dataset. This list should only contain integers  that are either 0 (negative) or 
        1 (positive) for each data point.

        The rest of the implementation can be fully customized.
        """

        X = self.feature_extraction(np.array(data['Text']))
        clf = len(self.weights)
        n = len(X)
        predicted_labels = np.zeros(len(data))
        for j in range(clf):
            label = []
            for i in range(n):
                if self.sigmoid(np.dot(X[i], self.weights[j]) + self.biases[j]) > 0.5:
                    label.append(1)
                else:
                    label.append(0)
            predicted_labels += np.array(label)
        predicted_labels = list((predicted_labels >= clf / 2).astype(int))
        return predicted_labels

    

