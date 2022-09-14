"""
You may need to import necessary modules like numpy and pandas. However, you can't use any external
libraries such as sci-kit learn, etc. to implement logistic regression and the training of the logistic function.
The implementation must be done completely by yourself.

We are allowing you to use two packages from nltk for text processing: nltk.stem and nltk.tokenize. You cannot import
nltk in general, but we are allowing the use of these two packages only. We will check the code in your programs to
make sure this is the case and if other packages in nltk are used then we will deduct points from your assignment.
"""
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

import numpy as np
from string import punctuation
from nltk.tokenize import word_tokenize

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

    def feature_extraction(self, text_arr, method=None):
        """
        Optional helper method to code the feature extraction function to transform the raw dataset into a processed
        dataset to be used in training. You need to implement unigram, bigram and trigram.
        """
        n = 1

        if method == 'bigram':
            n = 2
        
        if method == 'trigram':
            n = 3

        if (self.feature_dic == None):
            voc = np.array([])
            ngram_arr = []
            for text in text_arr:
                text = text.translate(str.maketrans('', '', punctuation))
                tokenizer = word_tokenize(text)
                temp = zip(*[tokenizer[i:] for i in range(0, n)])
                ans = [' '.join(ngram) for ngram in temp]
                voc = np.concatenate((voc, ans))
                ngram_arr.append(ans)
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
                temp = zip(*[tokenizer[i:] for i in range(0, n)])
                ans = [' '.join(ngram) for ngram in temp]
                ngram_arr.append(ans)
            X = np.zeros((len(ngram_arr), len(self.feature_dic)), dtype=int)
            for i in range(len(ngram_arr)):
                for ngram in ngram_arr[i]:
                    if (ngram in self.feature_dic):
                        X[i][self.feature_dic[ngram]] = 1
        return X
    def L1(self, n):
        if n == 0:
            return 0
        if n > 0:
            return 1
        return -1
    def regularizer(self, method=None, lam=None):
        """
        You need to implement at least L1 and L2 regularizer
        """
        if method == 'L1':
            return np.vectorize(self.L1)(self.weights), self.L1(self.bias)
        if method == 'L2':
            return self.weights, self.bias


    def predict_labels(self, data_point):
        """
        Optional helper method to produce predictions for a single data point
        """
        if np.dot(self.weights, data_point) + self.bias >= 0:
            return 1
        return 0

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def train(self, labeled_data, learning_rate=0.01, max_epochs=25, lam=0.09, feature_method='unigram', reg_method='L2'):
        """
        You must implement this function and it must take in as input data in the form of a pandas dataframe. 
        This dataframe must have the label of the data points stored in a column called 'Label'. For example, 
        the column labeled_data['Label'] must return the labels of every data point in the dataset. 
        Additionally, this function should not return anything.
        
        'learning_rate' and 'max_epochs' are the same as in HW2. 'reg_method' represents the regularier, 
        which can be 'L1' or 'L2' as in the regularizer function. 'lam' is the coefficient of the regularizer term. 
        'feature_method' can be 'unigram', 'bigram' or 'trigram' as in 'feature_extraction' method. Once you find the optimal 
        values combination, update the default values for all these parameters.

        There is no limitation on how you implement the training process.
        """

        X = self.feature_extraction(np.array(labeled_data['Text']), feature_method)
        Y = np.array(labeled_data['Label'])
        n = len(X)
        r = np.array([i for i in range(n)])
        self.weights = np.zeros(len(X[0]))
        self.bias = 0.5
        for cycle in range(max_epochs):
            np.random.shuffle(r)
            for j in range(n):
                pred = np.dot(self.weights, X[r[j]]) + self.bias
                self.weights = self.weights - learning_rate * ((self.sigmoid(pred) - Y[r[j]]) * X[r[j]] - lam * self.regularizer(reg_method)[0])
                self.bias = self.bias - learning_rate * (self.sigmoid(pred) - Y[r[j]] - lam *  self.regularizer(reg_method)[1])

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

        return true / n

    def predict(self, data):
        predicted_labels = []
        """
        This function is designed to produce labels on some data input. The only input is the data in the 
        form of a pandas dataframe. 

        Finally, you must return the variable predicted_labels which should contain a list of all the predicted 
        labels on the input dataset. This list should only contain integers  that are either 0 (negative) or 
        1 (positive) for each data point.

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

if __name__ == '__main__':
    x = np.random.randint(2)
    print(x)