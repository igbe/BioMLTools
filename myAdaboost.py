import random
import math
from numpy.random import choice
from sklearn.metrics import accuracy_score,roc_curve, auc
import numpy as np

class AdaBoost:
    def __init__(self,weakLearners,rounds=2,weightUpdate=True):

        self.weakLearners = weakLearners
        self.rounds = rounds
        self.hypotheses = [None] * rounds
        self.alpha = [0] * rounds
        self.error1 = []
        self.weightUpdate = weightUpdate
        #pass


    def computeError(self, h, y, weights=None):
        """
        :param h: the hypothesis or class prediction made by a classifier
        :param y: the true class labels from the original training data
        :param weights: the weight or probability distribution
        :return: the hypothesisResult which is +1 if classifier is correct or -1 otherwise, then the error
        """
        hypothesisResults = []
        if weights is None:
            weights = [1.] * len(y)
        for i in range(len(y)):
            hypothesisResults.append(h[i]*y[i])     # +1 if correct, else -1
        return hypothesisResults, sum(w for (z,w) in zip(hypothesisResults, weights) if z < 0)

    # normalize a distribution


    def normalize(self, weights):
        """
        :param weights: the weights to be normalized to 1
        :return: a tupple containing all the new normalized weights
        """
        norm = sum(weights)
        return tuple(m / norm for m in weights)



    def sign(self, x):
        """
        :param x: float or int who's value we want to know if it is + or -
        :return: 1 for positive x or -1 for negative x
        """
        return 1 if x >= 0 else -1


    def weighted_sample(self, X,initial_weights):
        """
        This function returns sample data-instances from the training data based on a weighted distribution.
        The higher your weight, the more likely you will get selected.
        :param initial_weights:  the weights or probability distribution
        :return: the newly selected data samples
        """
        indices = range(len(X))     #get the indices
        #print indices
        my_draw = choice(indices, len(X), p=initial_weights)  #using numpy choice method that supports weights
        #print my_draw
        return X[my_draw]

    def fit(self, X,y):
        """
        This is the main function of the adaboost algorithm. It calls other functions in this file.
        :param X: This is the training data to be use for training
        :param y: This is the class label for the training data
        :return: the final prediction
        """
        weight = self.normalize([1.] * len(X))     #the probability distribution or the weights assigned to each training set
        #print "weights",weight
        print "len of weakLearner", len(self.weakLearners)

        tru_y = y
        #self.hypotheses = [None] * self.rounds
        #self.alpha = [0] * self.rounds
        #self.error1 = []

        #Iterate through the number of self.rounds
        for t in range(self.rounds):
            #print "weight", weight


            if t == 0:      #since for the first iteration, the sample is same with the training sample
                sample = X
            else:
                # use the new weights in other iterations to choose the next training sample
                # this tries to help your model understand new things about your data expecially the samples that
                # wasn't well classified in previous iteration. Obviously setting this to X as in t==0, makes this
                # algorithm behave like majority voting. But this last statement applies to deterministic algorithms.

                if self.weightUpdate == True:
                    sample = self.weighted_sample(X,weight)
                else:
                    #print "in false"
                    sample = X
            #print "sample", sample

            # Choose the number t weakLeaner from the list of learners
            weakLearner = self.weakLearners[t]


            # call the fit method of the weakLeaner to train the data (sample), then use its predict method to get the
            # predicted class labels for the original training data. Note
            self.hypotheses[t] = weakLearner.fit(sample, y).predict(X)
            #print "hypotheses[t]", self.hypotheses[t]

            #compute the error in the labels retruned by the above weakLeaner
            hypothesisResults, error = self.computeError(self.hypotheses[t], y, weight)
            #print "hypothesisResults, error", hypothesisResults, error
            #print "Error", error


            # To make sure you have a list of all the errors made by your classifier
            self.error1.append(error)

            # To append new values for alpha for each round/hypothese to the alpha list
            self.alpha[t] = 0.5 * math.log((1 - error) / (.0001 + error))
            #print "self.alpha", self.alpha[t]

            #Get the new normalized weights
            weight = self.normalize([d * math.exp(-self.alpha[t] * h)
                              for (d,h) in zip(weight, hypothesisResults)])
            print("Round %d, error %.3f, Accuracy %.3f, Alpha %.3f" % (t, error,accuracy_score(y_true=y, y_pred=self.hypotheses[t]),self.alpha[t]))

        return self

    #def predict(self, X, y, tru_y):

    def predict(self, X):
        i = 0
        hypo = [None] * len(self.weakLearners)

        for weakLearner in self.weakLearners:
            hypo[i] = weakLearner.predict(X)
            i+=1

        #print "hypo", hypo
        pred = []
        for j in range(len(X)):
            value = []
            for l in range(len(self.alpha)):
                value.append(self.alpha[l]*hypo[l][j])
            #print sum(value)
            pred.append(self.sign(sum(value)))

        return np.array(pred)

            #value = self.sign(sum([self.alpha[0]*hypo[0][j], self.alpha[1]*hypo[1][j]]))
            #print value


        #fHypothesis = []

        #for i in range(len(X)):
        #    sum1 = []
        #    for (a,h) in zip(self.alpha,self.hypotheses):
        #        #print a*h[i]
        #        sum1.append(a*h[i])
        #        #print sign(sum(sum1))
        #   fHypothesis.append(sign(sum(sum1)))

      # print fHypothesis
       #return sign(sum(a * h(x) for (a, h) in zip(self.alpha, hypotheses)))






    #     def error(fHypothesis, tru_y):
    #         #print "tru_y",tru_y
    #         #print "hypothesis", fHypothesis
    #         sum2=[]
    #         for i in range(len(tru_y)):
    #             #print fHypothesis[i],tru_y[i]
    #             if fHypothesis[i] != tru_y[i]:
    #                 sum2.append(1)
    #         #print sum2
    #         return sum(sum2) / float(len(tru_y))
    #         #for i in range(len(y)):
    #     return fHypothesis, error(fHypothesis,tru_y)
    #        #return sum(1 for x, y in data if h(x) != y) / len(data)
    # final,error = finalHypothesis(y,tru_y)

        #return final,error
