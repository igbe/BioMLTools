import random
import math
from numpy.random import choice
from sklearn.metrics import accuracy_score,roc_curve, auc
import numpy as np
from sklearn.metrics import confusion_matrix


def sign(x):
    """
    :param x: float or int who's value we want to know if it is + or -
    :return: 1 for positive x or -1 for negative x
    """
    return 1 if x >= 0 else -1


def normalize(weights):
        """
        :param weights: the weights to be normalized to 1
        :return: a tupple containing all the new normalized weights
        """
        norm = sum(weights)
        return tuple(m / norm for m in weights)

def weighted_sample(X,initial_weights):
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
