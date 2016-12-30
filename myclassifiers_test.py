from sklearn.metrics import accuracy_score,roc_curve, auc
from sklearn.metrics import confusion_matrix
import random
import numpy as np


def testClassifier(X,threshold=3):

    """

    :param X: this is an array of the data to be predicted. note if it has only 1 element, then use [element]
        instead of just passing element.
    :return: a list containing the predicted class labels for the provided data.
    """
    #print X, type(X)
    #array to hold the predicted label
    predicted_class=[]
    for i in X:
        if i <=threshold:
            predicted_class.append(1)
        else:
            predicted_class.append(-1)

    return predicted_class

def score(X, y,threshold, sample_weight=None):
    #print X
    pred_score_ = testClassifier(X,threshold)
    #print pred_score_
    #print accuracy_score(y_true=y,y_pred=pred_score_)
    return  accuracy_score(y_true=y,y_pred=pred_score_)


















