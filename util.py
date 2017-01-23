import random
import math
from numpy.random import choice
from sklearn.metrics import accuracy_score,roc_curve, auc
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from pandas.tools.plotting import parallel_coordinates
import pandas as pd
import matplotlib.pyplot as plt


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

def convert_class(y,class_label):
    """

    :param y: the nx1 label data to be converted
    :param class_label: (normal_label, anomaly_label)
    :return:
    """
    cl = []
    for i in y:
        if i == class_label[0]:
            cl.append(-1)
        else:
            cl.append(1)
    return np.array(cl)

def get_data(data,isnorminal_data_inside = False, isdata_normalized = True, norminalAttributecolumn = []):

    #
    # print le.transform(my_data[1:,0])
    if isnorminal_data_inside == True:
        for column in norminalAttributecolumn:
            #print column, type(column)
            le = LabelEncoder()
            dt = data[0:, column]
            le.fit(dt)
            data[0:, column] = le.transform(dt)

        data = data.astype(np.float)
        not_normalized_data = data
        #print data

    else:
        not_normalized_data = data
    if isdata_normalized == False:
        # Normalize time series data
        # train the normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(data)
        #print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
        # normalize the dataset and print the first 5 rows
        normalized_data = scaler.transform(data)

    else:
        normalized_data = data

    return normalized_data, not_normalized_data

def get_parallel_cordinate(data_to_bo_plotted,label,my_data,class_column_title = "class"):
    """

    :param data_to_bo_plotted: this is the numpy array of the data to be plotted
    :param label: The label array for the data_to_bo_plotted array
    :param my_data: the is the initial data extracted from the .csv file proir to any processing. This will form the
                    column for the panda frame
    :return:
    """

    parallel_coordinates_data = np.concatenate((data_to_bo_plotted, label.T), axis=1)
    #print "parallel_coordinates_data", parallel_coordinates_data

    df = pd.DataFrame(data=parallel_coordinates_data[0:, 0:],
                      index=[str(i) for i in range(1, len(data_to_bo_plotted) + 1)],
                      columns=my_data[0, 0:])  # 1st row as the column names
    # print df
    parallel_coordinates(df, class_column_title)
    plt.show()

def confusion_matrix1(y_true, y_pred):
    TN, FP, FN, TP = 0, 0, 0, 0

    for y1, y2 in zip(y_true, y_pred):

        # print y1,y2
        if y1 == -1:  # normal
            if y2 == y1:
                TN += 1
            else:
                FP += 1
        else:
            if y2 == y1:
                TP += 1
            else:
                FN += 1
    return [TN, FP], [FN, TP]