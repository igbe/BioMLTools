import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
import random
from sklearn.metrics import accuracy_score,roc_curve, auc
from dDCA import *
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mp
from multiprocessing import Process
import multiprocessing
import sys
import Queue
import util
import csv

def check_conformity_sendto_q(queue,*args):
    """
    This function does as the name implies. It gets a sample weight, and adds them to a queue. Note by conformity,
    I mean check to see if there is any zero in the danger or safe signal if any exists, it discards this sample. Else,
    it appends it to a list.

    :param queue: The multiprocessing.queue to be use as storage for the function
    :param args:  The weight to be checked for conformity
    :return: nothing. returning something will cause it to be storing all returned values, which adds to the space
    complexity and more computation requirement
    """
    #global q
    param1 = list(args[0:3])
    param2 = list(args[3:])

    if 0 not in param1[1:]:
        if 0 not in param2[1:]:
            #print (param1,param2)
            queue.put((param1,param2))


def get_test_weights(max_min_range,queue):
    """
    This function is the function that generates the sample test weights
    :param max_min_range: The range for the generation. See the main test function below for explanation
    :param queue: the queue which will serve as the storage for the generated test weights
    :return: nothing
    """
    a, b, c, d, e, f = range(-max_min_range, max_min_range, 1), \
                       range(-max_min_range, max_min_range, 1), \
                       range(-max_min_range, max_min_range, 1), \
                       range(-max_min_range, max_min_range, 1), \
                       range(-max_min_range, max_min_range, 1), \
                       range(-max_min_range, max_min_range, 1)
    # print "a,b,c", a,b,c,d,e,f

    #the np.vectorize function is used to perform the combinatorial selection of the different possible values of the
    #weights.
    # in "queue,*np.ix_(a, b, c, d, e, f)", the queue is the same queue explained before. *np.ix_ is the function
    # doing more of the work

    np.vectorize(check_conformity_sendto_q,otypes=[np.ndarray])(queue,*np.ix_(a, b, c, d, e, f))


def evaluate_test_weights():
    while not q.empty():
        weight = q.get()
        dc = deterministicDCA(anomaly_threshold, num_of_dc_selected = num_of_dc_selected,
                         lifespan = 10, threshold = (5,15), signal_weights = weight)

        pred = dc.fit(tot_signal, issignalOrderedCorrectly=True).predict(tot_signal)
        [TN, FP], [FN, TP] = util.confusion_matrix1(label, pred)
        try:
            fpr = float(FP) / (FP + TN)
        except:
            fpr = 0.0
        try:
            tpr = float(TP) / (TP + FN)
        except:
            tpr = 0.0
        # print fpr,tpr
        f = open("gridsearch_results.csv", 'a')
        try:
            writer = csv.writer(f)
            #writer.writerow(('Weights', 'fpr', 'tpr'))
            writer.writerow((weight, fpr, tpr))
        finally:
            f.close()
    print "DONE"
        #return weight, fpr, tpr
    #print q.qsize()

    #print result


if __name__ == '__main__':

    #This the two variables that hold the data array for the DCA and label hold the class label in -1 for normal and
    # +1 for attack or abnormal
    global tot_signal
    global label

    #Below is the multiplier which is there to solve the issue of antigen deficiency
    multiplier = 10

    #Note this is using the get_data implemented in dDCA class
    data = get_data('breast_cancer_test.csv', datatype=np.float64, delimiter=',', skiprows=1, ndmin=2)
    # print 'data length, ',len(data)

    #use the get_multiplied_antigen function implemented in the dDCA. It multiplies the data returned by get_data by an
    #amount specified by multiplier
    new_data = get_multiplied_antigen(data, multiplier=multiplier)
    # print 'new_data length, ',len(new_data)
    # print new_data

    #To set the anomaly threshold
    #anomaly_threshold = ((460.0 * multiplier) / (700 * multiplier))
    anomaly_threshold = ((460 * multiplier) / (700* multiplier))

    num_of_dc_selected = 100
    lifespan = 10

    #To get the antigen for this experiment we run
    antigen = new_data[:, 0]

    #To get the constituent signals
    pamp = new_data[:, 7]
    danger = new_data[:, 8]
    safe = new_data[:, 9]

    #To get the class used therein and to convert it to the -1 or +1 type
    data_class = new_data[:, 6]
    label = convert_class(data_class)

    # To concatenate the antigen, pamp, danger and safe signals --> note that it ought to be in this order
    pamp1 = pamp[np.newaxis]
    danger1 = danger[np.newaxis]
    safe1 = safe[np.newaxis]
    # below, I called the antigen variable antigen1 to stop the original antigen array shape from being altered cause
    # predict(antigen) needs it)
    antigen1 = antigen[np.newaxis]

    #To concatenate all the signals in a specific order
    tot_signal = np.concatenate((antigen1.T, pamp1.T,danger1.T, safe1.T), axis=1)

    #To communicate between the constituent processes, we use python's multiprocessing manager to create a shared queue
    m = multiprocessing.Manager()
    q = m.Queue()

    #to specify the range of the random numbers used to generate the signal weight search space
    # an int value of say 3 here means that the weights all possible values in the following array and its elements is
    # searched ([-3:3,-3:3,-3:3],[-3:3,-3:3,-3:3]) i.e., all possible combinations
    minmax_range = 3
    # to start the two processes

    # this process generates the search weight options and puts it in a queue
    p1 = Process(target=get_test_weights,args=(minmax_range,q))
    p1.start()

    #this processes picks a sample weight, and runs the dDCA algorithm with it to evaluate its performance
    p2 = Process(target=evaluate_test_weights,) #args=('bob',))
    p2.start()
    #p1.join()
    #print q
    p2.join()
    #print "anomaly thresh", anomaly_threshold