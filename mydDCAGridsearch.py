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

def tester(queue,*args):
    #global q
    param1 = list(args[0:3])
    param2 = list(args[3:])

    if 0 not in param1[1:]:
        if 0 not in param2[1:]:
            #print (param1,param2)
            queue.put((param1,param2))
        #return (param1,param2)

    #print "args",args

    #return (param1,param2)


def func1(max_min_range,queue):
    a, b, c, d, e, f = range(-max_min_range, max_min_range, 1), \
                       range(-max_min_range, max_min_range, 1), \
                       range(-max_min_range, max_min_range, 1), \
                       range(-max_min_range, max_min_range, 1), \
                       range(-max_min_range, max_min_range, 1), \
                       range(-max_min_range, max_min_range, 1)
    # print "a,b,c", a,b,c,d,e,f
    np.vectorize(tester,otypes=[np.ndarray])(queue,*np.ix_(a, b, c, d, e, f))
    # vfunc = np.vectorize(tester,otypes=[np.ndarray])
    # #print answer
    # result = vfunc(a,b,c,d,e,f)

def func2():
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
    global tot_signal
    global label

    multiplier = 1
    # dca1 = dca.DCA(dca_pop_size=0,is_it_static=True)
    data = get_data('breast_cancer_test.csv', datatype=np.float64, delimiter=',', skiprows=1, ndmin=2)
    # print 'data length, ',len(data)
    new_data = get_multiplied_antigen(data,
                                      multiplier=multiplier)  # data #dca.get_multiplied_antigen(data, multiplier=2)
    # print 'new_data length, ',len(new_data)
    # print new_data
    anomaly_threshold = ((460.0 * multiplier) / (700 * multiplier))
    num_of_dc_selected = 100
    lifespan = 10
    antigen = new_data[:, 0]
    data_class = new_data[:, 6]
    safe = new_data[:, 9]
    pamp = new_data[:, 7]
    danger = new_data[:, 8]
    label = convert_class(data_class)
    # cells = dc.initialize_dDCA(num_of_dc_selected)
    # print cells
    # To concatenate the antigen, dnager and safe signals
    pamp1 = pamp[np.newaxis]
    danger1 = danger[np.newaxis]
    safe1 = safe[np.newaxis]
    antigen1 = antigen[np.newaxis]  # I called it antigen1 which will stop the original antigen array shape from being
    #  altered cause predict(antigen) needs it)
    tot_signal = np.concatenate((antigen1.T, pamp1.T,danger1.T, safe1.T), axis=1)

    m = multiprocessing.Manager()
    q = m.Queue()

    minmax_range = 3
    p1 = Process(target=func1,args=(minmax_range,q))
    p1.start()
    p2 = Process(target=func2,) #args=('bob',))
    p2.start()
    #p1.join()
    #print q
    p2.join()