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

#test_data = []
#test_class = []

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


tot_param1 =[]
tot_param2 =[]
def tester(*args):
    global tot_param1
    global  tot_param2
    param1 = list(args[0:3])
    param2 = list(args[3:])

    if 0 in param1[1:]:
        return 0
    if 0 in param2[1:]:
        return 0
    tot_param1.append(param1)
    tot_param2.append(param2)
    #print tot_param1
    #print tot_param2


def get_fpr_tpr(classifer):
    #print classifer
    #return 0
    #print test_data
    #print classifer
    pred = classifer.fit(tot_signal, issignalOrderedCorrectly=True).predict(tot_signal)
    #print pred
    #print "fpr,tpr",dc.roc(label)
    #fpr, tpr = dc.roc(label)
    [TN, FP], [FN, TP] = confusion_matrix1(label, pred)
    try:
        fpr = float(FP) / (FP + TN)
    except:
        fpr = 0.0
    try:
        tpr = float(TP) / (TP + FN)
    except:
        tpr = 0.0
    #print fpr,tpr
    return classifer.signal_weights, fpr,tpr



functions = {"dca":deterministicDCA}
def search_grid(zip_grid,anomaly_threshold,num_of_dc_selected,lifespan):
    global functions
    jobs=[]
    for a,b in zip_grid:
        #print (a,b)
        jobs.append(functions["dca"](anomaly_threshold, num_of_dc_selected=num_of_dc_selected, lifespan=10, threshold=(5, 15),signal_weights = (a,b)))
    #print jobs
    #     jobs.append([X,sample,y,i,weight])
    #
    #     # Make the Pool of workers
    #print jobs
    max_number_of_cpu = mp.cpu_count()
    pool = ThreadPool(max_number_of_cpu -2)
    #     # Open the urls in their own threads
    #     # and return the results
    results = pool.map(get_fpr_tpr, jobs)
    print results
    import csv
    #import sys

    f = open("gridsearch_results.csv", 'a')
    try:
        writer = csv.writer(f)
        writer.writerow(('Weights', 'fpr', 'tpr'))
        for i in results:
            writer.writerow((i[0], i[1], i[2]))
    finally:
        f.close()

    pool.close()
    pool.join()


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

    # print tot_signal
    # tuned_params = [{'signal_weights': [([0, 1, 2], [0, 2, -2]), ([0, 1, 2], [0, 1, -5]), ([0, 1, 2], [0, 1, -2])]}]
    #
    #pred = dc.fit(tot_signal, issignalOrderedCorrectly=True).predict(tot_signal)
    # print pred
    # print "fpr,tpr",dc.roc(label)
    #fpr, tpr = dc.roc(label)
    a, b ,c ,d ,e ,f= range(-3,3,1), range(-3,3,1),range(-3,3,1),range(-3,3,1), range(-3,3,1),range(-3,3,1)
    #print "a,b,c", a,b,c,d,e,f
    np.vectorize(tester)(*np.ix_(a, b,c,d,e,f))

    #print tot_param1
    #print tot_param2

    search_grid(zip(tot_param1,tot_param2),anomaly_threshold,num_of_dc_selected,lifespan)