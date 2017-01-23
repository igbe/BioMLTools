import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
import random
from sklearn.metrics import accuracy_score,roc_curve, auc
from dDCA import *


class testclass():
    def __init__(self,a,b):
        self.a = a
        self.b = b

    def answer(self):
        return self.a + self.b


class myGridsearch:

    def __init__(self,classifier_instance,param_grid,cv=2,n_jobs=2):

        pass


    def get_score(self):
        pass

tot_param1 =[]
tot_param2 =[]

def tester(*args):
    global tot_param1
    global  tot_param2
    param1 = list(args[0:3])
    param2 = list(args[3:])
    #dc = deterministicDCA(anomaly_threshold, num_of_dc_selected=num_of_dc_selected, lifespan=10, threshold=(5, 15),
    #                      signal_weights=([0, 1, 2], [0, 1, -2]))
    print "param1", param1
    print "param2", param2
    if 0 in param1[1:]:
        print "0 in danger or safe"
        return 0
    if 0 in param2[1:]:
        print "0 in danger or safe"
        return 0

    else:
        if (param1 or map(abs, param1)) in tot_param1:
            print param1, param2
            return 0

        if (param2 or map(abs, param2))  in tot_param2:
            print param1, param2
            return 0
        if ((param1 == param2) and ((param1 or map(abs, param1)) in tot_param1)):
            return  0
        tot_param1.append(param1)
        tot_param2.append(param2)
    print tot_param1
    print tot_param2



if __name__ == '__main__':
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
    a, b ,c ,d ,e ,f= range(-2,2,1), range(-2,2,1),range(-2,2,1),range(-2,2,1), range(-2,2,1),range(-2,2,1)
    #print "a,b,c", a,b,c,d,e,f


    grids = np.vectorize(tester)(*np.ix_(a, b,c,d,e,f))
    addition = grids

# def my_func(x, y, z):
#     print "x,y,z",x,y,z
#     return (x + y + z) / 3.0, x * y * z, max(x, y, z)