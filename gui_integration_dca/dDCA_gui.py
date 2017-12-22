import numpy as np
import random

from scipy import interp
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score,roc_curve, auc
import csv
from sklearn.neighbors import KDTree
import pandas as pd
#import myclassifiers_var_nsa
import multiprocessing as mp
from sklearn.cross_validation import StratifiedKFold


def get_data(filename, datatype='float', delimiter=',', skiprows=0, ndmin=2):
    data = np.loadtxt(
        fname=filename,
        dtype='float',#np.float64
        delimiter=',',
        skiprows=skiprows,
        ndmin=2)
    return data

def convert_class(y,normal_class):
    """

    :param y: the classes to be converted to -1 for normal and +1 for anomaly
    :param normal_class: the label for the normal instances
    :return:
    """
    #le = LabelEncoder()
    #y = le.fit_transform(y)
    cl = []
    for i in y:
        if i == normal_class:
            cl.append(-1)
        else:
            cl.append(1)
    return cl

def get_multiplied_antigen(data, multiplier=10):
    data2 = []
    for i in range(len(data)):
        for j in range(multiplier):
            data2.append(data[i])
    # print data2
    data2 = np.array(data2)
    # print data2
    return data2

class deterministicDCA(BaseEstimator,ClassifierMixin):
    def __init__(self,anomaly_threshold,num_of_dc_selected = 100,lifespan = 100,class_label = (-1,1),threshold =(5, 15),signal_weights = ([0,1,2],[0,1,-2])):
        """

        :param anomaly_threshold: this is the threshold to be used for deciding if an antigen is anomalous or not
        :param num_of_dc_selected: This is the total number of cells to be initialized for this system
        :param lifespan: This is how long the DC's will sample an antigen before dying off or migrating.
        :param class_label: this is the 1xn array of teh class lebels.
        :param threshold: this is an alternative to to using a fixed lifespan. for this, the DC lifespans or threshold will be
                            random within a specific range.
        :param signal_weights: this is the weights to be used for the signals. it is of a 2x3 shape. where 2 is for the
                                number of signal equations i.e., csm and k, and the 3 is for the weights of different
                                signals in each of the equations which are wP,wD and wS for weight of PAMP, weight of
                                Danger and weight of Safe signal respectively.
        """
        self.threshold = threshold
        self.lifespan = lifespan
        self.class_label = class_label
        self.num_of_dc_selected = num_of_dc_selected
        self.anomaly_threshold = anomaly_threshold
        self.signal_weights = signal_weights
        #pass

    def initialize_dDCA(self,num_of_cells):
        """
        num_of_cells is the number of DC you want to create, and threshold is a list with [min,max] threshold
        DC structure is given by:
        [[antigen],(signal),[csm,k],lifespan,location(int),no_cell_iter,threshold,context(int)]'
        where [signal] is the DC signal matrix, [antigen] the DC antigen matrix, [output_matrix] output signal matrix i.e csm and k, lifespan is the age
        of the dDCA cell, and location(int) is the location in cells where this dc is located, threshold, and context(int) is the context

        :return: a list of cells in format [[antigen],(signal),[csm,k],lifespan,location(int),no_cell_iter,threshold,context(int)]
        """
        DC = []
        for i in range(num_of_cells):
            if isinstance(self.threshold, tuple):
                thresh = self.threshold[0] + (self.threshold[1] - self.threshold[0]) * random.random()
            else:
                thresh = self.threshold
            dc = {'antigen': [], 'k': 0, 'csm':0,'lifespan': self.lifespan, 'thresh':thresh,'location': 0, 'context': 0}
            DC.append(dc)
        #print DC
        return DC

    def select_dDCA(self,dc_population, num_dc_to_select):
        # print 'dc_population',dc_population
        location = random.sample(range(len(dc_population)), num_dc_to_select)
        selected_dc = [dc_population[i] for i in location]
        # print selected_dc
        k = 0
        for i in location:
            selected_dc[k]['location'] = i
            k += 1
        return selected_dc, location

    def fit(self,antigen_plus_siganls,issignalOrderedCorrectly = True):
        #print "antigen_plus_siganls", antigen_plus_siganls
        """
        A function for fitting the data. Requires an array
        :param antigen_plus_signals: this an array in the shape of a nx4. With four columns being the antigen, the PAMP signal,
                                        the Danger signal and the Safe signal. Note that this must follow this other to get
                                        the right results

        :return: self
        """
        print "fitting dca classifier"

        if issignalOrderedCorrectly ==False:
            print "Signals not ordered correctly. Read the dDCA's fit function help string on how to order your input signals"

            return 0
        try:
            dimension = np.shape(antigen_plus_siganls)
            row_dimension = dimension[0]
            col_dimension = dimension[1]
        except:
            print "antigen_plus_siganls array must be a numpy array"
            return 0

        self.migrated_DCs = []
        antigenCounter = 0

        cells = self.initialize_dDCA(self.num_of_dc_selected)
        #print cells
        selected_DCs, locations = self.select_dDCA(cells, self.num_of_dc_selected)
        #print locations
        #print selected_DCs


        for i in range(len(antigen_plus_siganls)):  # len(new_data)
            sampled_antigen = antigen_plus_siganls[i, 0]
            cell_index = antigenCounter % self.num_of_dc_selected

            selected_DCs[cell_index]['antigen'].append( sampled_antigen)
            #print selected_DCs

            antigenCounter = antigenCounter + 1  # count the number of antigens and use it to get the cell index to assign it to.

            #Note that the signal processing equation used below can be replaced instead by the equation used in \
            #the non-deterministic DCA, which in-coporates the PAMP signal. Note for this, you can try to see what\
            #happens when all PAMP and Danger signals are combined to form one Danger siganl and applied into the equation below.

            if col_dimension == 3:
                signal_weights = self.signal_weights#([0,1,2],[0,1,-2])
                csm = ((signal_weights[0][1] * antigen_plus_siganls[i, 1]) + (signal_weights[0][2] * antigen_plus_siganls[i, 2]))
                k = ((signal_weights[1][1] * antigen_plus_siganls[i, 1]) + (signal_weights[1][2] * antigen_plus_siganls[i, 2]))

            elif col_dimension == 4:
                #print "in column four"
                signal_weights = self.signal_weights
                csm = (signal_weights[0][0] * antigen_plus_siganls[i,1]) + (signal_weights[0][1] * antigen_plus_siganls[i,2])+ (signal_weights[0][2] * antigen_plus_siganls[i,3])
                k = (signal_weights[1][0] * antigen_plus_siganls[i,1]) + (signal_weights[1][1] * antigen_plus_siganls[i,2])+ (signal_weights[1][2] * antigen_plus_siganls[i,3])
                #print "sampled_antigen",sampled_antigen,"k:",k," csm:",csm
                #print "k:",k," csm:",csm
            else:
                print "check your antigen_plus_siganls array dimensions"
                return 0
                #print selected_DCs
            # print selected_DCs,'------>', csm, k
            for j in range(self.num_of_dc_selected):
                # print selected_DCs
                # update signals
                #selected_DCs[j][1] = (safe[i], danger[i])
                #for reducing lifespan
                #print "in second for"
                selected_DCs[j]['lifespan']  -= csm
                # the outpts csm and k

                selected_DCs[j]['k'] += k
                #selected_DCs[j]['csm'] +=csm       #incase you

                # if lifespan is less than zero, re-initialize the cell
                #if (selected_DCs[j]['csm'] > selected_DCs[j]['thresh']) and (selected_DCs[j]['antigen'] != []):
                if (selected_DCs[j]['lifespan'] <= 0)and (selected_DCs[j]['antigen'] != []):

                    # to add a context k <0 is givn a normal context and vis versa
                    #print "selected_DCs[j]['k']", selected_DCs[j]['k']
                    if selected_DCs[j]['k'] < 0:
                        selected_DCs[j]['context'] = 'normal'
                    else:
                        selected_DCs[j]['context'] = 'anormaly'

                    self.migrated_DCs.append(selected_DCs[j])
                    selected_DCs[j] = self.initialize_dDCA(1)[0]

                #print selected_DCs[j]
                #print self.migrated_DCs
        return self




    def get_mcav(self,antigen, migrated_DCs):

        all_mcav = []
        #print 'migrated_DCs', migrated_DCs
        for i in antigen:
            num_cells, num_antigen=0,0
            anomaly_context = 0
            normal_context = 0
            #print 'migrated_DCs', migrated_DCs
            for l in migrated_DCs:
                #print "migrated DC ------>",l
                #print "l",l

                if (i in l['antigen']):         #l['antigen']
                    if (l['context'] !='normal'):
                        anomaly_context +=1
                    else:
                        normal_context +=1
            try:
                mcav = float(anomaly_context)/(normal_context + anomaly_context)
                #print "num_cells", num_cells, "num_antigen", num_antigen, mcav
            except:
                #print "setting mcav as 0"
                mcav = 0
            all_mcav.append((i, mcav))
        # a=[]
        # #print "########all_mcav#############"
        # file3 = open("index_mcav.csv", "a")
        # wtr3 = csv.writer(file3, delimiter=',')
        # for i in all_mcav:
        #     a.append(i[1])
        #     #print i
        #     wtr3.writerows([[i[0],i[1]]])
        #
        # file3.close()
        #
        # plt.plot(range(len(a)),a)
        # plt.show()
        #print "all_mcav",all_mcav
        return all_mcav

    def predict1(self,antigen_plus_siganls):
        print "classifying antigens"
        print "len of test data",len(antigen_plus_siganls)
        pred =[]
        all_mcav = self.get_mcav(antigen_plus_siganls[0:, 0], self.migrated_DCs)
        #print "all_mcav", all_mcav
        self.all_mcav = np.array(all_mcav)
        #print all_mcav


        for i,mcav in all_mcav:
            if mcav > self.anomaly_threshold:
                pred.append(self.class_label[1])
            else:
                pred.append(self.class_label[0])

        print "done classfication"
        s = list(set(all_mcav))
        s.sort()
        return np.array(pred), s

    def confusion_matrix2(self,y_true, y_pred):

        print "getting confusion matrix"
        TN, FP, FN, TP = 0,0,0,0

        for y1,y2 in zip(y_true, y_pred):

            #print y1,y2
            if y1 ==self.class_label[0]:         #normal
                if y2 ==y1:
                    TN+=1
                else:
                    FP +=1
            else:
                if y2 ==y1:
                    TP +=1
                else:
                    FN +=1
        return [TN, FP], [FN, TP]


    def roc(self,true_label):
        """
        This function gets the ROC curve. Note, the line in pos_label=self.class_label[1]) could have been
        pos_label=self.class_label[0]) but the later creates a ROC curve which is inverted. But when the same data is
        plotted using a self made ROC curve function, it create a curve which is not inverted. Hence, I decide to use
        the former.
        :param true_label: this is the actual label of the test or train data
        :return:  false positive rate, and true positive rate
        """
        print "getting the ROC curve"

        # Note that the two variables used below is to solve the issue of sckit-learns tool plotting very few points only
        # when self.class_label[1]) is used. Hence, all traces of this two can be removes, and the code will still work
        # but given that this later curve will have few points, it will look ugly
        ex_tpr = 0.0
        ex_fpr = np.linspace(0, 1, 100)

        #print true_label
        #print list(self.all_mcav[:,1])

        fpr, tpr, thresholds = roc_curve(true_label, self.all_mcav[:,1], pos_label=self.class_label[1])

        ex_tpr += interp(ex_fpr, fpr, tpr)
        ex_tpr[0] = 0.0
        #print "ex_tpr",ex_tpr
        #print "ex_fpr", ex_fpr
        #print thresholds
        #plt.plot(fpr, tpr, 'b-', lw=1)
        #plt.show()
        return ex_fpr,ex_tpr        # These are same as fpr, and tpr

    def score(self, X, y, sample_weight=None):          #antigen_plus_siganls, label
        print "calling dDCA's score method"
        #print X
        #print y
        pred_score_ = self.predict(X)
        #print "pred_score_",pred_score_
        print "accuracy",accuracy_score(y_true=y,y_pred=pred_score_)
        #print "self.migrated_DCs", self.migrated_DCs
        return  accuracy_score(y_true=y,y_pred=pred_score_)

def confusion_matrix2(y_true, y_pred, antigen_for_test_index):
    """

    :param y_true: the true labels of the instances being evaluated
    :param y_pred:  the predicted label of the instances
    :param antigen_for_test_index: the antigen corresponding to these labels. this is gotten by antigen[test/train]
            i.e, test if this is being evaluated on test and training if evaluated on trainnig.
    :return:
    """

    print "getting confusion matrix"
    TN, FP, FN, TP = 0, 0, 0, 0
    already_processed = []
    for y1, y2, antigen in zip(y_true, y_pred, antigen_for_test_index):
        # print "index_proc",antigen
        if antigen not in already_processed:
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
            already_processed.append(antigen)

    return [TN, FP], [FN, TP]


#def movingavearge(Train,test)
#########################################################
# TEST
#########################################################
if __name__ == '__main__':
    multiplier = 1
    #multiplier = 1

    data = get_data('data_new.csv', datatype=np.float64, delimiter=',', skiprows=1, ndmin=2)
    # print 'data length, ',len(data)
    new_data = data  # get_multiplied_antigen(data, multiplier=multiplier)

    num_of_dc_selected = multiplier#20
    # antigen = new_data[:, 0]
    data_class = new_data[:, 3]

    label = np.array(convert_class(data_class,1))

    ################################################################
    #To get the anomaly threshold to be used
    #####################################################
    an_count = 0
    for n in label:
        if n == -1:
            an_count += 1
    anomaly_threshold = an_count / len(new_data)
    ################################################################




    ################################################################
    #To multiply the data instances
    ######################################################
    X_test = get_multiplied_antigen(new_data, multiplier=multiplier)
    y_test = convert_class(X_test[:, 3],1)

    data_class = y_test  # new_data[:, 3]

    #print "y_test after convert", y_test, "len", len(y_test)
    #################################################################



    ###################################################################################
    #Get the signals for DCA
    #####################################################
    test_S= X_test[:,2]
    #print train_S
    #print test_S

    test_P  = X_test[:,1]
    #print train_P

    antigen = X_test[:,0]

    #To concatenate the antigen, dnager and safe signals for tests
    #pamp1 = pamp[np.newaxis]
    danger1 = test_P[np.newaxis]
    safe1 = test_S[np.newaxis]
    antigen1 = antigen[np.newaxis]      #I called it antigen1 which will stop the original antigen array shape from being
    #print "len of test danger", len(danger1), np.shape(danger1)
    #print "len of test safe", len(safe1), np.shape(safe1)
    #print "len of test antigen", len(antigen1), np.shape(antigen1)
    #print "danger",danger1.T
    #print np.shape(antigen1.T), np.shape(danger1.T),np.shape(safe1.T)
    tot_signal = np.concatenate((antigen1.T, danger1.T, safe1.T), axis=1)

    ###################################################################################


    ##########################################################################################
    #Create an instance of the model class and pass data to model and predict
    ##############################################################################
    dc = deterministicDCA(anomaly_threshold, num_of_dc_selected=num_of_dc_selected, lifespan=2,
                          threshold=(5, 15), signal_weights=([0, 1, 1], [0, 1, -2]))

    pred, mcav = dc.fit(tot_signal, issignalOrderedCorrectly=True).predict1(tot_signal)
    #print "mcav", mcav

    ###########################################################################################




    ##########################################################################################
    #To compute performance metrics
    ##########################################################################
    score = accuracy_score(y_test, pred)

    print "accuracy", score


    print "\nconfusion matrix"
    [TN, FP], [FN, TP] = confusion_matrix2(y_test, pred, antigen)
    print "[TN, FP], [FN, TP]", [TN, FP], [FN, TP]

    instantenous_fpr = (FP / float(FP + TN))
    instantenous_tpr = (TP / float(TP + FN))

    print '\ninstantenous_fpr', instantenous_fpr
    print 'instantenous_tpr', instantenous_tpr
    print 'instanteneous_auc', score


    #fpr, tpr, thresholds = roc_curve(y_test, bc.summation, pos_label=1)
    fpr, tpr = dc.roc(y_test)


    fpr1, tpr1 = fpr, tpr
    roc_auc = auc(fpr1, tpr1)
    # print "roc_auc", roc_auc
    #####################################################################################


















