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
import myclassifiers_var_nsa
import multiprocessing as mp
from sklearn.model_selection import StratifiedKFold


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

    def predict(self,antigen_plus_siganls):
        print "classifying antigens"

        pred =[]
        all_mcav = self.get_mcav(antigen_plus_siganls[0:, 0], self.migrated_DCs)
        #print "all_mcav", all_mcav
        self.all_mcav = np.array(all_mcav)
        #print all_mcav
        seen_antigen=[]

        for i,mcav in all_mcav:
            if i not in seen_antigen:

                if mcav > self.anomaly_threshold:
                    pred.append(self.class_label[1])
                else:
                    pred.append(self.class_label[0])
                seen_antigen.append(i)
        print "len of test data", len(pred)
        print "done classfication"
        return np.array(pred)

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
        return np.array(pred)

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

    def confusion_matrix1(self,y_true, y_pred, antigen_for_test_index):
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
                if y1 == self.class_label[0]:  # normal
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

def arranged_test(X_train,X_test):
    arrange = []
    normal = []
    normal_class = []
    # normal_antigen = []

    anomaly = []
    anomaly_class = []
    # anomaly_antigen = []


    tree = KDTree(X_train, leaf_size=30)

    for i, X in enumerate(X_test):
        dist, ind = tree.query([X], k=1)
        # print "i,X", i, X, ind
        # print "dist, ind, ytest,  predclass"
        # print dist[0][0],ind[0][0],y_test[i] ,y_train[ind[0][0]]
        arrange.append(y_train[ind[0][0]])
        # print X
        if y_train[ind[0][0]] == -1:
            normal.append(X)
            normal_class.append(-1)
            # normal_antigen.append(X[0])
        else:
            anomaly.append(X)
            anomaly_class.append(1)
            # anomaly_antigen.append(X[0])

    normal = np.array(normal)
    anomaly = np.array(anomaly)
    normal_class = np.array(normal_class)
    anomaly_class = np.array(anomaly_class)
    # normal_antigen = np.array(normal_antigen)
    # anomaly_antigen = np.array(anomaly_antigen)



    x_test = np.concatenate((normal, anomaly))
    y_test = np.concatenate((normal_class, anomaly_class))

    return x_test,y_test

def safe_signal_handler(X_train,X_test):
    len_xtrain = len(X_train[:, 5])

    S1 = np.concatenate((X_train[:, 5], X_test[:, 5]))
    S1 = (pd.rolling_mean(S1, window=100))

    S2 = np.concatenate((X_train[:, 16], X_test[:, 16]))
    S2 = (pd.rolling_mean(S2, window=100))

    S0 = []
    for i, j in zip(S1, S2):
        average = sum([i, j]) / 2.0
        S0.append(average * 2)
    S0 = np.array(S0)

    S = S0[len_xtrain:]

    return S0[99:len_xtrain],S

def danger_signal_handler(X_train,X_test):
    len_xtrain = len(X_train[:, 5])

    P1 = np.concatenate((X_train[:, 1], X_test[:, 1]))
    P1 = (pd.rolling_mean(P1, window=100))

    P2 = np.concatenate((X_train[:, 2], X_test[:, 2]))
    P2 = (pd.rolling_mean(P2, window=100))

    P0 = []
    for i, j in zip(P1, P2):
        average = sum([i, j]) / 2.0
        P0.append(average * 2)
    P0 = np.array(P0)

    P = P0[len_xtrain:]

    # comment out below to plot PAMP
    # plt.plot(range(len(P)),P)
    # plt.show()
    return P0[99:len_xtrain],P

def confusion_matrix(y_true, y_pred):

    print "getting confusion matrix"
    TN, FP, FN, TP = 0,0,0,0

    for y1,y2 in zip(y_true, y_pred):

        #print y1,y2
        if y1 ==-1:         #normal
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

def normalize(weights):
    """
    :param weights: the weights to be normalized to 1
    :return: a tupple containing all the new normalized weights
    """
    norm = sum(weights)
    return tuple(m / norm for m in weights)

class Boost():
    def __init__(self,weakLearners,num_of_gen_for_weight,dd):
        self.weakLearners = weakLearners
        self.num_of_gen_for_weight = num_of_gen_for_weight
        self.dd=dd


    def fit(self,X,y):
        nsa_results = []
        dca_results = []

        for i in self.weakLearners['nsa']:
            ans = i.fit(X[:, 1:], y).predict(X[:, 1:])
            #nsa_results.append(ans)
            nsa_results.append(ans)

        nsa_results = np.array(nsa_results)
        #print nsa_results.T

        for i in self.weakLearners['dca']:
            ans = i.fit(X,issignalOrderedCorrectly = True).predict1(X)
            #dca_results.append(ans)
            dca_results.append(ans)

        dca_results = np.array(dca_results)

        #print dca_results.T

        all = np.concatenate((nsa_results.T,dca_results.T), axis=1)

        #print all
        self.weight_ = self.get_weight(all, y, self.num_of_gen_for_weight)

        return self


    def predict(self,X):

        nsa_results = []
        dca_results = []

        for i in self.weakLearners['nsa']:
            pred = i.predict(X[:, 1:])
            nsa_results.append(pred)
        nsa_results = np.array(nsa_results)

        for i in self.weakLearners['dca']:
            ans = i.fit(X,issignalOrderedCorrectly = True).predict1(X)
            #dca_results.append(ans)
            dca_results.append(ans)

        dca_results = np.array(dca_results)
        all = np.concatenate((nsa_results.T, dca_results.T), axis=1)

        list_times_weight = np.multiply(all, self.weight_)
        print "y_test", list_times_weight
        summ = np.sum(list_times_weight, axis=1)
        print "sum", summ
        pred = np.sign(summ)

        self.summation = summ


        return  pred



    def get_weight(self,list_of_results,y_truth,number_of_attempts):

        size_of_x_axis = np.shape(list_of_results)[1]

        self.current_weight=[]
        self.current_accuracy = 0
        self.current_pred = 0
        self.current_sum = []

        print "getting weights ..."
        for i in range(number_of_attempts):
            new_random_weight = [random.uniform(0, 1) for i in range(size_of_x_axis)]
            #print new_random_weight
            weight = normalize(new_random_weight)
            list_times_weight = np.multiply(list_of_results,weight)
            #print "y_test",list_times_weight
            summ = np.sum(list_times_weight,axis=1)
            #print "sum",summ
            pred = np.sign(summ)

            #print "weight pred = ", pred

            score = accuracy_score(y_truth,pred)
            if score > self.current_accuracy:
                self.current_accuracy = score
                self.current_weight = weight
                self.current_pred = pred
                self.current_sum = summ
        print "weight for hypothesis", self.current_weight
        print "current_accuracy on training set", self.current_accuracy
        # print "[TN, FP], [FN, TP]", confusion_matrix(y_truth,self.current_pred)
        # fpr, tpr, thresholds = roc_curve(y_truth, self.current_sum, pos_label=-1)
        # plt.plot(tpr,fpr)
        # plt.show()
        file5 = open("weights_NSA_DCA_{0}_{1}.csv".format(multiplier,self.dd), "a")
        wtr5 = csv.writer(file5, delimiter=',')
        wtr5.writerows([self.current_weight])
        file5.close()

        return  self.current_weight


    #def predict(self,X):




#def movingavearge(Train,test)
#########################################################
# TEST
#########################################################
if __name__ == '__main__':
    multiplier_list = [1,5,10,15,20]
    #multiplier = 1
    for multiplier in multiplier_list:
        import time

        time_start = time.time()
        #multiplier = 6
        # dca1 = dca.DCA(dca_pop_size=0,is_it_static=True)
        data = get_data('data_new.csv', datatype=np.float64, delimiter=',', skiprows=1, ndmin=2)
        # print 'data length, ',len(data)
        new_data = data  # get_multiplied_antigen(data, multiplier=multiplier)
        # print 'new_data length, ',len(new_data)
        # print new_data
        # anomaly_threshold = (9234.0/22664)
        num_of_dc_selected = multiplier#20
        # antigen = new_data[:, 0]
        data_class = new_data[:, 3]

        label = np.array(convert_class(data_class,1))

        skf = StratifiedKFold(n_splits=10)
        print(skf)
        k = 1


        for dd in [
            (1, 1)]:  # ,(1,2),(1,3),(1,4),(2,1),(2,2),(2,3),(2,4),(3,1),(3,2),(3,3),(3,4),(4,1),(4,2),(4,3),(4,4)]:


            mean_roc_fpr = []
            mean_roc_tpr = []
            mean_roc_auc = []
            mean_instant_tpr = []
            mean_instant_fpr = []
            mean_instant_accu = []

            for train_index, test_index in skf.split(new_data, label):
                print "In fold number --------->", k
                k += 1
                print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = new_data[train_index], new_data[test_index]
                y_train, y_test = label[train_index], label[test_index]

                # X_train, X_test, y_train, y_test = train_test_split(new_data, label, train_size=0.90, random_state=42) #None
                print "y_test before convert", y_test

                an_count = 0
                for n in y_test:
                    if n == -1:
                        an_count += 1
                anomaly_threshold = an_count / len(y_test)

                X_test = get_multiplied_antigen(X_test, multiplier=multiplier)
                y_test = convert_class(X_test[:, 3],1)

                data_class = y_test  # new_data[:, 3]
                #safe = X_test[:, 2]
                # pamp = new_data[:, 7]
                #danger = X_test[:, 1]
                #antigen = X_test[:, 0]
                print "y_test after convert", y_test


                ###################################################################################
                #Get the signals for DCA
                #####################################################################################
                train_S,test_S= X_train[:,2], X_test[:,2]
                #print train_S
                #print test_S

                train_P, test_P  = X_train[:,1], X_test[:,1]
                #print train_P
               # print test_P



                #train_P,P = danger_signal_handler(X_train, X_test)

                y_train = y_train[:]  # 49 because the dca is initiallizing moving average with 50 data

                antigen = X_test[:,0]

                #for NSA
                antigen_train = X_train[:,0]  #I added 49 because of the window of sliding average go to the signal function above


                #To concatenate the antigen, dnager and safe signals for tests
                #pamp1 = pamp[np.newaxis]
                danger1 = test_P[np.newaxis]
                safe1 = test_S[np.newaxis]
                antigen1 = antigen[np.newaxis]      #I called it antigen1 which will stop the original antigen array shape from being
                #print "len of test danger", len(danger1), np.shape(danger1)
                #print "len of test safe", len(safe1), np.shape(safe1)
                #print "len of test antigen", len(antigen1), np.shape(antigen1)

                #print "\n --------------------------------------------"

                #for concatenate the antigen, dnager and safe signals for train
                danger_train = train_P[np.newaxis]
                safe_train = train_S[np.newaxis]
                antigen_train = antigen_train[np.newaxis]
                #print "len of train danger", len(danger_train),np.shape(danger_train)
                #print "len of train safe", len(safe_train), np.shape(safe_train)
                #print "len of train antigen", len(antigen_train), np.shape(antigen_train)


                #for one without pamp
                print "danger",danger1.T
                print np.shape(antigen1.T), np.shape(danger1.T),np.shape(safe1.T)
                tot_signal = np.concatenate((antigen1.T, danger1.T, safe1.T), axis=1)

                #for NSA
                tot_signal_train = np.concatenate((antigen_train.T, danger_train.T, safe_train.T), axis=1)
                #print "tot_signal_train,y_train",tot_signal_train,y_train


                no_of_weaklearners = dd      #(2,1) where the first element is for number of NSA and second is the number of DCA.


                no_of_weakLearners = no_of_weaklearners

                weakLearners = {}#[]

                nsa = []
                nsa_weight = []
                dca = []
                dca_weight = []

                for c in range(no_of_weakLearners[0]):
                    nsa.append(
                        myclassifiers_var_nsa.NsaConstantDetectorClassifier(number_of_detectors=500, self_radius_size=0.013,
                                                                    random_gen_param=(0.00000001, 1.0), class_label=(-1, 1)))

                for c in range(no_of_weakLearners[1]):
                    dca.append(
                        deterministicDCA(anomaly_threshold, num_of_dc_selected=num_of_dc_selected, lifespan=3,
                                         class_label=(-1, 1), threshold=(5, 15), signal_weights=([0, 1, 1], [0, 1, -2])))

                weakLearners['nsa'] = nsa
                weakLearners['dca'] = dca

                print weakLearners

                bc = Boost(weakLearners,5000,dd)
                bc.fit(tot_signal_train,y_train)
                pred = bc.predict(tot_signal)

                score = accuracy_score(y_test, pred)

                print "accuracy", score


                print "/n confusion matrix"
                [TN, FP], [FN, TP] = confusion_matrix2(y_test, pred, antigen)
                print "[TN, FP], [FN, TP]", [TN, FP], [FN, TP]

                instantenous_fpr = (FP / float(FP + TN))
                instantenous_tpr = (TP / float(TP + FN))

                print '/n instantenous_fpr', instantenous_fpr
                print 'instantenous_tpr', instantenous_tpr
                print 'instanteneous_auc', score

                mean_instant_fpr.append(instantenous_fpr)
                mean_instant_tpr.append(instantenous_tpr)
                mean_instant_accu.append(score)




                fpr, tpr, thresholds = roc_curve(y_test, bc.summation, pos_label=1)

                mean_roc_fpr.append(fpr)
                mean_roc_tpr.append(tpr)


                fpr1, tpr1 = fpr, tpr
                roc_auc = auc(fpr1, tpr1)

                #print "roc_auc", roc_auc

                mean_roc_auc.append(roc_auc)

            #print "mean_roc_tpr", mean_roc_tpr
            #print "mean_roc_fpr", mean_roc_fpr
            #print "mean_roc_auc", mean_roc_auc
            #print "mean_instant_tpr", mean_instant_tpr
            #print "mean_instant_fpr", mean_instant_fpr
            #print "man_instant_accu", mean_instant_accu

            #print "np.array(mean_inst_tpr)", np.array(mean_inst_tpr)
            mean_instant_tpr = np.mean(np.array(mean_instant_tpr))
            mean_instant_fpr = np.mean(np.array(mean_instant_fpr))
            mean_instant_accu = np.mean(np.array(mean_instant_accu))
            mean_roc_auc = np.mean(np.array(mean_roc_auc))

            try:
                mean_roc_fpr = np.mean(np.array(mean_roc_fpr), axis=0)
            except:
                mean_roc_fpr = mean_roc_fpr[-1]

            try:
                mean_roc_tpr = np.mean(np.array(mean_roc_tpr), axis=0)
            except:
                mean_roc_tpr = mean_roc_tpr[-1]

            print "mean_roc_tpr", mean_roc_tpr
            print "mean_roc_fpr", mean_roc_fpr
            print "mean_roc_auc", mean_roc_auc
            print "mean_instant_tpr", mean_instant_tpr
            print "mean_instant_fpr", mean_instant_fpr
            print "man_instant_accu", mean_instant_accu

            file1 = open("roc_curve_summary_mult_{0}_{1}.csv".format(multiplier, dd), "w")
            wtr1 = csv.writer(file1, delimiter=',')
            wtr1.writerows([["fpr", "tpr"]])

            print "\n"
            print "fpr,tpr"
            for f, t in zip(mean_roc_fpr, mean_roc_tpr):
                print f, t

                wtr1.writerows([[f, t]])

                # print "threshold", thresholds

            wtr1.writerows([["the mean_roc_auc ", "is", mean_roc_auc]])
            #wtr1.writerows([["TN", "FP", "FN", "TP"]])
            #wtr1.writerows([[TN, FP, FN, TP]])

            wtr1.writerows([["mean_instantenous_fpr", "mean_instantenous_tpr","mean_instant_accu"]])
            wtr1.writerows([[mean_instant_fpr, mean_instant_tpr,mean_instant_accu]])
            file1.close()

















