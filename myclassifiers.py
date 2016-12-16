from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y,check_array,check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score,roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KDTree
import random
import numpy as np

class NsaConstantDetectorClassifier (BaseEstimator,ClassifierMixin):


    def __init__(self,number_of_detectors = 1000, self_radius_size = 0.16, random_gen_param = (0.0001,1.0),
                 class_label = (1.0,2.0),otherParam=None):
        """

        :param number_of_detectors: int value of the number of detector you will like to generate
        :param self_radius_size:  the self radius of the normal samples
        :param random_gen_param: a tuple for the random generator use for generating initial detector positions or points
            default is is (0.0001,1) which mean generator range will start from very small values. Note also that this
             can cause the random points to be more distributed towards more small values. The final value 1 means that
             the maximum value of the search space is 1 and 0.0001 means that the minimum value is 0.0001.
        :param class_label:this is a tuple containing the two possible labels found in the dataset. To be supplied.
            the first index in class_label is that of the normal and the second is for the abnormal
        :param otherParam: not used for now
        """

        self.number_of_detectors = number_of_detectors
        self.self_radius_size = self_radius_size
        self.random_gen_param = random_gen_param
        self.class_label = class_label

    def fit(self,X,y):
        """

        :param X: this is a numpy array of the self_set without the class_label
        :param y: array-like, shape= [n_samples]. this is the vector of target class labels
        :return: it returns a self object
        """
        #print "self.number_of_detectors ", self.number_of_detectors
        #To check and ensure that the parameter types correspond to what is needed or expected, we run a type check

        if type(self.number_of_detectors) != int:
            raise Exception("number_of_detectors must be an integer value")
        elif type(self.self_radius_size) != float:
            raise Exception ('self_radius_size must be an floating pont value')
        elif type(self.random_gen_param) != tuple:
            raise Exception ('random_gen_param must be a tuple value')
        elif type(X) != np.ndarray:
            raise Exception ('self list X must be a numpy array')
        elif type(y) != np.ndarray:
            raise Exception ('class label y must be a numpy array')

        normal = []
        #attack = []
        j = 0
        for i in X:
            #print "X = ", i, y[j], self.class_label[0]
            if y[j] == self.class_label[0]:
                normal.append(i)
            #else:
            #    attack.append(i)
            j+=1

        len_normal_ = len(normal)
        #len_attack_ = len(attack)
        X = np.array(normal)
        #attack = np.array(attack)



        #To find the number of columns in the self_set matrix whic will help use generate same dimension of random detectors
        #X = np.array(X)
        dimension = np.shape(X)[1]
        min = self.random_gen_param[0]
        max = self.random_gen_param[1]

        #this list is only used here inside this class hence, the "_" at the end of the detector_list name

        self.detector_list_=[]
        kdt = KDTree(X, leaf_size=30, metric='euclidean')

        while len(self.detector_list_) < self.number_of_detectors:
            #print len(self.detector_list_)
            new_random_detector = [10 ** random.uniform(np.log10(min), np.log10(max)) for i in range(dimension)]

            dist, ind = kdt.query([new_random_detector], k=1)
            #print 'first x distance: ', dist

            if dist > self.self_radius_size:
                #print 'second d distance: ',dist
                if len(self.detector_list_) == 0:
                    self.detector_list_.append(new_random_detector)
                else:
                    kdt = KDTree(self.detector_list_, leaf_size=30, metric='euclidean')
                    dist2, ind = kdt.query([new_random_detector], k=1)
                    if dist2 > self.self_radius_size:
                        #print  dist2
                        self.detector_list_.append(new_random_detector)
        self.detector_list_ = np.array(self.detector_list_)
        return self

    def predict(self,X):

        """

        :param X: this is an array of the data to be predicted. note if it has only 1 element, then use [element]
            instead of just passing element.
        :return: a list containing the predicted class labels for the provided data.
        """

        #check is fit had been called
        check_is_fitted(self,'detector_list_')

        #input validation. This will display error message if there is no input
        #X = check_array(X)

        #array to hold the predicted label
        predicted_class=[]

        kdt = KDTree(self.detector_list_, leaf_size=30, metric='euclidean')
        self.distanceValues_ = []
        for i in range(len(X)):
            dist, ind = kdt.query([X[i]], k=1)
#            print 'dist =', dist
            self.distanceValues_.append(dist[0])

            if dist >= self.self_radius_size:
                predicted_class.append(self.class_label[0])
            else:
                predicted_class.append(self.class_label[1])
        self.distanceValues_ = np.array(self.distanceValues_)

        return predicted_class

    def score(self, X, y, sample_weight=None):
        #print X
        pred_score_ = self.predict(X)
        #print pred_score_
        #print accuracy_score(y_true=y,y_pred=pred_score_)
        return  accuracy_score(y_true=y,y_pred=pred_score_)


    def roc(self,test_data,ground_truth):
        """

        :param test_data: we need this again even though it has been predicted before because we will have to iterate
            over it for new values of self_radius
        :param ground_truth: the real truth
        :return: fpr1, tpr1 both are numpy arrays
        """

        option = self.distanceValues_
        #print self. distanceValues_
        #print option
        np.sort(option)
        copy_radius_ =self.self_radius_size
        fpr1=[]
        tpr1=[]
        for i in range(len(option)+1):
            #print self.self_radius_size
            if i >len(option)-1:
                self.self_radius_size = (max(self.distanceValues_) +1)[0]

                #self_radius change requires another prediction. Not the stackexchange question
                predicted_class_ = self.predict(test_data)
                #print confusion_matrix(ground_truth, predicted_class_)
                try:
                    [TN, FP], [FN, TP] = confusion_matrix(ground_truth, predicted_class_)
                except:
                    TN = confusion_matrix(ground_truth, predicted_class_)[0][0]
                    FP, FN, TP = 0,0,0

                #print "TN,FP,FN,TP", TN, FP, FN, TP
                try:
                    fpr = float(FP) / (FP + TN)
                except:
                    fpr = 0.0
                try:
                    tpr = float(TP) / (TP + FN)
                except:
                    tpr = 0.0
                #print "fpr, tpr", fpr, tpr
                fpr1.append(fpr)
                tpr1.append(tpr)

            else:
                self.self_radius_size = option[i][0]
                predicted_class_ = self.predict(test_data)
                #print confusion_matrix(ground_truth, predicted_class_)

                #if below is run with only normal, it will create issues at one point to solve this, use try and except
                try:
                    [TN, FP], [FN, TP] = confusion_matrix(ground_truth, predicted_class_)
                except:
                    TN = confusion_matrix(ground_truth, predicted_class_)[0][0]
                    FP, FN, TP = 0,0,0
                #print "TN,FP,FN,TP", TN, FP, FN, TP
                #print [TN, FP], [FN, TP]
                try:
                    fpr = float(FP) / (FP + TN)
                except:
                    fpr = 0.0
                try:
                    tpr = float(TP) / (TP + FN)
                except:
                    tpr = 0.0
                #print "fpr, tpr", fpr, tpr
                fpr1.append(fpr)
                tpr1.append(tpr)

        self.self_radius_size = copy_radius_
        #print self.self_radius_size

        return fpr1, tpr1











        # print predictions
        # tp, tn, fn, fp = 0.0, 0.0, 0.0, 0.0
        # for l, m in enumerate(ground_truth):
        #     if m == predictions[l] and m == 1:
        #         tp += 1
        #     if m == predictions[l] and m == 0:
        #         tn += 1
        #     if m != predictions[l] and m == 1:
        #         fn += 1
        #     if m != predictions[l] and m == 0:
        #         fp += 1
        # `
        # return tn / (tn + fp)










