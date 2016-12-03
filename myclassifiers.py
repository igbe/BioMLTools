from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y,check_array,check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
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

        for i in range(len(X)):
            dist, ind = kdt.query([X[i]], k=1)
#            print 'dist =', dist

            if dist >= self.self_radius_size:
                predicted_class.append(self.class_label[0])
            else:
                predicted_class.append(self.class_label[1])

        return predicted_class

    def score(self, X, y, sample_weight=None):
        #print X
        pred_score_ = self.predict(X)
        #print pred_score_
        #print accuracy_score(y_true=y,y_pred=pred_score_)
        return  accuracy_score(y_true=y,y_pred=pred_score_)







