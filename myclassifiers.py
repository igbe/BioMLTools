from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KDTree

class NsaConstantDetectorClassifier(BaseEstimator,ClassifierMixin):


    def __init__(self,number_of_detectors = 1000, self_radius_size = 0.16, random_gen_param = (0.0001,1.0), otherParam=None):
        """

        :param number_of_detectors: int value of the number of detector you will like to generate
        :param self_radius_size:  the self radius of the normal samples
        :param random_gen_param: a tuple for the random generator use for generating initial detector positions or points
            default is is (0.0001,1) which mean generator range will start from very small values. Note also that this
             can cause the random points to be more distributed towards more small values.
        :param otherParam: not used for now
        """

        self.number_of_detectors = number_of_detectors
        self.self_radius_size = self_radius_size
        self.random_gen_param = random_gen_param



