import myclassifiers
import myAdaboost
import dDCA
import util
import numpy as np
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from myclassifiers_test import testClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score,roc_curve, auc










if __name__ == '__main__':
    my_data = np.genfromtxt('nsl_kdd_15.csv', dtype=None, delimiter=',')
    label = my_data[1:, -1]
    #print my_data
    label = np.array(util.convert_class(label, ['normal','anomaly']))[np.newaxis]
    #print label
    normalized_data, not_normalized_data = util.get_data(my_data[1:,0:-1], isnorminal_data_inside=True, isdata_normalized=False,  norminalAttributecolumn=[0,1])

    get_parallel_cord = False

    if get_parallel_cord == True:
        util.get_parallel_cordinate(normalized_data,label,my_data,class_column_title="class")


