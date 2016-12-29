import myclassifiers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing

my_data = np.genfromtxt('breast_cancer_test_norm.csv', delimiter=',')
normal = my_data[1:201,1:6]#my_data[1:201,1:6]
classs_normal=my_data[1:201,6]
classs_test=my_data[1:400,6]#my_data[241:,6]#201:,6
test_data=my_data[1:400,1:6]#my_data[241:,1:6] or 201 #201:,1:6

tuned_params={'self_radius_size':[1.15,1.156]}#63,1.17]}


mc = myclassifiers.NsaConstantDetectorClassifier(number_of_detectors = 100, self_radius_size = 0.16, random_gen_param = (0.0001,1.0),
                 class_label = (1.0,2.0),otherParam=None)
gs = GridSearchCV(mc,param_grid=tuned_params,cv=2,n_jobs=2,verbose=100)
gs.fit(test_data,classs_test)#(normal,classs_normal)
print gs.best_params_