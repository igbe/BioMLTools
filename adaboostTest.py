from myclassifiers_test import testClassifier
import myclassifiers
import numpy as np
import matplotlib.pyplot as plt
from myAdaboost import boost
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,roc_curve, auc

my_data = np.genfromtxt('breast_cancer_test_norm.csv', delimiter=',')
normal = my_data[1:201, 1:6]  # my_data[1:201,1:6]
classs_normal = my_data[1:201, 6]
classs_test = my_data[1:400, 6]  # my_data[241:,6]#201:,6
test_data = my_data[1:400, 1:6]  # my_data[241:,1:6] or 201 #201:,1:6


def convert_class(y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    cl = []
    for i in y:
        if i == 0:
            cl.append(1)
        else:
            cl.append(-1)
    return cl


ytrain =  np.array(convert_class(classs_test))

train = test_data


no_of_weakLearners = 2
weakLearner = []
for i in range(no_of_weakLearners):
    #weakLearner.append((testClassifier,tresh[i]))
    weakLearner.append(myclassifiers.NsaConstantDetectorClassifier(number_of_detectors=500, self_radius_size=0.1,
                                                random_gen_param=(0.001, 1.0), class_label=(1, -1)))

#mc = myclassifiers.NsaConstantDetectorClassifier(number_of_detectors=500, self_radius_size=0.1,
#                                                     random_gen_param=(0.001, 1.0), class_label=(1, -1))
#weakLearner = testClassifier
#print testClassifier(train)
rounds = 2
hypothesis,error = boost(train,ytrain,weakLearner,rounds)

[TN, FP], [FN, TP] = confusion_matrix(ytrain, hypothesis)

print accuracy_score(y_true=ytrain,y_pred=hypothesis)

print TN, FP, FN, TP

print hypothesis
print error





#error_range = np.arange(0,1,0.1)
#plt.plot(error_range,[0.5,0.4,0.583,0.371,0.222,0.395,0.908,0.692,0.299,0.500])
#
