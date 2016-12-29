from myclassifiers_test import testClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier as ada
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from boosting import boost
from data import adult

#train = [(1,1),(2,1),(3,1),(4,-1),(5,-1),(6,-1),(1,1),(7,-1),(8,-1),(9,-1),(10,-1)]
#test =  [(1,1),(2,1),(3,1),(4,-1),(5,-1),(6,-1),(1,1),(7,-1),(8,-1),(9,-1),(10,-1)]


train, test = adult.load()
print len(train)

weakLearner = testClassifier
rounds = 20
h = boost(train, weakLearner, rounds)

