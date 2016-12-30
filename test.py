from myclassifiers_test import testClassifier
import numpy as np
import matplotlib.pyplot as plt
from myAdaboost import boost

train = [1,2,3,4,5,6,7,8,9,10]
ytrain = [ 1 , 1 , 1 ,-1 ,-1, -1,  1,  1,  1, -1]


weakLearner = testClassifier
print testClassifier(train)
rounds = 100
hypothesis,error = boost(train,ytrain,weakLearner,rounds)
print hypothesis
print error
#error_range = np.arange(0,1,0.1)
#plt.plot(error_range,[0.5,0.4,0.583,0.371,0.222,0.395,0.908,0.692,0.299,0.500])
#
