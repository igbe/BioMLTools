import myclassifiers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

mc=myclassifiers.NsaConstantDetectorClassifier(number_of_detectors = 1000, self_radius_size = 0.16,
                                               random_gen_param = (0.001,1.0),class_label = (1.0,2.0))

my_data = np.genfromtxt('breast_cancer_test_norm.csv', delimiter=',')
normal = my_data[1:201,1:6]#my_data[1:201,1:6]
classs_normal=my_data[1:201,6]
classs_test=my_data[241:,6]#my_data[241:,6]#201:,6
test_data=my_data[241:,1:6]#my_data[241:,1:6] or 201 #201:,1:6



k=mc.fit(normal,classs_normal)
#X=[[2,3],[3,4]]
#y=['self','self']


#X=np.array(X)
#y=np.array(y)
#k=mc.fit(X,y)

#print k

#X1=[[0,40]]
#X1 = np.array(X1)
#plt.scatter(y[:,0],y[:,1])
#plt.show()
#print mc.predict(test_data, y=(1.0,2.0))
print mc.score(test_data, classs_test)

tuned_params={'self_radius_size':[i/10.0 for i in range(0,21)]}

gs = GridSearchCV(myclassifiers.NsaConstantDetectorClassifier(),tuned_params)

gs.fit(test_data,classs_test)
gs.best_params_