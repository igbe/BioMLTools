import myclassifiers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

mc=myclassifiers.NsaConstantDetectorClassifier(number_of_detectors = 2000, self_radius_size = 0.16,
                                               random_gen_param = (0.001,1.0),class_label = (1.0,2.0))

my_data = np.genfromtxt('breast_cancer_test_norm.csv', delimiter=',')
normal = my_data[1:201,1:6]#my_data[1:201,1:6]
classs_normal=my_data[1:201,6]
classs_test=my_data[202:350,6]#my_data[241:,6]#201:,6
test_data=my_data[202:350,1:6]#my_data[241:,1:6] or 201 #201:,1:6



#k=mc.fit(normal,classs_normal)


#X1=[[0,40]]
#X1 = np.array(X1)
#plt.scatter(y[:,0],y[:,1])
#plt.show()
#print mc.predict(test_data, y=(1.0,2.0))
#print mc.score(test_data, classs_test)

#clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,scoring='%s_macro' % score)
#clf.fit(X_train, y_train)
#print("Best parameters set found on development set:")
#print()
#print(clf.best_params_)

#tuned_params={'self_radius_size':[1.15,1.163,1.17]}


#gs = GridSearchCV(myclassifiers.NsaConstantDetectorClassifier(),param_grid=tuned_params,n_jobs=100,verbose=100)
#print gs

#gs.fit(normal,classs_normal)#(normal,classs_normal)
#print gs.best_params_
#print gs.best_score_

from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing



#cv = StratifiedKFold(classs_normal,n_folds=3,random_state=1)
#cv = StratifiedKFold(cdata,n_folds=2,random_state=1)

fig = plt.figure(figsize=(7,5))

mean_tpr = 0.0
mean_fpr = np.linspace(0,1,100)

le = preprocessing.LabelEncoder()



#for i, (train, test) in enumerate(cv):
    #print i, (train, test)
    # normal = []
    # classs_normal = []
    # test_list = []
    # classs_test = []
    # for data in ndata[train]:
    #     if data[5] ==1.0:
    #         normal.append(data[0:5])
    #         classs_normal.append(data[5])
    # for data in ndata[test]:
    #     test_list.append(data[0:5])
    #     classs_test.append(data[5])
    # normal = np.array(normal)
    # classs_normal = np.array(classs_normal)
    # test_list = np.array(test_list)
    # classs_test = np.array(classs_test)

#
k = mc.fit(normal, classs_normal).predict(test_data)
print classs_test
#print "k[:,1]", k#[:,1]
#fpr, tpr, threshold, roc_auc =mc.roc_curve_main(classs_test, pos_label=2.0)
fpr, tpr = mc.roc(test_data,classs_test)

#fpr,tpr,threshold = roc_curve(classs_test,k[:],pos_label=2)
print "fpr,tpr,threshold", fpr,tpr
mean_tpr +=interp(mean_fpr,fpr,tpr)
mean_tpr[0] = 0.0
#roc_auc = auc(fpr,tpr)
#print "auc = ", roc_auc

plt.plot(fpr,tpr,'ro',lw = 3,color='blue')#,label= "ROC fold %d (area = %0.2f)"%(i+1,roc_auc))
plt.xlabel('false positives')
plt.ylabel(('true positives'))
#
plt.plot([0,1],[0,1],linestyle='--',color=(0.6,0.6,0.6),label='random guessing')
# mean_tpr /= len(cv)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr,mean_tpr)
# print "mean_auc= ", mean_auc
#
#
# plt.plot(mean_fpr,mean_tpr,'k--', label= "mean ROC (area = %0.2f)"%(mean_auc),lw=2)
plt.show()
#
