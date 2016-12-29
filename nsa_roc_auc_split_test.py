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
classs_test=my_data[230:300,6]#my_data[241:,6]#201:,6
test_data=my_data[230:300,1:6]#my_data[241:,1:6] or 201 #201:,1:6

fig = plt.figure(figsize=(7,5))

plot_others = False
n = 5   #no of times to run
t = 0
mean_fpr =np.zeros(len(test_data)+1)
mean_tpr = np.zeros(len(test_data)+1)
for i in range(n):

    mc=myclassifiers.NsaConstantDetectorClassifier(number_of_detectors = 2000, self_radius_size = 0.16,
                                                   random_gen_param = (0.001,1.0),class_label = (1.0,2.0))

    print "########## Fitting the data #############"

    k = mc.fit(normal, classs_normal).predict(test_data)
    #print k
    #print classs_test
    print "########## Plotting ROC #############"
    fpr, tpr = mc.roc(test_data,classs_test)#normal, classs_normal)#test_data,classs_test)
    #print fpr,tpr

    mean_fpr = np.sum([mean_fpr,fpr], axis=0)   #  np.sum([[0, 1], [0, 5]], axis=0)
    mean_tpr = np.sum([mean_tpr, tpr], axis=0)
    #print mean_fpr, mean_tpr
    #roc_auc = auc(fpr,tpr)
    #print "fpr,tpr,threshold", fpr,tpr#, roc_auc
    ##mean_tpr +=interp(mean_fpr,fpr,tpr)
    ##mean_tpr[0] = 0.0

    fpr.sort()

    tpr.sort()
    roc_auc = auc(fpr, tpr)

    if plot_others == True:

        if t ==0:
            plt.plot(fpr, tpr, 'bo',lw=1, label="ROC{} (area = {})".format(i,roc_auc))  # ,'bo'
            plt.plot(fpr, tpr, color='blue',lw=1)
        elif t == 1:
            plt.plot(fpr, tpr, 'ro', lw=1, label="ROC{} (area = {})".format(i,roc_auc))  # ,'bo'
            plt.plot(fpr, tpr, color='red',lw=1)
        else:

            plt.plot(fpr, tpr, 'co', lw=1, label="ROC{} (area = {})".format(i,roc_auc))  # ,'bo'
            plt.plot(fpr, tpr, color='cyan', lw=1)

        #plt.plot(fpr, tpr,'bo', lw=1, label="ROC")  # ,'bo'
        t+=1

plt.plot([0, 1], [0, 1], linestyle='--', color='green', label='random guessing')

mean_tpr = np.divide(mean_tpr,float(n))
mean_fpr = np.divide(mean_fpr,float(n))
#mean_tpr[-1] = 1.0
mean_fpr.sort()
mean_tpr.sort()


print "lenght of tpr ", len(mean_tpr)
plt.plot(mean_fpr,mean_tpr,'k--',lw = 3,label= "mean ROC (area = {})".format(mean_auc))
#plt.plot(mean_fpr,mean_tpr,'ko',lw = 2,label= "mean ROC")


plt.xlabel('false positives')
plt.ylabel(('true positives'))
#plt.plot([0,0,1],[0,1,1],lw = 2,linestyle = '--',color = 'green',label = 'perfect performance')
# #
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
#(0.6,0.6,0.6)
plt.legend(loc = 'lower right')
plt.grid()
plt.show()

