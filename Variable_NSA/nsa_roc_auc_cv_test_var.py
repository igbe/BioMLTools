import myclassifiers_var_nsa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


def stratifiedKFold(test_data, test_class, no_folds=2):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    cv = StratifiedKFold(test_class, n_folds=no_folds,random_state=1)
    classs_test = test_class
    TP_out, TN_out, FP_out, FN_out, ACC_out, FPR_out, TPR_out= [],[],[],[],[],[],[]
    for i, (train, test) in enumerate(cv):
        print "########## Fitting the data #############"

        #k = mc.fit(test_data[train], classs_test[train]).predict(test_data[test])
        k = mc.fit(test_data[train], classs_test[train]).predict(test_data[test])
        #=classs_test[test]
        print k
        print "getting fpr and tpr for fold {}".format(i)
        fpr, tpr = mc.roc(test_data[test], k)

        #Uncomment the line below to plot the ROC curve whose mean is use to plot the mean ROC
        #plt.plot(fpr, tpr, '--', lw=1, label="ROC{}".format(i))


        #print classs_test[test], "size" , np.shape(classs_test[test])
        #print k, "size", np.shape(k)

        # To find mmetrics
        [TN, FP], [FN, TP] = confusion_matrix(classs_test[test], k)
        print "[TN, FP], [FN, TP]",[TN, FP], [FN, TP]

        TP_out.append(TP)
        TN_out.append(TN)
        FP_out.append(FP)
        FN_out.append(FN)

        ACC_out.append ((TP + TN )/ float(FP + FN + TP + TN))
        FPR_out.append(FP / float(FP + TN))
        TPR = TP /float (FN + TP)
        TPR_out.append(TPR)

        #Uncomment to prove that before and after sorting plotis the same
        #plt.plot(fpr, tpr, 'ro', lw=1, label="ROC{}".format(i))
        mean_tpr +=interp(mean_fpr,fpr,tpr)
        mean_tpr[0] = 0.0
    #print "TP_out, TN_out, FP_out, FN_out, ACC_out, FPR_out, TPR_out, PRE_out,F1_out", \
   #     TP_out, TN_out, FP_out, FN_out, ACC_out, FPR_out, TPR_out, PRE_out, F1_out
    return mean_fpr,mean_tpr,cv,TP_out, TN_out, FP_out, FN_out, ACC_out, FPR_out, TPR_out



if __name__ == '__main__':

    my_data = np.genfromtxt('danger_test.csv', delimiter=',')   #'iris1.csv', danger_test.csv
    #print my_data
    classs_test = my_data[:, 2]  # my_data[241:,6]#201:,6
    test_data = my_data[:, :2]  # my_data[241:,1:6] or 201 #201:,1:6
   # print classs_test

    fig = plt.figure(figsize=(7, 5))

    le = LabelEncoder()

    y = le.fit_transform(classs_test)
    #print y
    classs_test = y
    unique = np.unique(y)

    mc = myclassifiers_var_nsa.NsaConstantDetectorClassifier(number_of_detectors=500, self_radius_size=0.013,
                                                     random_gen_param=(0.01, 1.0), class_label=(unique[0], unique[1]))
    # print mc
    mean_tpr1 = 0.0
    mean_fpr1 = np.linspace(0, 1, 100)


    n = 1   #number of times to repeat k fold. If set to one, it becomes normal k-fold
    k = 2   #number of folds

    TP, TN, FP, FN =[],[],[],[]
    ACC_out1, FPR_out1, TPR_out1 = [],[],[]
    #using the technique of n times k-fold. Its ok to use k=n=10
    for j in range(n):
        print "repeating for n = {}".format(j)
        mean_fpr, mean_tpr,cv,TP_out, TN_out, FP_out, FN_out, ACC_out, FPR_out, TPR_out = stratifiedKFold(test_data, classs_test, no_folds=k)

        #update the list for mean metric
        TP += TP_out
        TN += TN_out
        FP += FP_out
        FN += FN_out
        ACC_out1 += ACC_out
        FPR_out1 += FPR_out
        TPR_out1 += TPR_out

        mean_tpr1 += interp(mean_fpr1, mean_fpr, mean_tpr)
        mean_tpr1[0] = 0.0

    #print "ACC_out, FPR_out, TPR_out, PRE_out,F1_out",TP, TN, FP, FN,ACC_out1, FPR_out1, TPR_out1, PRE_out1, F1_out1

    print "mean ACC",np.mean(ACC_out1)
    print "mean ERR", 1 - (np.mean(ACC_out1))
    print "mean FPR",np.mean(FPR_out1)
    print "mean TPR",np.mean(TPR_out1)
    print "mean FNR",1 - np.mean(TPR_out1)
    print "mean TNR",(np.mean(TN))/(np.mean(FP) + np.mean(TN))


    #print mean_tpr
    #print mean_fpr

    #To see what the mean tpr and fpr looks like, uncomment the line below
    #print mean_tpr1
    #print mean_fpr1

    #to plot the ROC curve, unomment the line below
    plt.plot([0, 1], [0, 1], linestyle='--', color='green', label='random guessing')

    #plot the error
    #print [1-i for i in ACC_out1]
    #plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],[1-i for i in ACC_out1])





    mean_tpr1 /=k * n
    mean_tpr1[1] = 1.0
    #mean_tpr[-1] = 1.0
    mean_fpr1.sort()
    mean_tpr1.sort()


    mean_auc = auc(mean_fpr1, mean_tpr1)

    #plt.plot(mean_fpr,mean_tpr,'k--',lw = 3,label= "mean ROC")

    #to plot binary ROC uncomment below
    #fp = [0] + [np.mean(FPR_out1)] + [1]
    #tp = [0] + [np.mean(TPR_out1)] + [1]
    #plt.plot(fp,tp, 'b-', lw=1, label="mean ROC (area = {})".format(mean_auc))

    plt.plot(mean_fpr1,mean_tpr1,'b-',lw = 1,label= "mean ROC (area = {})".format(mean_auc))


    plt.xlabel('false positives')
    plt.ylabel(('true positives'))
    # #
    #plt.xlim([-0.05,1.05])
    #plt.ylim([-0.05,1.05])
    #plt.xlim([0,1.05])
    #plt.ylim([0,1.05])
    #(0.6,0.6,0.6)
    plt.legend(loc = 'lower right')
    plt.grid()
    plt.show()




