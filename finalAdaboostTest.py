import myclassifiers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score,roc_curve, auc
from scipy import interp
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from myclassifiers_test import testClassifier
import myAdaboost


def confusion_matrix1(y_true, y_pred,classlabel=(1,-1)):        #where 1 is te normal label, -1 is te abnormal label
    TN, FP, FN, TP = 0, 0, 0, 0

    for y1, y2 in zip(y_true, y_pred):

        # print y1,y2
        if y1 == classlabel[0]:  # normal
            if y2 == y1:
                TN += 1
            else:
                FP += 1
        else:
            if y2 == y1:
                TP += 1
            else:
                FN += 1
    return [TN, FP], [FN, TP]


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


def stratifiedKFold(test_data, test_class,no_of_weaklearners = 2,rounds = 2, no_folds=2):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    cv = StratifiedKFold(test_class, n_folds=no_folds,random_state=1)
    classs_test = test_class


    TP_out, TN_out, FP_out, FN_out, ACC_out, FPR_out, TPR_out, PRE_out,F1_out = [],[],[],[],[],[],[],[],[]
    for i, (train, test) in enumerate(cv):
        print "########## Fitting the data #############"

        no_of_weakLearners = no_of_weaklearners

        weakLearners = []

        for c in range(no_of_weakLearners):
            # weakLearner.append((testClassifier,tresh[i]))
            weakLearners.append(myclassifiers.NsaConstantDetectorClassifier(number_of_detectors=500, self_radius_size=0.1,
                                                            random_gen_param=(0.001, 1.0), class_label=(1, -1)))


        Ada = myAdaboost.AdaBoost(weakLearners,rounds,adatype=2,weightedSampling=True,classlabel=(1,-1))

        p = Ada.fit(test_data[train], classs_test[train]).predict(test_data[test])
        #k = mc.fit(test_data[train], classs_test[train]).predict(test_data[test])
        print "getting fpr and tpr for fold {}".format(i)
        fpr, tpr = Ada.roc(classs_test[test],p)
        #print mc.predict(test_data[test])
        #print classs_test[test]
        #print mc.score(test_data[test],classs_test[test])
        print  "fpr,tpr",fpr,tpr
        #print tpr

        #Uncomment the line below to plot the ROC curve whose mean is use to plot the mean ROC
        plt.plot(fpr, tpr, 'o', lw=1, label="ROC{}".format(i))


        #print classs_test[test], "size" , np.shape(classs_test[test])
        #print k, "size", np.shape(k)
        #print confusion_matrix(classs_test[test], k)
        # To find mmetrics
        [TN, FP], [FN, TP] = confusion_matrix1(classs_test[test], p,classlabel=(1,-1))#confusion_matrix(classs_test[test], k)
        #print "[TN, FP], [FN, TP]",[TN, FP], [FN, TP]

        TP_out.append(TP)
        TN_out.append(TN)
        FP_out.append(FP)
        FN_out.append(FN)

        ACC_out.append ((TP + TN )/ float(FP + FN + TP + TN))
        FPR_out.append(FP / float(FP + TN))

        #TPR = TP /float (FN + TP)

        if FN + TP == 0:
            TPR = 0.0
        else:
            TPR = TP / float(TP + FP)

        TPR_out.append(TPR)

        if TP + FP == 0:
            PRE = 0.0
        else:
            PRE = TP / float(TP + FP)

        PRE_out.append(PRE)
        #print "PRE + TPR", PRE + TPR

        if PRE + TPR == 0.0:
            F1_out.append(0.0)
        else:
            F1_out.append( 2 * ((PRE * TPR)/float(PRE + TPR)))

        #Sort the data else, mean interpolation wont work. Note at the time of writing this,
        #ploting fpr and tpr of pre-sort and post-sort gave same curve
        fpr.sort()
        tpr.sort()
        #print  fpr
        #print tpr

        #Uncomment to prove that before and after sorting plotis the same
        #plt.plot(fpr, tpr, 'ro', lw=1, label="ROC{}".format(i))
        mean_tpr +=interp(mean_fpr,fpr,tpr)
        mean_tpr[0] = 0.0
    #print "TP_out, TN_out, FP_out, FN_out, ACC_out, FPR_out, TPR_out, PRE_out,F1_out", \
   #     TP_out, TN_out, FP_out, FN_out, ACC_out, FPR_out, TPR_out, PRE_out, F1_out
    return mean_fpr,mean_tpr,cv,TP_out, TN_out, FP_out, FN_out, ACC_out, FPR_out, TPR_out, PRE_out, F1_out



if __name__ == '__main__':

    my_data = np.genfromtxt('breast_cancer_test_norm.csv', delimiter=',')
    normal = my_data[1:201, 1:6]  # my_data[1:201,1:6]
    classs_normal = my_data[1:201, 6]
    classs_test = my_data[1:400, 6]  # my_data[241:,6]#201:,6
    test_data = my_data[1:400, 1:6]  # my_data[241:,1:6] or 201 #201:,1:6

    fig = plt.figure(figsize=(7, 5))

    classs_test = np.array(convert_class(classs_test))

    #train = test_data



    #mc = myclassifiers.NsaConstantDetectorClassifier(number_of_detectors=2000, self_radius_size=0.1,
    #                                                 random_gen_param=(0.001, 1.0), class_label=(unique[0], unique[1]))
    # print mc
    mean_tpr1 = 0.0
    mean_fpr1 = np.linspace(0, 1, 100)


    n = 1   #number of times to repeat k fold. If set to one, it becomes normal k-fold
    k = 2   #number of folds

    TP, TN, FP, FN =[],[],[],[]
    ACC_out1, FPR_out1, TPR_out1, PRE_out1, F1_out1 = [],[],[],[],[]
    #using the technique of n times k-fold. Its ok to use k=n=10
    for j in range(n):
        print "repeating for n = {}".format(j)
        mean_fpr, mean_tpr,cv,TP_out, TN_out, FP_out, FN_out, ACC_out, FPR_out, TPR_out, PRE_out, F1_out= stratifiedKFold(test_data,
                                                                                                                          classs_test,
                                                                                                                          no_of_weaklearners = 50,
                                                                                                                          rounds=20,
                                                                                                                          no_folds=k)

        #update the list for mean metric
        TP += TP_out
        TN += TN_out
        FP += FP_out
        FN += FN_out
        ACC_out1 += ACC_out
        FPR_out1 += FPR_out
        TPR_out1 += TPR_out
        PRE_out1 += PRE_out
        F1_out1  += F1_out

        mean_tpr1 += interp(mean_fpr1, mean_fpr, mean_tpr)
        mean_tpr1[0] = 0.0

    #print "ACC_out, FPR_out, TPR_out, PRE_out,F1_out",TP, TN, FP, FN,ACC_out1, FPR_out1, TPR_out1, PRE_out1, F1_out1

    print "mean ACC",np.mean(ACC_out1)
    print "mean ERR", 1 - (np.mean(ACC_out1))
    print "mean FPR",np.mean(FPR_out1)
    print "mean TPR",np.mean(TPR_out1)
    print "mean FNR",1 - np.mean(TPR_out1)
    print "mean TNR",(np.mean(TN))/(np.mean(FP) + np.mean(TN))
    print "mean PRE",np.mean(PRE_out1)
    print "mean F1",np.mean(F1_out1)

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
    plt.plot(mean_fpr1,mean_tpr1,'k--',lw = 3,label= "mean ROC (area = {})".format(mean_auc))


    plt.xlabel('false positives')
    plt.ylabel(('true positives'))
    # #
    #plt.xlim([-0.05,1.05])
    #plt.ylim([-0.05,1.05])
    plt.xlim([0,1.05])
    plt.ylim([0,1.05])
    #(0.6,0.6,0.6)
    plt.legend(loc = 'lower right')
    plt.grid()
    plt.show()




