import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

def var_classifier(test_data,detectors,self_radius_size,classs,normal=1,attack=2):
    TP=0
    TN=0
    FP=0
    FN=0
    kdt = KDTree(detectors, leaf_size=30, metric='euclidean')

    for i in range(len(test_data)):
        dist, ind = kdt.query([test_data[i]], k=1)
        print 'dist, ind',dist, ind
        #print '[j]: ', j
        #print 'class[j]',classs[j]
        if dist < self_radius_size:
            print 'classs[i]', classs[i], 'dist',dist,'<',self_radius_size
            if classs[i] == attack:
                print 'non-self'
                TP +=1
            else:
                FP +=1
        else:
            print 'classs[i]', classs[i], 'dist',dist,'>',self_radius_size
            if classs[i] == normal:
                print 'self'
                TN +=1
            else:
                FN +=1

    return  TP,TN,FP,FN

def nsa(Dmax,self_radius_size,normal,dimension=5):
    D=[]
    min=0.0001
    max=1.0
    kdt = KDTree(normal, leaf_size=30, metric='euclidean')
    while len(D) < Dmax:
        print len(D)
        x=[10 ** random.uniform(np.log10(min), np.log10(max)) for i in range(dimension)]#,10 ** random.uniform(np.log10(min), np.log10(max)),10 ** random.uniform(np.log10(min), np.log10(max)),10 ** random.uniform(np.log10(min), np.log10(max)),10 ** random.uniform(np.log10(min), np.log10(max))]#,random.randint(0,10),random.randint(0,10),random.randint(0,10)]
        #print 'x: ', x
        dist, ind = kdt.query([x], k=1)
        print 'first x distance: ',dist
#        if dist < self_radius_size:
#            continue
#        else:
#            D.append(x)
        if dist > self_radius_size:
            #print 'second d distance: ',dist
            if len(D) ==0:
                D.append(x)
            else:
                kdt = KDTree(D, leaf_size=30, metric='euclidean')
                dist2, ind = kdt.query([x], k=1)
                if dist2 > self_radius_size:
                    D.append(x)
    return D

if __name__ == '__main__':
    Dmax = 1000
    self_radius_size=0.16#0.095
    dimension = 5
    save_detector = 'no'
    normal_class=1
    attack_class=2
    plot =True#False#True#False
    my_data = np.genfromtxt('breast_cancer_test_norm.csv', delimiter=',')
    normal = my_data[1:201,1:6]#my_data[1:201,1:6]
    classs=my_data[201:,6]#my_data[241:,6]#201:,6
    test_data=my_data[201:,1:6]#my_data[241:,1:6] or 201 #201:,1:6
    #print normal
    D=nsa(Dmax,self_radius_size,normal,dimension)
    TP,TN,FP,FN = var_classifier(test_data,D,self_radius_size,classs,normal_class,attack_class)
    print 'TP',TP,'TN',TN,'FP',FP,'FN',FN
    if save_detector == 'yes':
        f=open('detector_list','w')
        f.write('{0},{1},{2},{3}'.format(D,TP,TN,FP,FN))
        f.close()
    if plot==True:
        D = np.array(D)
        plt.scatter(D[:,0],D[:,2],color='red')
        plt.scatter(normal[:,0],normal[:,1],color='green')
        plt.scatter( test_data[:,0], test_data[:,1],color='blue')
        plt.show()






