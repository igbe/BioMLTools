# main.py

from nsa_genetics import *
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# warnings.filterwarnings("ignore", category=RuntimeWarning)
def confusion_matrix1(y_true, y_pred,classlabel=(-1,1)):        #where 1 is te normal label, -1 is te abnormal label
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

def get_data(filename, datatype='float', delimiter='	', skiprows=1, ndmin=2):
    data = np.loadtxt(fname=filename, dtype='float', delimiter=',', skiprows=1, ndmin=2)  # np.float64

    return data[:, 1:]


def main():

    alpha = 1
    beta = 100
    prob_of_m = 0.9  # probability of NO mutation
    prob_of_cross = 1
    total_gen = 200  # number of generations
    data = get_data('breastcancernorm.csv')
    train_data =data[:400,0:4]
    train_class = data[:400,5]

    test_data = data[400:600, 0:4]
    test_class = data[400:600, 5]

    #print train_data, train_class
    a = NsaGenetics(alpha,beta,prob_of_m,prob_of_cross,total_gen,number_of_detectors = 1000, self_radius_size = 0.1,
                    random_gen_param = (0.001,1.0),class_label = (-1,1),otherParam=None)
    a.fit(train_data,train_class)
    k = a.predict(train_data)
    print train_class
    print k

    fpr, tpr = a.roc(train_data, train_class)

    [TN, FP], [FN, TP] = confusion_matrix1(train_class, k)

    print "[TN, FP], [FN, TP]", [TN, FP], [FN, TP]#, "fpr, tpr",fpr, tpr
    plt.plot(fpr,tpr,color = 'r')
    plt.show()
    #             prob_of_cross, total_gen)

    # a.findFitness(a.new_pop)
    # a.new_gen(a.new_pop,50)


    #a.run_generations(a.total_gen)


main()