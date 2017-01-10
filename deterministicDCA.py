import numpy as np
import random
#import DCA as dca
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,roc_curve, auc

def get_data(filename, datatype='float', delimiter=',', skiprows=1, ndmin=2):
    data = np.loadtxt(
        fname=filename,
        dtype='float',#np.float64
        delimiter=',',
        skiprows=1,
        ndmin=2)
    return data

def convert_class(y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    cl = []
    for i in y:
        if i == 0:
            cl.append(-1)
        else:
            cl.append(1)
    return cl

def get_multiplied_antigen(data, multiplier=10):
    data2 = []
    for i in range(len(data)):
        for j in range(multiplier):
            data2.append(data[i])
    # print data2
    data2 = np.array(data2)
    # print data2
    return data2

class deterministicDCA:
    def __init__(self,anomaly_threshold,num_of_dc_selected = 100,lifespan = 1000,class_label = (-1,1),threshold =(5, 15)):
        self.threshold = threshold
        self.lifespan = lifespan
        self.class_label = class_label
        self.num_of_dc_selected = num_of_dc_selected
        self.anomaly_threshold = anomaly_threshold
        #pass

    def initialize_dDCA(self,num_of_cells):
        """
        num_of_cells is the number of DC you want to create, and threshold is a list with [min,max] threshold
        DC structure is given by:
        [[antigen],(signal),[csm,k],lifespan,location(int),no_cell_iter,threshold,context(int)]'
        where [signal] is the DC signal matrix, [antigen] the DC antigen matrix, [output_matrix] output signal matrix i.e csm and k, lifespan is the age
        of the dDCA cell, and location(int) is the location in cells where this dc is located, threshold, and context(int) is the context

        :return: a list of cells in format [[antigen],(signal),[csm,k],lifespan,location(int),no_cell_iter,threshold,context(int)]
        """
        DC = []
        for i in range(num_of_cells):
            if isinstance(self.threshold, tuple):
                tm = self.threshold[0] + (self.threshold[1] - self.threshold[0]) * random.random()
            else:
                tm = self.threshold
            dc = [[], (), [0, 0], self.lifespan, 0, 0, tm, '']
            DC.append(dc)
        return DC

    def select_dDCA(self,dc_population, num_dc_to_select):
        # print 'dc_population',dc_population
        location = random.sample(range(len(dc_population)), num_dc_to_select)
        selected_dc = [dc_population[i] for i in location]
        # print selected_dc
        k = 0
        for i in location:
            selected_dc[k][4] = i
            k += 1
        return selected_dc, location

    def fit(self,new_data,safe,danger):
        self.migrated_DCs = []
        antigenCounter = 0

        cells = self.initialize_dDCA(self.num_of_dc_selected)
        #print cells
        selected_DCs, locations = self.select_dDCA(cells, self.num_of_dc_selected)
        #print locations
        #print selected_DCs


        for i in range(len(new_data)):  # len(new_data)
            sampled_antigen = new_data[i, 0]
            cell_index = antigenCounter % self.num_of_dc_selected
            selected_DCs[cell_index][0].append(sampled_antigen)  # store antigen
            antigenCounter = antigenCounter + 1  # count the number of antigens and use it to get the cell index to assign it to.
            csm = safe[i] + danger[i]
            k = danger[i] - (2 * safe[i])

            # print selected_DCs,'------>', csm, k
            for j in range(self.num_of_dc_selected):
                # print selected_DCs
                # update signals
                selected_DCs[j][1] = (safe[i], danger[i])
                # for reducing lifespan
                selected_DCs[j][3] -= csm
                # the outpts csm and k
                selected_DCs[j][2][0] += csm
                selected_DCs[j][2][1] += k
                # print selected_DCs
                # if lifespan is less than zero, re-initialize the cell
                if (selected_DCs[j][3] <= 0):
                    selected_DCs[j] = self.initialize_dDCA(1)[0]
                elif (selected_DCs[j][2][0] >= selected_DCs[j][6]) and (selected_DCs[j][0] != []):
                    # to add a context k <0 is givn a normal context and vis versa
                    if selected_DCs[j][2][1] < 0:
                        selected_DCs[j][7] = 'normal'
                    else:
                        selected_DCs[j][7] = 'anormaly'

                        self.migrated_DCs.append(selected_DCs[j])
                    selected_DCs[j] = self.initialize_dDCA(1)[0]

        #print self.migrated_DCs
        return self

    def get_mcav(self,antigen, migrated_DCs):
        all_mcav = []


        for i in antigen:
            num_cells, num_antigen=0,0
            for l in migrated_DCs:
                if (i in l[0]) and (l[7] !='normal'):
                    num_antigen = num_antigen + l[0].count(i)
                    #if :
                    num_cells +=1

            try:
                print "num_cells",num_cells,"num_antigen",num_antigen,mcav
                mcav = float(num_cells)/(num_antigen)
            except:
                #print "in mcav==0"
                mcav = 0
            all_mcav.append((i, mcav))

        return all_mcav

    def predict(self,antigen):
        pred =[]
        all_mcav = self.get_mcav(antigen, self.migrated_DCs)
        self.all_mcav = np.array(all_mcav)
        #print all_mcav

        for i,mcav in all_mcav:
            if mcav > self.anomaly_threshold:
                pred.append(-1)
            else:
                pred.append(1)
        return np.array(pred)

    def roc(self,true_label):
        print true_label
        print list(self.all_mcav[:,1])
        fpr, tpr, thresholds = roc_curve(true_label, self.all_mcav[:,1], pos_label=self.class_label[1])
        print fpr
        print tpr
        print thresholds
        #plt.plot(fpr, tpr, 'b-', lw=1)
        #plt.show()

#########################################################
# TEST
#########################################################
if __name__ == '__main__':

    #dca1 = dca.DCA(dca_pop_size=0,is_it_static=True)
    data = get_data('breast_cancer_test.csv', datatype=np.float64, delimiter=',', skiprows=1, ndmin=2)
    #print 'data length, ',len(data)
    new_data =get_multiplied_antigen(data, multiplier=10)#data #dca.get_multiplied_antigen(data, multiplier=2)
    #print 'new_data length, ',len(new_data)
    #print new_data
    anomaly_threshold = (460.0/700)
    num_of_dc_selected = 100
    antigen = new_data[:, 0]
    data_class =new_data[:, 6]
    safe = new_data[:, 9]
    pamp = new_data[:, 7]
    danger = new_data[:, 8]

    label = convert_class(data_class)
    dc = deterministicDCA(anomaly_threshold,num_of_dc_selected = 100,lifespan = 1000,threshold =(5, 15))
    pred = dc.fit(new_data, safe, danger).predict(antigen)
    dc.roc(label)
