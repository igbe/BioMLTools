from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score,roc_curve, auc
from sklearn import preprocessing
from sklearn.neighbors import KDTree
import math
from pprint import pprint
import matplotlib.pyplot as plt
import itertools


class NsaGenetics(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha, beta, prob_of_m, prob_of_cross, total_gen, number_of_detectors=1000,
                 self_radius_size=0.16, random_gen_param=(0.0001, 1.0),
                 class_label=(0, 1), otherParam=None):
        self.number_of_detectors = number_of_detectors
        self.self_radius_size = self_radius_size
        self.alpha = alpha
        self.beta = beta
        self.prob_of_m = prob_of_m
        self.prob_of_cross = prob_of_cross
        self.total_gen = total_gen
        self.class_label = class_label
        self.random_gen_param = random_gen_param

    #def findFitness(self,ch2):
    #    return random.randint(0, 100)

    def findFitness(self, pop):
        # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html

        if (sum(1 for x in pop if isinstance(x, list))) > 1:
            # D1
            tree = KDTree(pop, leaf_size=30, metric='euclidean')
            D1 = []  # distance to nearest neighbor in pop itself
            for ch in range(len(pop)):
                X = pop[ch]
                dist, ind = tree.query(X, k=2, return_distance=True)
                D1.append(dist[0][
                              1])  # now we have a list of the distance of each chromosome to closest neighbor in same population

            # D
            tree = KDTree(self.self_pop, leaf_size=30, metric='euclidean')
            D = []
            for ch in range(len(pop)):
                X = pop[ch]
                dist, ind = tree.query(X, k=2, return_distance=True)
                D.append(dist[0][
                             1])  # now we have a list of the distance of each chromosome to closest neighbor in self population

            alpha_term = []
            beta_term = []
            for bbb, ccc in zip(D, D1):
                if (bbb != 0 and ccc != 0):
                    alpha_term.append(self.alpha * math.exp(-self.self_radius_size / bbb))
                    beta_term.append(self.beta * math.exp(-self.self_radius_size / ccc))
                elif (bbb == 0 and ccc == 0):
                    alpha_term.append(0)
                    beta_term.append(0)
                elif (bbb == 0 and ccc != 0):
                    alpha_term.append(0)
                    beta_term.append(self.beta * math.exp(-self.self_radius_size / ccc))
                else:
                    alpha_term.append(self.alpha * math.exp(-self.self_radius_size / bbb))
                    beta_term.append(0)

            fitness_list = []
            for i in range(len(alpha_term)):
                fitness_list.append(alpha_term[i] + beta_term[i])

            fitness = np.mean(fitness_list)
            return fitness

        else:  # fitness of single chromosome, beta term is 0. Some of this code is unneeded, cleanup later
            # D
            tree = KDTree(self.self_pop, leaf_size=30, metric='euclidean')
            D = []
            X = pop
            dist, ind = tree.query(X, k=2, return_distance=True)
            D.append(dist[0][1])

            alpha_term = beta_term = []
            for bbb in D:
                if bbb != 0:
                    alpha_term.append(self.alpha * math.exp(-self.self_radius_size / bbb))
                else:
                    alpha_term.append(0)

            fitness_list = []
            for i in range(len(alpha_term)):
                fitness_list.append(alpha_term[i])

            fitness = np.mean(fitness_list)
            return fitness

    def cross_genes(self, gene1, gene2):
        if len(gene1) == 2:  # genes are same length
            new_ch1 = [gene1[0], gene2[1]]
            new_ch2 = [gene2[0], gene1[1]]
            new_gene = [new_ch1, new_ch2]
            return new_gene
        else:  # general case, breaks for n < 3 which is why it branches
            slice_index = int(math.floor(len(gene1) / 2))

            new_ch1 = [gene1[0:slice_index], gene2[slice_index:]]
            new_ch2 = [gene2[0:slice_index], gene1[slice_index:]]

            new_ch1 = list(itertools.chain.from_iterable(new_ch1))  # remove nested lists
            new_ch2 = list(itertools.chain.from_iterable(new_ch2))
            new_gene = [new_ch1, new_ch2]
            return new_gene

    def mutate(self, ch, gen_num):  # mutate a gene. assumes 2D array right now, have to fix
        num = random.random()
        m_num = random.random()  # probability of which gene to mutate:

        if m_num > 0.5:
            m_gene = 0
        else:
            m_gene = 1
        if num > 0.5:  # add
            m_function = (1 - ch[m_gene]) * (1 - random.random() ** ((1 - gen_num / self.total_gen) ** 5))
            ch[m_gene] = ch[m_gene] + m_function
        else:  # subtract
            m_function = ch[m_gene] * (1 - random.random() ** ((1 - gen_num / self.total_gen) ** 5))
            ch[m_gene] = ch[m_gene] - m_function

        return ch
    def new_gen(self, pop, gen_num):  # keep track of what generation we are in for mutation function
        next_pop = []
        for aaa in range(len(pop) / 2):
            cross = []
            # we need a set of two chromosomes, add each set to cross[]
            for i in range(2):
                # pick a set of 2 random chromosomes
                num = random.randint(0, len(pop) - 1)
                ch1 = pop[num]
                num = random.randint(0, len(pop) - 1)
                ch2 = pop[num]

                if self.findFitness(ch1) > self.findFitness(ch2):
                    cross.append(ch1)
                else:
                    cross.append(ch2)

            # probablity of cross over
            num = random.random()
            if num < self.prob_of_cross:  # crossover
                ch_pair = self.cross_genes(cross[0], cross[1])
            else:  # don't cross over
                ch_pair = cross

            # add floating point mutation
            # http://umsl.edu/divisions/artscience/math_cs/about/People/Faculty/CezaryJanikow/folder%20two/Experimental.pdf
            num = random.random()  # crossover and mutation are independent
            if num > self.prob_of_m:
                ch_to_mutate = random.random()
                if ch_to_mutate < 0.5:
                    ch_pair[0] = self.mutate(ch_pair[0], gen_num)
                else:
                    ch_pair[1] = self.mutate(ch_pair[1], gen_num)
            else:
                pass

            # pass new genes into new population
            next_pop.append(ch_pair[0])
            next_pop.append(ch_pair[1])

        # pass on population with higher fitness
        f_new = self.findFitness(next_pop)
        #print "old", self.f_old,"new",f_new
        if self.f_old >= f_new:
            #print "OLD IS BETTER", self.f_old
            return pop, self.f_old
        else:
           # print "BETTER FOUND", f_new
            self.f_old = f_new
            #print "OLD", f_old, "NEW", f_new
            return next_pop, f_new


    def fit(self, X, y):

        """

		:param X: this is a numpy array of the self_set without the class_label
		:param y: array-like, shape= [n_samples]. this is the vector of target class labels
		:return: it returns a self object
		"""
        # print "self.number_of_detectors ", self.number_of_detectors
        # To check and ensure that the parameter types correspond to what is needed or expected, we run a type check

        if type(self.number_of_detectors) != int:
            raise Exception("number_of_detectors must be an integer value")
        elif type(self.self_radius_size) != float:
            raise Exception('self_radius_size must be an floating pont value')
        elif type(self.random_gen_param) != tuple:
            raise Exception('random_gen_param must be a tuple value')
        elif type(X) != np.ndarray:
            raise Exception('self list X must be a numpy array')
        elif type(y) != np.ndarray:
            raise Exception('class label y must be a numpy array')

        normal = []
        # attack = []
        j = 0
        for i in X:
            # print "X = ", i, y[j], self.class_label[0]
            if y[j] == self.class_label[0]:
                normal.append(i)
            # else:
            #    attack.append(i)
            j += 1

        len_normal_ = len(normal)

        self.self_pop = normal
        # len_attack_ = len(attack)
        X = np.array(normal)
        # attack = np.array(attack)

        # To find the number of columns in the self_set matrix whic will help use generate same dimension of random detectors
        # X = np.array(X)
        num_of_features = np.shape(X)[1]

        # print "dimension", dimension
        min = self.random_gen_param[0]
        max = self.random_gen_param[1]
        population = []
        for i in range (self.number_of_detectors):
            population.append([10 ** random.uniform(np.log10(min), np.log10(max)) for i in range(num_of_features)])
        population = np.array(population)

        #print population
        avg_fitness = []
        for i in range(self.total_gen):
            if i % 10 == 0:
                print '%d%%' % (i)
            if i ==0:
                self.f_old = self.findFitness(population)
                self.pop_ = population
            pop,fitness = self.new_gen(self.pop_, i)        # iterate for the number of generations
            self.pop_ = pop
            avg_fitness.append(fitness)
            #norm = np.array(normal)
            #pp = np.array(self.pop)
            #plt.scatter(norm[:, 0], norm[:, 1], color='red')
            #plt.scatter(pp[:, 0], pp[:, 1], color='blue')
            #plt.grid()
            #plt.show()

        norm = np.array(normal)
        pp = np.array(self.pop_)
        plt.scatter(norm[:,0],norm[:,1],color='red')
        plt.scatter(pp[:, 0], pp[:, 1], color='blue')
        plt.grid()
        plt.show()

        #print "average fitness", avg_fitness
        x,yy=[],[]
        for i in range(len(avg_fitness)):
            x.append(i)
            yy.append(avg_fitness[i])

        plt.plot(x, yy)
        plt.show()

    def predict(self, X):

        """

        :param X: this is an array of the data to be predicted. note if it has only 1 element, then use [element]
            instead of just passing element.
        :return: a list containing the predicted class labels for the provided data.
        """

        # check is fit had been called
        #check_is_fitted(self, 'self.pop_')

        # print "self.self_radius_size", self.self_radius_size

        # input validation. This will display error message if there is no input
        # X = check_array(X)

        # array to hold the predicted label
        predicted_class = []

        kdt = KDTree(self.pop_, leaf_size=30, metric='euclidean')
        self.distanceValues_ = []
        for i in range(len(X)):
            dist, ind = kdt.query([X[i]], k=1)
            #            print 'dist =', dist
            self.distanceValues_.append(dist[0])

            if dist >= self.self_radius_size:
                predicted_class.append(self.class_label[0])
            else:
                predicted_class.append(self.class_label[1])
        self.distanceValues_ = np.array(self.distanceValues_)

        return predicted_class

    def score(self, X, y, sample_weight=None):
        # print X
        pred_score_ = self.predict(X)
        # print pred_score_
        # print accuracy_score(y_true=y,y_pred=pred_score_)
        return accuracy_score(y_true=y, y_pred=pred_score_)

    def roc(self, test_data, ground_truth):
        """

        :param test_data: we need this again even though it has been predicted before because we will have to iterate
            over it for new values of self_radius
        :param ground_truth: the real truth
        :return: fpr, tpr both are numpy arrays
        """

        option = self.distanceValues_
        scores = [option[i][0] for i in range(len(option))]
        # used self.class_label[0] which should be the normal class instead of the abnormal class of self.class_label[1]
        # because the sckitlearn roc curve being used inverts the graph if correct self.class_label[1] is used.
        fpr, tpr, thresholds = roc_curve(ground_truth, scores, pos_label=self.class_label[0])
        # print "fprt, tprt", fpr, tpr
        # print thresholds
        # print auc(fpr,tpr)
        # plt.show()
        return fpr, tpr
