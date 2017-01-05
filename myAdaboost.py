from multiprocessing.dummy import Pool as ThreadPool
import random
import math
from numpy.random import choice
from sklearn.metrics import accuracy_score,roc_curve, auc
import numpy as np
from sklearn.metrics import confusion_matrix

class AdaBoost:
    def __init__(self,weakLearners,rounds=2,adatype=1,weightedSampling=True,classlabel=(1,-1),otherParam=None):

        """
        :param weakLearners: this is a list containing the instance of all the weak learners to bbe used in adaboost
        :param rounds: this is the number of adaboost rounds intended
        :param adatype: - this the adaboost variation intended. For value of 1, the adaboost will find the weight distribution
                          then it will train the first weak classifier, and the weight update will assign more weight to
                          misclassified examples, and this new weights are used to select samples from the training set,
                          and used to train the next weak classifier.
                        - For type 2, the initial weights are normalized, and all classifiers are trained, then the one
                          with the lowest error is selected as the classifier for the rest of the computation. After
                          weight normalization, the next classifier with the smallest error is selected again and so on.
        :param weightedSampling: this is a boolean that decides wether you will use the weights to pick samples or if you will
                                 continue training with the entire training set.
        :param classlabel: This is for specifying the class label =(normal,abnormal) that is normal and for abnormal
        """

        self.weakLearners = weakLearners
        self.rounds = rounds
        self.hypotheses = [None] * rounds
        self.alpha = [0] * rounds
        self.error1 = []
        self.weightedSampling = weightedSampling
        self.adatype = adatype
        self.classlabel = classlabel

        # note for compatibility with skitlearn, using a variable which is not in the supplied class parameters is not advised
        self.summed_values_ = []

        #pass


    def computeError(self, h, y, weights=None):
        """
        :param h: the hypothesis or class prediction made by a classifier
        :param y: the true class labels from the original training data
        :param weights: the weight or probability distribution
        :return: the hypothesisResult which is +1 if classifier is correct or -1 otherwise, then the error
        """
        hypothesisResults = []
        if weights is None:
            weights = [1.] * len(y)
        for i in range(len(y)):
            hypothesisResults.append(h[i]*y[i])     # +1 if correct, else -1
        return hypothesisResults, sum(w for (z,w) in zip(hypothesisResults, weights) if z < 0)

    # normalize a distribution


    def normalize(self, weights):
        """
        :param weights: the weights to be normalized to 1
        :return: a tupple containing all the new normalized weights
        """
        norm = sum(weights)
        return tuple(m / norm for m in weights)



    def sign(self, x):
        """
        :param x: float or int who's value we want to know if it is + or -
        :return: 1 for positive x or -1 for negative x
        """
        return 1 if x >= 0 else -1


    def weighted_sample(self, X,initial_weights):
        """
        This function returns sample data-instances from the training data based on a weighted distribution.
        The higher your weight, the more likely you will get selected.
        :param initial_weights:  the weights or probability distribution
        :return: the newly selected data samples
        """
        indices = range(len(X))     #get the indices
        #print indices
        my_draw = choice(indices, len(X), p=initial_weights)  #using numpy choice method that supports weights
        #print my_draw
        return X[my_draw]

    def fit(self,X,y):
        if self.adatype == 1:
            return self.fit1(X,y)
        else:
            return self.fit2(X,y)

    def fit1(self, X,y):
        """
        This is the main function of the adaboost algorithm. It calls other functions in this file.
        :param X: This is the training data to be use for training
        :param y: This is the class label for the training data
        :return: the final prediction
        """
        weight = self.normalize([1.] * len(X))     #the probability distribution or the weights assigned to each training set
        #print "weights",weight
        #print "len of weakLearner", len(self.weakLearners)

        tru_y = y
        #self.hypotheses = [None] * self.rounds
        #self.alpha = [0] * self.rounds
        #self.error1 = []

        #Iterate through the number of self.rounds
        for t in range(self.rounds):
            #print "weight", weight


            if t == 0:      #since for the first iteration, the sample is same with the training sample
                sample = X
            else:
                # use the new weights in other iterations to choose the next training sample
                # this tries to help your model understand new things about your data expecially the samples that
                # wasn't well classified in previous iteration. Obviously setting this to X as in t==0, makes this
                # algorithm behave like majority voting. But this last statement applies to deterministic algorithms.

                if self.weightedSampling == True:
                    sample = self.weighted_sample(X,weight)
                else:
                    #print "in false"
                    sample = X
            #print "sample", sample

            # Choose the number t weakLeaner from the list of learners
            weakLearner = self.weakLearners[t]


            # call the fit method of the weakLeaner to train the data (sample), then use its predict method to get the
            # predicted class labels for the original training data. Note
            self.hypotheses[t] = weakLearner.fit(sample, y).predict(X)
            #print "hypotheses[t]", self.hypotheses[t]

            #compute the error in the labels retruned by the above weakLeaner
            hypothesisResults, error = self.computeError(self.hypotheses[t], y, weight)
            #print "hypothesisResults, error", hypothesisResults, error
            #print "Error", error


            # To make sure you have a list of all the errors made by your classifier
            self.error1.append(error)

            # To append new values for alpha for each round/hypothese to the alpha list
            self.alpha[t] = 0.5 * math.log((1 - error) / (.0001 + error))
            #print "self.alpha", self.alpha[t]

            #Get the new normalized weights
            weight = self.normalize([d * math.exp(-self.alpha[t] * h)
                              for (d,h) in zip(weight, hypothesisResults)])
            print("Round %d, error %.3f, Accuracy %.3f, Alpha %.3f" % (t, error,accuracy_score(y_true=y, y_pred=self.hypotheses[t]),self.alpha[t]))

        return self

    def get_classifer_error_(self,job):
        X,sample,y,weakLearner,weight = job
        hypotheses_ = weakLearner.fit(sample, y).predict(X)

        hypothesisResults_, error_ = self.computeError(hypotheses_, y, weight)
        #print "classifier error ",hypothesisResults_, error_,hypotheses_,weakLearner
        return [hypothesisResults_, error_,hypotheses_,weakLearner]



    #def predict(self, X, y, tru_y):
    def get_best_classifier_(self,X,sample,y,weakLearners,weight):

        """
        :param X: the original training data
        :param sample: the sample training data
        :param y: the training class label
        :param weaklearners: a list of the weakleaners/weakclassifiers
        :return: classifier instance -of the classifier with lowest error
        """

        jobs=[]
        for i in weakLearners:
            jobs.append([X,sample,y,i,weight])

        # Make the Pool of workers
        pool = ThreadPool(13)
        # Open the urls in their own threads
        # and return the results
        results = pool.map(self.get_classifer_error_, jobs)
        #print "result", results

        final_hypothesisResults_ = None
        final_error_ = None
        final_weakLearner_ = None
        final_hypotheses_ = None

        t =0

        for hypothesisResults_, error_ ,hypotheses_,weakLearner_ in results:

            #print "result2", results
            #print "error_ ,weakLearner_  ",error_ ,weakLearner_

            if t==0:
                final_hypothesisResults_ = hypothesisResults_
                final_error_ = error_
                final_weakLearner_ = weakLearner_
                final_hypotheses_ = hypotheses_
            #get the lowest error
            if final_error_ > error_:
                final_hypothesisResults_ = hypothesisResults_
                final_error_ = error_
                final_weakLearner_= weakLearner_
                final_hypotheses_ = hypotheses_
                # close the pool and wait for the work to finish
            t=1

        pool.close()
        pool.join()
        #print "info",final_hypothesisResults_,final_error_,final_hypotheses_,final_weakLearner_

        return final_hypothesisResults_,final_error_,final_hypotheses_,final_weakLearner_


    def fit2(self, X,y):
        """
        This is the main function of the adaboost algorithm. It calls other functions in this file.
        :param X: This is the training data to be use for training
        :param y: This is the class label for the training data
        :return: the final prediction
        """
        weight = self.normalize([1.] * len(X))     #the probability distribution or the weights assigned to each training set
        #print "weights",weight
        #print "len of weakLearner", len(self.weakLearners)

        tru_y = y
        #self.hypotheses = [None] * self.rounds
        #self.alpha = [0] * self.rounds
        #self.error1 = []

        #the selected learners choosen at each iteration
        selected_learners=[]

        #Iterate through the number of self.rounds
        for t in range(self.rounds):
            #print "weight", weight


            if t == 0:      #since for the first iteration, the sample is same with the training sample
                sample = X
            else:
                # use the new weights in other iterations to choose the next training sample
                # this tries to help your model understand new things about your data expecially the samples that
                # wasn't well classified in previous iteration. Obviously setting this to X as in t==0, makes this
                # algorithm behave like majority voting. But this last statement applies to deterministic algorithms.

                if self.weightedSampling == True:
                    sample = self.weighted_sample(X,weight)
                else:
                    #print "in false"
                    sample = X
            #print "sample", sample

            #use multiprocessing to get the classifier with the lowest error
            hypothesisResults, error, prediction,weakLearner = self.get_best_classifier_(X,sample,y,self.weakLearners,weight)#)weakLearners[t]

            #print hypothesisResults, error, prediction,weakLearner
            #to update the selected_learners with the new classifier
            selected_learners.append(weakLearner)

            #get the predicted labeels
            self.hypotheses[t] = prediction
            #print "hypotheses[t]", self.hypotheses[t]

            # To make sure you have a list of all the errors made by your classifier
            self.error1.append(error)

            # To append new values for alpha for each round/hypothese to the alpha list
            self.alpha[t] = 0.5 * math.log((1 - error) / (.0001 + error))
            #print "self.alpha", self.alpha[t]

            #Get the new normalized weights
            weight = self.normalize([d * math.exp(-self.alpha[t] * h)
                              for (d,h) in zip(weight, hypothesisResults)])
            print("Round %d, error %.3f, Accuracy %.3f, Alpha %.3f" % (t, error,accuracy_score(y_true=y, y_pred=self.hypotheses[t]),self.alpha[t]))
        #print "self.weakLearners",self.weakLearners
        self.weakLearners = selected_learners
        #print "self.weakLearners", self.weakLearners
        return self

    #def predict(self, X, y, tru_y):

    def predict(self, X):
        #print self.weakLearners
        i = 0
        hypo = [None] * len(self.weakLearners)

        for weakLearner in self.weakLearners:
            hypo[i] = weakLearner.predict(X)
            i+=1

        #print "hypo", hypo
        pred = []
        for j in range(len(X)):
            value = []
            for l in range(len(self.alpha)):
                value.append(self.alpha[l]*hypo[l][j])
            #print sum(value)
            #this line is to enable the ROC method have access to the final values
            self.summed_values_.append(sum(value))

            pred.append(self.sign(sum(value)))

        return np.array(pred)

            #value = self.sign(sum([self.alpha[0]*hypo[0][j], self.alpha[1]*hypo[1][j]]))


    def confusion_matrix1(self,y_true, y_pred):
        TN, FP, FN, TP = 0,0,0,0

        for y1,y2 in zip(y_true, y_pred):

            #print y1,y2
            if y1 ==self.classlabel[0]:         #normal
                if y2 ==y1:
                    TN+=1
                else:
                    FP +=1
            else:
                if y2 ==y1:
                    TP +=1
                else:
                    FN +=1
        return [TN, FP], [FN, TP]



    def roc(self,true_label,k,):
        #print self.confusion_matrix1([1,1,-1,1],[1,1,-1,-1])
        fpr1 = []
        tpr1 = []
        #print "self.summed_values_", len(self.summed_values_),self.summed_values_
        #self.summed_values_.sort()
        #print "self.summed_values_", len(self.summed_values_), self.summed_values_


        for threshold in self.summed_values_:
            new_k_ = []
            for value in self.summed_values_:
                if value >= threshold:
                    new_k_.append(1)
                else:
                    new_k_.append(-1)
            # print new_k_
            #print self.confusion_matrix1(true_label, new_k_)
            #try:
            [TN, FP], [FN, TP] = self.confusion_matrix1(true_label, new_k_)#confusion_matrix(true_label, new_k_)
            #    print true_label
            #    print self.confusion_matrix1(true_label, new_k_),"old",[TN, FP], [FN, TP]
            #except:
            #    pass
            #     TN = confusion_matrix(true_label, new_k_)[0][0]
            #     FP, FN, TP = 0, 0, 0
            #     print true_label
            #     print self.confusion_matrix1(true_label, new_k_),"old",[TN, FP], [FN, TP]

            try:
                fpr = float(FP) / (FP + TN)
            except:
                fpr = 0.0
            try:
                tpr = float(TP) / (TP + FN)
            except:
                tpr = 0.0
            # print "fpr, tpr", fpr, tpr
            fpr1.append(fpr)
            tpr1.append(tpr)

            #print FP, FN, TP,TN, fpr,tpr
        return fpr1, tpr1