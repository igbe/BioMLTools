import random
import math
from numpy.random import choice

def computeError(h, y, weights=None):
    hypothesisResults = []
    if weights is None:
        weights = [1.] * len(y)
    for i in range(len(y)):
        hypothesisResults.append(h[i]*y[i])     # +1 if correct, else -1
    return hypothesisResults, sum(w for (z,w) in zip(hypothesisResults, weights) if z < 0)

# normalize a distribution
def normalize(weights):
    norm = sum(weights)
    return tuple(m / norm for m in weights)


def sign(x):
    return 1 if x >= 0 else -1



def weighted_sample(X,initial_weights):
    indices = range(len(X))
    #print indices
    my_draw = choice(indices, len(X), p=initial_weights)
    #print my_draw

    return X[my_draw]

def boost(X,y,weakLearners, rounds):
   distr = normalize([1.] * len(X))     #the probability distribution or the weights assigned to each training set
   #print "weights",distr
   print "len of weakLearner", len(weakLearners)
   tru_y = y
   hypotheses = [None] * rounds
   alpha = [0] * rounds
   error1 = []
   #tresh = [i for i in range(len(y))]
   for t in range(rounds):
       #print "weight", distr
       if t == 0:
           sample = X
       else:
           sample = weighted_sample(X,distr)
       #print "sample", sample
       try:
           weakLearner, tresh = weakLearners[t]
       except:
           weakLearner = weakLearners[t]

       hypotheses[t] = weakLearner.fit(sample, y).predict(X)
       print "hypotheses[t]", hypotheses[t]
       hypothesisResults, error = computeError(hypotheses[t], y, distr)
       #print "hypothesisResults, error", hypothesisResults, error
       #print "Error", error
       error1.append(error)
       alpha[t] = 0.5 * math.log((1 - error) / (.0001 + error))
       #print "alpha", alpha[t]
       distr = normalize([d * math.exp(-alpha[t] * h)
                          for (d,h) in zip(distr, hypothesisResults)])
       print("Round %d, error %.3f" % (t, error))

   def finalHypothesis(y,tru_y):
       fHypothesis = []

       for i in range(len(y)):
           sum1 = []
           for (a,h) in zip(alpha,hypotheses):
               #print a*h[i]
               sum1.append(a*h[i])
               #print sign(sum(sum1))
           fHypothesis.append(sign(sum(sum1)))

      # print fHypothesis
       #return sign(sum(a * h(x) for (a, h) in zip(alpha, hypotheses)))
       def error(fHypothesis, tru_y):
           #print "tru_y",tru_y
           #print "hypothesis", fHypothesis
           sum2=[]
           for i in range(len(tru_y)):
               #print fHypothesis[i],tru_y[i]
               if fHypothesis[i] != tru_y[i]:
                   sum2.append(1)
           #print sum2
           return sum(sum2) / float(len(tru_y))
           #for i in range(len(y)):
       return fHypothesis, error(fHypothesis,tru_y)
           #return sum(1 for x, y in data if h(x) != y) / len(data)
   final,error = finalHypothesis(y,tru_y)

   return final,error
