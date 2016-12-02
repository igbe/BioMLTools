import pandas as pd
import mlgrapher as gp
from sklearn import datasets



#data  = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
#data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data", header=None)
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data")
print data
grapher=gp.Mlgrapher()

class_column_index =11
print data.iat[0, class_column_index]
#grapher.parallel_coord(data,class_column_index,class_color_tuple=('red','blue'))

#print data
#print len(data)