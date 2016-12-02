import pandas as pd
import matplotlib.pyplot as plot


class Mlgrapher:



    def parallel_coord(self,data,class_column_index,class_color_tuple=('red','blue')):
        """
        supports only binary classifiers for now
        :param data: the data frame or dataset you are using
        :param class_column_index: the index of the column that contains the class labels
        :param class_color_tuple:this is a tuple of the color to use for coloring the two classes in the dataset
        :return:
        """
        len_data = len(data)
        first_class=[]
        for i in range(len_data):
            if i ==0:
                first_class.append(data.iat[i, class_column_index])
            #assign color based on "M" or "R" labels
            if data.iat[i,class_column_index] in first_class:
                pcolor = class_color_tuple[0]
            else:
                pcolor = class_color_tuple[1]
                #plot rows of data as if they were series data
            dataRow = data.iloc[i,0:class_column_index]
            dataRow.plot(color=pcolor)
        plot.xlabel("Attribute Index")
        plot.ylabel(("Attribute Values"))
        plot.grid()
        plot.show()