import numpy as pandas
import pandas as pd
import matplotlib.pyplot as mtp
dataset=pd.read_csv('Malldata.csv')
x=dataset.iloc[:,[3,4]].values
import scipy.cluster.hierarchy as shc
dendro = shc.dendrogram(shc.linkage(x, method="ward"))
mtp.title("Dendrogram plot")
mtp.ylabel("Euclidean Distance")
mtp.xlabel("Customers")
mtp.show()