import pandas as pd
import matplotlib.pyplot as mtp
dataset=pd.read_csv('Malldata.csv')
x=dataset.iloc[:,[3,4]].values
import scipy.cluster.hierarchy as shc
dendro = shc.dendrogram(shc.linkage(x, method="ward"))
mtp.title("Dendrogrma Plot")
mtp.ylabel("Euclidean Distance")
mtp.xlabel("Customers")
mtp.show()
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_pred= hc.fit_predict(x)
mtp.scatter(x[y_pred == 0,0], x[y_pred == 0,1], s=100, c='Blue', label='cluster 1')
mtp.scatter(x[y_pred == 1,0], x[y_pred == 1,1], s=100, c='Green', label='cluster 2')
mtp.scatter(x[y_pred == 2,0], x[y_pred == 2,1], s=100, c='Red', label='cluster 3')
mtp.scatter(x[y_pred == 3,0], x[y_pred == 3,1], s=100, c='Cyan', label='cluster 4')
mtp.scatter(x[y_pred == 4,0], x[y_pred == 4,1], s=100, c='Magenta', label='cluster 5')
mtp.title('Clusters of customers')
mtp.xlabel('Anuual Income(kS)')
mtp.ylabel('Spending Score (1-100)')
mtp.legend()
mtp.show()







