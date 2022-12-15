import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
dataset=pd.read_csv('creditcard.csv')
dataset
x = dataset.iloc[:,[3,4]].values
print(x)
from sklearn.cluster import KMeans
wcss_list=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)
mtp.plot(range(1,11),wcss_list)
mtp.title('The Elobw Method Graph')
mtp.xlabel('Number of Cluster(k)')
mtp.ylabel('wcss_list')
mtp.show()
kmeans = KMeans(n_clusters=3, init='k-means++',random_state=42)
y_predict= kmeans.fit_predict(x)
mtp.scatter(x[y_predict == 0,0],x[y_predict == 0,1], s=100, c='Blue', label = 'Cluster 1')
#for first cluster
mtp.scatter(x[y_predict == 1,0],x[y_predict == 1,1], s=100, c='Green', label='Cluster 2')
#for second cluster
mtp.scatter(x[y_predict == 2,0],x[y_predict ==2,1], s=100, c='Red', label='Cluster 3')
#for third cluster
mtp.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='Yellow', label='Centroid')
mtp.title('Clusters of credit card')
mtp.xlabel('V3')
mtp.ylabel('V4')
mtp.legend()
mtp.show()