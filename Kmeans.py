import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv("E:\Datasets\driver-data.csv")
print(data.head())

data.columns = ['id','distance','over_speed']
f_1 = data['distance'].values
f_2 = data['over_speed'].values
X = np.array(list(zip(f_1,f_2)))



sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km =  KMeans(n_clusters=k)
    km = km.fit(X)
    sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K,sum_of_squared_distances,'bx-')
plt.xlabel('Number of Cluster')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method to Find Optimal K')
plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
labels = kmeans.predict(X)

plt.scatter(X[:,0],X[:,1],c=labels,s=20,cmap='rainbow')
centres = kmeans.cluster_centers_
plt.scatter(centres[:,0],centres[:,1],marker='*',c='black',s=200,alpha=1)
plt.xlabel('distance feature')
plt.ylabel('speeding feature')
plt.title('NUmber of clusters = 2')


kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
labels = kmeans.predict(X)

plt.scatter(X[:,0],X[:,1],c=labels,s=20,cmap='rainbow')
centres = kmeans.cluster_centers_
plt.scatter(centres[:,0],centres[:,1],marker='*',c='black',s=200,alpha=1)
plt.xlabel('distance feature')
plt.ylabel('speeding feature')
plt.title('NUmber of clusters = 4')
