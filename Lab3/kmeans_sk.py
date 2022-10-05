from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

#We generate the example in the book
blob_centers = np.array(
    [[ 0.2,  2.3],
     [-1.5 ,  2.3],
     [-2.8,  1.8],
     [-2.8,  2.8],
     [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)
#(We could do supervised learning since we have the labels, but not now)

#We set k and run the algorithm
k = 5
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X)

print(y_pred) #These are the predicted cluster labels
print(kmeans.cluster_centers_) #These are the predicted cluster centers

#We scatterplot the input data and the centroids highlighting the latter
plt.scatter(X[:, 0], X[:, 1], s=1)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X')
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.show()

#Let's try the iris dataset from before
data = load_iris()
X = data.data #length and width for sepals and petals
y = data.target

k = 3 #We know there are three types of flowers
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X)

print(y_pred)
print(kmeans.cluster_centers_)

