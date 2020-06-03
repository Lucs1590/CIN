import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets

data = datasets.load_wine()
X = data.data[:, [0, 11]]
y = data.target

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')  # within cluster sum of squares
# -------------

np.random.seed(7)

X = data.data
name = 'k_means_3'


fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

est = KMeans(n_clusters=3, init='k-means++',
             max_iter=300, random_state=5).fit(X)
labels = est.labels_
ax.scatter(X[:, 11], X[:, 0], X[:, 10],
           c=labels.astype(np.float), edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel(data.feature_names[11])
ax.set_ylabel(data.feature_names[0])
ax.set_zlabel(data.feature_names[10])
ax.set_title('3 clusters')
ax.dist = 12

fig = plt.figure(2, figsize=(4, 3))

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
for name, label in [('Classe_0', 0),
                    ('Classe_1', 1),
                    ('Classe_2', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))

y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 11], X[:, 0], X[:, 10], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel(data.feature_names[11])
ax.set_ylabel(data.feature_names[0])
ax.set_zlabel(data.feature_names[10])
ax.set_title('Ground Truth')
ax.dist = 12

fig.show()

fig = plt.figure(3, figsize=(4, 3))
kmeans = KMeans(n_clusters=3, init='k-means++',
                max_iter=300, random_state=5).fit(data.data[:, [0, 11]])

plt.scatter(data.data[:, [0]], data.data[:, [11]], c=kmeans.labels_)
plt.title('3 clusters')
plt.xlabel('Alchool')
plt.ylabel('od280/od315_of_diluted_wines')
plt.show()
