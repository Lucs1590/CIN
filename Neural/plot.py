from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_wine()
X = iris.data
y = iris.target

plt.scatter(X[:, 10], X[:, 9], c=y, cmap='gist_rainbow')
plt.title("Wine Dataset")
plt.xlabel('Hue', fontsize=18)
plt.ylabel('Color Intensity', fontsize=18)
plt.show()
