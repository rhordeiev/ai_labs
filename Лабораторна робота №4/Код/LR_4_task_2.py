from sklearn.svm import SVC
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X = iris["data"]
y = iris["target"]
#Створення об'єкта KMeans
kmeans = KMeans(n_clusters = 5)
#Начання моделі
kmeans.fit(X)
#Прогнозування результату
y_kmeans = kmeans.predict(X)
#Відображення вхідних точок
plt.scatter(X[:, 0], X[:, 1], c = y_kmeans, s = 50, cmap = "viridis")
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = "black", s = 200, alpha = 0.5)

#Функція пошуку кластерів
def find_clusters(X, n_clusters, rseed = 2):
    #Випадковий вибір кластерів
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        #Призначення міток на основі найближчого до центру
        labels = pairwise_distances_argmin(X, centers)
        #Пошук нових центрів з середнього значення точок
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        #Перевірка на конвергенцію(збіжність)
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels

centers, labels = find_clusters(X, 3)
plt.scatter(X[:, 0], X[:, 1], c = labels,
s = 50, cmap = "viridis")
#Вивід глобально не оптимального результату з конвергенцією
centers, labels = find_clusters(X, 3, rseed = 0)
plt.scatter(X[:, 0], X[:, 1], c = labels,
s = 50, cmap = "viridis")
plt.show()
#Вивід з заданою кількістю кластерів
labels = KMeans(3, random_state = 0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c = labels,
s = 50, cmap = "viridis")
plt.show()
