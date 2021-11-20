import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import TrainingData as td
from sklearn.datasets import make_blobs


Default_data, label, centroids = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0,
                                            return_centers=True, cluster_std=1)

data = np.linspace((1, 1), (10, 10), 10)




def euclidean_distance_calc(a, b):
    distance = 0
    for i in range(len(a)):
        bma = (a[i] - b[i])
        bmas = np.square(bma)
        distance = np.sum(bmas) + distance
    return np.sqrt(distance)


def scatter_plot(td, new_points,mode):
    clr = np.array(["red", "blue"])

    for i in td.keys():
        td[i] = np.array(td[i])

        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.scatter(td[i][:, 0], td[i][:, 1], c=clr[i], alpha=0.5, s=10)
        plt.scatter(new_points[0], new_points[1], c=clr[mode], marker="x", s=(5 * 5) ** 2)  # c=clr[i]

def classify_point(point, td, k):
    euclidean = []
    labels = []
    for i in td:
        for j in td[i]:
            Euclidean = euclidean_distance_calc(point, j)
            euclidean.append(Euclidean)
            labels.append(i)
    s_euclidean = np.array(euclidean)
    s_euclidean = np.argsort(s_euclidean)
    s_labels = np.array(labels)
    s_labels = s_labels[s_euclidean]
    labels = s_labels[:k]
    return mode(labels)[0]

def main():
    new_point = np.array([[2, 0],[-1, 1],[5, 10],[3, 4]])
    training_data = td.training_data_generator(Default_data, centroids)
    for p in new_point:
        mode = classify_point(point=p, td=training_data, k=2)
        scatter_plot(training_data, p, mode)

if __name__ == '__main__':
    main()

plt.grid()
plt.show()

