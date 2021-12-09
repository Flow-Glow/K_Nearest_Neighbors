import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import TrainingData as tds
from sklearn.datasets import make_blobs

fig = plt.figure()
ax = fig.add_subplot()

Default_data, label, centroids = make_blobs(n_samples=1000, centers=2, n_features=2,
                                            return_centers=True, cluster_std=10,)

def scatter_plot(td, new_points, point, regression=False, classification=False):
    clr = np.array(["red", "blue"])
    training_data = tds.training_data_generator(Default_data, centroids)
    for i in training_data:
        training_data[i] = np.array(training_data[i])

        ax.set_ylabel("Y axis")
        ax.set_xlabel("X axis")
        # ax.set_zlabel("Z axis")
        if classification:

            ax.scatter(training_data[i][:, 0], training_data[i][:, 1], c=clr[i], alpha=0.5, s=10)  # td[i][:, 2]
            ax.scatter(new_points[0], new_points[1], c=clr[point], marker="x", s=(5 * 5) ** 2)  # c=clr[i] new_points[2]
    if regression:
        ax.scatter(td[:, 0], td[:, 1], c="g", alpha=0.03, s=10)  # td[i][:, 2]
        ax.scatter(new_points[0], new_points[1], c="black", marker="x", s=(5 * 5) ** 2)  # new_points[2],
        ax.scatter(point[0], point[1], color="y", marker="x", s=(5 * 5) ** 2)  # point[2]


def classify_point(point, td, k, classification=False, regression=False):
    euclidean = []
    labels = []

    if classification:
        training_data = tds.training_data_generator(Default_data, centroids)
        for i in training_data:
            for j in training_data[i]:
                Euclidean = tds.euclidean_distance_calc(point, j)
                euclidean.append(Euclidean)
                labels.append(i)
        s_euclidean = np.array(euclidean)
        s_euclidean = np.argsort(s_euclidean)
        s_labels = np.array(labels)
        s_labels = s_labels[s_euclidean]
        labels = s_labels[:k]
        return mode(labels)[0]
    if regression:
      
        for i in td:
            Euclidean = tds.euclidean_distance_calc(point, i)
            euclidean.append(Euclidean)
        s_euclidean = np.array(euclidean)
        s_euclidean = np.argsort(s_euclidean)
        s_td = np.array(td)
        s_td = s_td[s_euclidean]
        closest = s_td[:k]
        print(closest)
        return np.mean(closest, axis=0)


def main():
    new_point = (np.random.random([10,2]) * 20)-10
    for test_point in new_point:
        point = classify_point(point = test_point, td = Default_data, k=3, classification=True)
        scatter_plot(Default_data, test_point, point, classification=True)


if __name__ == '__main__':
    main()
plt.grid(True)
plt.show()
