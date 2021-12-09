import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import TrainingData as tds
from sklearn.datasets import make_blobs

fig = plt.figure()
ax = fig.add_subplot()


Default_data, label, centroids = make_blobs(n_samples=1000, centers=2, n_features=2,
                                            return_centers=True, cluster_std=20,random_state=10)


def scatter_plot(td, new_points, point, c_r,closest):
    clr = np.array(["red", "blue"])
    training_data = tds.training_data_generator(Default_data, centroids)
    # classification

    if c_r:
        for i in training_data:
            training_data[i] = np.array(training_data[i])

            ax.set_ylabel("Y axis")
            ax.set_xlabel("X axis")
            # ax.set_zlabel("Z axis")

            ax.scatter(training_data[i][:, 0], training_data[i][:, 1], c=clr[i], alpha=0.5, s=10)  # td[i][:, 2]
            ax.scatter(new_points[0], new_points[1], c=clr[point], marker="x", s=(5 * 5) ** 2)  # c=clr[i] new_points[2]
    # regression
    else:
        plt.axis("equal")
        ax.scatter(td[:, 0], td[:, 1], c="g", alpha=.03, s=10)  # td[i][:, 2]
        ax.scatter(new_points[0], new_points[1], c="black", marker="x", s=(5 * 5) ** 2)  # new_points[2],
        ax.scatter(point[0], point[1], color="r", marker="x", s=(5 * 5) ** 2)  # point[2]
        ax.scatter(closest[:,0], closest[:,1], color="r", marker="o", s=(5) ** 2)  # point[2]

def classify_point(point, td, k, c_r=False):
    euclidean = []
    labels = []
    closest = None
    # classification
    if c_r:
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
        return mode(labels)[0],closest
    # regression
    else:
        for i in td:
            Euclidean = tds.euclidean_distance_calc(point, i)
            euclidean.append(Euclidean)
        s_euclidean = np.array(euclidean)
        print(s_euclidean)
        s_euclidean = np.argsort(s_euclidean)
        s_td = np.array(td)
        s_td = s_td[s_euclidean]
        closest = s_td[:k]
        return np.mean(closest, axis=0),closest



def main():
    bool_var = True
    new_point = (np.random.random([10, 2]) * 20) - 10

    for test_point in new_point:
        point,closest = classify_point(point=test_point, td=Default_data, k=2, c_r=bool_var)
        scatter_plot(Default_data, test_point, point,c_r=bool_var,closest=1)


if __name__ == '__main__':
    main()
plt.grid(True)
plt.show()
