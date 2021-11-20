import numpy as np
def euclidean_distance_calc(a, b):
    distance = 0
    for i in range(len(a)):
        bma = (a[i] - b[i])
        bmas = np.square(bma)
        distance = np.sum(bmas) + distance
    return np.sqrt(distance)

def training_data_generator(dataset, new_points):
    m = {i: [] for i in range(len(new_points))}
    l = np.empty(len(new_points))

    for i in dataset:
        for lc, c in enumerate(new_points):
            old_distance = euclidean_distance_calc(i, c)
            l[lc] = old_distance
        m[np.argmin(l)].append(i)
    return m