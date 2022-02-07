# K Nearest Neighbors

## What is K Nearest Neighbors
K-Nearest Neighbors is a supervised machine learning algorithm that can be used for classification and regression. K-Nearest Neighbor looks at diffrent classes of data and predicts what class a new data point belongs to by checking K number of the closest data points to the new data point 

1. Classification : Then finds the most frequent class label which will then become the new data point's class.
2. Regression : Then finds the mean which will then become the new data point's class.

- - - - 
### Sources :
1. [Unite.ai](https://www.unite.ai/what-is-k-nearest-neighbors/)
2. [towards data science](https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e)
3. [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- - - -
## How K Nearest Neighbors works
1. Setting K to the chosen number of neighbors.
2. Calculating the distance between a provided/test example and the dataset examples.
3. Sorting the calculated distances.
4. Getting the labels of the top K entries.
5. Returning a prediction about the test example.

### Flow Chart
![picture alt](https://gyazo.com/2ccc1a24b8bc2fe51f6b19e9c780b834.png "Flow Chart")
- - - -
## K Nearest Neighbors in action

### Classification:
#### 2D:
![picture alt](https://i.gyazo.com/74dbc131b931881726322a748eb834a1.png " K Nearest Neighbors in 2D")
#### 3D:
![picture alt](https://i.gyazo.com/a386f313344695b4db75189dad4f9375.png " K Nearest Neighbors in 3D")

### Regression
#### 2D:
![picture alt](https://gyazo.com/013177e535677d421b1751d518776001.png " K Nearest Neighbors in 2D")
#### 3D:
![picture alt](https://gyazo.com/ce0faf2673e684384e4a786d9a1752d3.png " K Nearest Neighbors in 3D")
