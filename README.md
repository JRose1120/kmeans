# k-Means Arduino library
K-Means Arduino library - Unsupervised machine learning clustering method of vector quantization 

Library developed and optimized specific for IoT devices such as Arduino/ESP32 with low memory requirement under embedded machine learning topic.

Using the k-means algorithm on an Arduino or any other microcontroller platform can be beneficial in various applications where real-time data processing and clustering are necessary but computational resources are limited. Here are a few reasons why k-means might be utilized on Arduino:

**Data Compression:** Arduino and similar microcontrollers often have limited memory and processing power. By clustering data using k-means, you can reduce the amount of data that needs to be stored and processed. Instead of storing every data point individually, you can represent each cluster by its centroid, thus saving memory.

**Pattern Recognition:** K-means clustering can be used for simple pattern recognition tasks where the objective is to group similar data points together. This can be useful in applications like gesture recognition, where Arduino needs to differentiate between different hand movements or gestures based on sensor data.

**Anomaly Detection:** K-means clustering can also be used for anomaly detection. By clustering normal behavior patterns during a training phase, the algorithm can identify deviations from these patterns in real-time data, which might indicate anomalies or unusual events.

**Sensor Networks:** In IoT applications or sensor networks where multiple sensor nodes are deployed, k-means clustering can be used to aggregate data from different nodes. This can help in reducing the amount of data transmitted over the network, saving bandwidth and energy.

**Embedded Machine Learning:** Implementing k-means on Arduino can serve as an educational tool for learning about machine learning algorithms in embedded systems. It provides hands-on experience with clustering techniques and how they can be applied in resource-constrained environments.

Find classes, methods and descriptions below Don't need to use Point, Tuple and Centroid classes individually. KMeans class manage all of those classes internally;

## Point:
| Method | Description |
| --- | --- |
| `Point()` | Constructor with initial values |
| `Point(float x, float y)` | Constructor with X, Y coordinate parameters |
| `void setX(float x)` | Set abscissa of point |
| `void setY(float y)` | Set ordinate of point |
| `float getX()` | Get abscissa of point |
| `float getY()` | Get ordinate of point |

## Tuple:
| Method | Description |
| --- | --- |
| `Tuple(float x, float y)` | Constructor with X,Y coordinate parameters |
| `setClusterId(int clusterid)` | Set cluster Id where it belongs to |
| `int getClusterId()` | Get cluster Id where it belongs to |
| `Point* getPoint()` | Get point object of tuple |

## Centroid:
| Method | Description |
| --- | --- |
| `Centroid(int id, float x, float y)` | Constructor with clusterid, X, Y parameters |
| `Point* getPoint()` | Get point object of centroid |
| `void setX(float x)` | Set abscissa of point |
| `void setY(float y)` | Set ordinate of point |
| `void setPoint(Point* point)` | Set point object of centroid |
| `int getId()` | Get cluster Id |
| `int getNumberOfMembers()` | Get number of members of cluster |
| `void addNumberOfMembers()` | Increase number of member counter of cluster |
| `void reset()` | Refresh X, Y, number of members belong to cluster |

## KMeans:
| Method | Description |
| --- | --- |
| `KMeans(int number_of_tuples, int number_of_centroids)` | Constructor with # of tuples, # of centroids parameters. Number of tuples and centroids are allocated in c'tor. Number of addTuple() and addCentroid() calls should be same. |
| `void addTuple(float x, float y)` | Add a new tuple to  |
| `void addCentroid(float x, float y)` | Set abscissa of point |
| `void setIterationCount(int iteration_count)` | Set ordinate of point |
| `Tuple** getTuples()` | Set point object of centroid |
| `Centroid** getCentroids()` | Get cluster Id |
| `void run()` | Set abscissa of point |
| `void dispose()` | Set ordinate of point |
| `int getNumberOfClusters()` | Set point object of centroid |
| `int getNumberOfTuples()` | Get cluster Id |

# Send Us Feedback!
Our library is open source for research purposes, and we want to improve it! So let us know by creating a new GitHub issue or pull request, email

# License
Code released under GNU General Public License v3.0
