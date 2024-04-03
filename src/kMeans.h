/*
	K-Means Arduino library - Unsupervised machine learning clustering
	method of vector quantization

	For detailed information https://en.wikipedia.org/wiki/K-means_clustering

	Copyright(C) 2024 Orkun Gedik <orkungdk@outlook.com>

	This program is free software : you can redistribute it and /or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.If not, see < https://www.gnu.org/licenses/>.
*/

#ifndef _KMEANS_h
#define _KMEANS_h

#define FLT_MAX 3.402823466e+38F

class Point
{
public:
	Point()
	{
		_x = 0;
		_y = 0;
	}
	Point(float x, float y)
	{
		_x = x;
		_y = y;
	}
	void setX(float x)
	{
		_x = x;
	}
	void setY(float y)
	{
		_y = y;
	}
	float getX()
	{
		return (_x);
	}
	float getY()
	{
		return (_y);
	}

private:
	float _x = 0;
	float _y = 0;
};

class Tuple
{
public:
	Tuple(float x, float y);
	void setClusterId(int clusterid);
	int getClusterId();
	Point* getPoint();
	float getEuclidDistance();
	void setEuclidDistance(float dist);

private:
	Point _point;
	int _clusterid = 0;
	float _euclid_dist = 0;
};

class Centroid
{
public:
	Centroid(int id, float x, float y);
	Point* getPoint();
	void setX(float x);
	void setY(float y);
	void setPoint(Point* point);
	int getId();
	int getNumberOfMembers();
	void setNumberOfMembers(int number_of_members);
	void reset();

private:
	Point _point;
	int _id = 0;
	int _number_of_members = 0;
};

class KMeans
{
public:
	KMeans(int number_of_tuples, int number_of_centroids);
	void addTuple(float x, float y);
	void addCentroid(float x, float y);
	void setIterationCount(int iteration_count);
	Tuple** getTuples();
	Centroid** getCentroids();
	void run();
	void dispose();
	int getNumberOfClusters();
	int getNumberOfTuples();
	void filterOutliers(float sensitivity);
	Point* getClusterUpperBound(int clusterid);
	Point* getClusterLowerBound(int clusterid);

private:
	void updateCentroids();
	void updateTuples();
	int _num_of_centroids = 0;
	int _num_of_tuples = 0;
	int _iteration = 0;
	int _tuple_index = 0;
	int _centroid_index = 0;
	Tuple** _tuple_list = 0;
	Centroid** _centroid_list = 0;
};

#endif
