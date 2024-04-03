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

#include <malloc.h>
#include <math.h>
#include "kmeans.h"

Tuple::Tuple(float x, float y)
{
	_point.setX(x);
	_point.setY(y);
}

void Tuple::setClusterId(int clusterid)
{
	_clusterid = clusterid;
}

int Tuple::getClusterId()
{
	return (_clusterid);
}

Point* Tuple::getPoint()
{
	return (&_point);
}

float Tuple::getEuclidDistance() {
	return(_euclid_dist);
}

void Tuple::setEuclidDistance(float dist) {
	_euclid_dist = dist;
}

Centroid::Centroid(int id, float x, float y)
{
	_id = id;
	_number_of_members = 0;
	_point.setX(x);
	_point.setY(y);
}

Point* Centroid::getPoint()
{
	return (&_point);
}

int Centroid::getId() {
	return(_id);
}

void Centroid::reset() {
	_point.setX(0);
	_point.setY(0);
	_number_of_members = 0;
}

int Centroid::getNumberOfMembers() {
	return(_number_of_members);
}

void Centroid::setNumberOfMembers(int number_of_members) {
	_number_of_members = number_of_members;
}

void Centroid::setPoint(Point* point) {
	_point.setX(point->getX());
	_point.setY(point->getY());
}

void Centroid::setX(float x) {
	_point.setX(x);
}

void Centroid::setY(float y) {
	_point.setY(y);
}

KMeans::KMeans(int num_of_tuples, int num_of_centroids)
{
	_num_of_tuples = num_of_tuples;
	_num_of_centroids = num_of_centroids;

	_tuple_list = (Tuple**) new Tuple * [num_of_tuples];
	_centroid_list = (Centroid**) new Centroid * [num_of_centroids];
}

void KMeans::addTuple(float x, float y)
{
	*(_tuple_list + _tuple_index++) = new Tuple(x, y);
}
void KMeans::addCentroid(float x, float y)
{
	*(_centroid_list + _centroid_index++) = new Centroid(_centroid_index, x, y);
}

void KMeans::setIterationCount(int iteration_count)
{
	_iteration = iteration_count;
}

Tuple** KMeans::getTuples()
{
	return (_tuple_list);
}

void KMeans::updateTuples()
{
	Point* tuple_point;
	Point* centroid_point;
	float euclid_distance = 0;
	float minimum_dist = FLT_MAX;

	for (int tup_idx = 0; tup_idx != _num_of_tuples; tup_idx++)
	{
		tuple_point = ((Tuple*)*(_tuple_list + tup_idx))->getPoint();

		for (int cent_idx = 0; cent_idx != _num_of_centroids; cent_idx++)
		{
			centroid_point = ((Centroid*)*(_centroid_list + cent_idx))->getPoint();

			// Calculate euclid distance between tuple and centroid
			euclid_distance = (float)sqrt(
				pow((tuple_point->getX() - centroid_point->getX()), 2) +
				pow((tuple_point->getY() - centroid_point->getY()), 2));

			if (minimum_dist > euclid_distance)
			{
				minimum_dist = euclid_distance;
				((Tuple*)*(_tuple_list + tup_idx))->setClusterId(
					((Centroid*)*(_centroid_list + cent_idx))->getId()
				);
			}
		}

		// Set Euclid distance of tuple
		((Tuple*)*(_tuple_list + tup_idx))->setEuclidDistance(minimum_dist);

		minimum_dist = FLT_MAX;
	}
}

void KMeans::updateCentroids()
{
	Point* tuple_point;
	Point* total;
	Centroid* pcentroid;

	for (int cent_idx = 0; cent_idx != _num_of_centroids; cent_idx++)
	{
		((Centroid*)*(_centroid_list + cent_idx))->reset();
	}

	for (int tup_idx = 0; tup_idx != _num_of_tuples; tup_idx++)
	{
		pcentroid = ((Centroid*)*(_centroid_list + ((Tuple*)*(_tuple_list + tup_idx))->getClusterId()));

		tuple_point = ((Tuple*)*(_tuple_list + tup_idx))->getPoint();
		total = pcentroid->getPoint();

		total->setX(total->getX() + tuple_point->getX());
		total->setY(total->getY() + tuple_point->getY());

		pcentroid->setPoint(total);
		pcentroid->setNumberOfMembers(pcentroid->getNumberOfMembers() + 1);
	}

	// Set new coordinates of centroid
	for (int cent_idx = 0; cent_idx != _num_of_centroids; cent_idx++)
	{
		pcentroid = ((Centroid*)*(_centroid_list + cent_idx));

		if (pcentroid->getNumberOfMembers() > 0) {
			pcentroid->setX(
				pcentroid->getPoint()->getX() /
				pcentroid->getNumberOfMembers()
			);

			pcentroid->setY(
				pcentroid->getPoint()->getY() /
				pcentroid->getNumberOfMembers()
			);
		}
	}
}

void KMeans::filterOutliers(float sensitivity) {
	Tuple* tuple = 0;
	Centroid* current_centroid = 0;
	Tuple** new_tuple_list = 0;
	float total_distance = 0;
	float avg_euclid_dist = 0;
	float farthest_distance = 0;
	float border_distance = 0;
	int new_tuple_index = 0;

	new_tuple_list = (Tuple**) new Tuple * [_num_of_tuples];

	for (int cent_idx = 0; cent_idx != _num_of_centroids; cent_idx++)
	{
		current_centroid = ((Centroid*)*(_centroid_list + cent_idx));

		for (int tup_idx = 0; tup_idx != _num_of_tuples; tup_idx++)
		{
			tuple = ((Tuple*)*(_tuple_list + tup_idx));

			// Read average distance value of tuple in cluster
			if (tuple->getClusterId() == current_centroid->getId()) {
				total_distance += tuple->getEuclidDistance();

				// Get farthest tuple
				if (tuple->getEuclidDistance() > farthest_distance) {
					farthest_distance = tuple->getEuclidDistance();
				}
			}
		}

		avg_euclid_dist = total_distance / current_centroid->getNumberOfMembers();

		// Calculate outlier border
		border_distance = avg_euclid_dist + ((farthest_distance - avg_euclid_dist) * (1 - (sensitivity / 100)));

		// Filter outliers
		for (int tup_idx = 0; tup_idx != _num_of_tuples; tup_idx++)
		{
			tuple = ((Tuple*)*(_tuple_list + tup_idx));

			if (tuple->getClusterId() == current_centroid->getId())
			{
				if (border_distance > tuple->getEuclidDistance())
				{
					*(new_tuple_list + new_tuple_index++) = tuple;
				}
				else
				{
					delete tuple;
				}
			}
		}
	}

	delete _tuple_list;

	_tuple_list = new_tuple_list;
	_num_of_tuples = new_tuple_index;

	border_distance = 0;
	avg_euclid_dist = 0;
	farthest_distance = 0;
	total_distance = 0;
}

void KMeans::run()
{
	for (int idx = 0; idx != _iteration; idx++)
	{
		updateCentroids();
		updateTuples();
	}
}

int KMeans::getNumberOfClusters()
{
	return (_num_of_centroids);
}

void KMeans::dispose()
{
	// Dispose all objects from memory
	int idx = 0;

	// delete tuple objects from memory
	for (idx = 0; idx != _num_of_tuples; idx++)
	{
		delete* (_tuple_list + idx);
	}

	// delete centroid objects from memory
	for (idx = 0; idx != _centroid_index; idx++)
	{
		delete* (_centroid_list + idx);
	}

	delete (_tuple_list);
	delete (_centroid_list);
}

int KMeans::getNumberOfTuples() {
	return(_num_of_tuples);
}

Centroid** KMeans::getCentroids() {
	return(_centroid_list);
}

Point* KMeans::getClusterUpperBound(int clusterid) {
	Tuple* tuple_point = 0;
	Point* point = 0;
	float upper_bound = 0;

	for (int tup_idx = 0; tup_idx != _num_of_tuples; tup_idx++)
	{
		tuple_point = ((Tuple*)*(_tuple_list + tup_idx));

		if (tuple_point->getClusterId() == clusterid)
		{
			if (tuple_point->getPoint()->getX() + tuple_point->getPoint()->getY() > upper_bound) {
				upper_bound = tuple_point->getPoint()->getX() + tuple_point->getPoint()->getY();
				point = tuple_point->getPoint();
			}
		}
	}

	return(point);
}

Point* KMeans::getClusterLowerBound(int clusterid) {
	Tuple* tuple_point = 0;
	Point* point = 0;
	float lower_bound = FLT_MAX;

	for (int tup_idx = 0; tup_idx != _num_of_tuples; tup_idx++)
	{
		tuple_point = ((Tuple*)*(_tuple_list + tup_idx));

		if (tuple_point->getClusterId() == clusterid)
		{
			if (tuple_point->getPoint()->getX() + tuple_point->getPoint()->getY() < lower_bound) {
				lower_bound = tuple_point->getPoint()->getX() + tuple_point->getPoint()->getY();
				point = tuple_point->getPoint();
			}
		}
	}

	return(point);
}
