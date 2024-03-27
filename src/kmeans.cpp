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

#include <stdio.h>

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

void Centroid::addNumberOfMembers() {
	_number_of_members++;
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
	Point* _tuple_point;
	Point* _centroid_point;
	float euclid_distance = 0;
	float minimum_dist = FLT_MAX;

	for (int tup_idx = 0; tup_idx != _num_of_tuples; tup_idx++)
	{
		_tuple_point = ((Tuple*)*(_tuple_list + tup_idx))->getPoint();

		for (int cent_idx = 0; cent_idx != _num_of_centroids; cent_idx++)
		{
			_centroid_point = ((Centroid*)*(_centroid_list + cent_idx))->getPoint();

			// Calculate euclid distance between tuple and centroid
			euclid_distance = (float)sqrt(
				pow((_tuple_point->getX() - _centroid_point->getX()), 2) +
				pow((_tuple_point->getY() - _centroid_point->getY()), 2));

			if (minimum_dist > euclid_distance)
			{
				minimum_dist = euclid_distance;
				((Tuple*)*(_tuple_list + tup_idx))->setClusterId(
					((Centroid*)*(_centroid_list + cent_idx))->getId()
				);
			}
		}

		minimum_dist = FLT_MAX;
	}
}

void KMeans::updateCentroids()
{
	Point* _tuple_point;
	Point* _total;
	Centroid* _pcentroid;

	for (int cent_idx = 0; cent_idx != _num_of_centroids; cent_idx++)
	{
		((Centroid*)*(_centroid_list + ((Tuple*)*(_tuple_list + cent_idx))->getClusterId()))->reset();
	}

	for (int tup_idx = 0; tup_idx != _num_of_tuples; tup_idx++)
	{
		_pcentroid = ((Centroid*)*(_centroid_list + ((Tuple*)*(_tuple_list + tup_idx))->getClusterId()));

		_tuple_point = ((Tuple*)*(_tuple_list + tup_idx))->getPoint();
		_total = _pcentroid->getPoint();

		_total->setX(_total->getX() + _tuple_point->getX());
		_total->setY(_total->getY() + _tuple_point->getY());

		_pcentroid->setPoint(_total);
		_pcentroid->addNumberOfMembers();
	}

	for (int cent_idx = 0; cent_idx != _num_of_centroids; cent_idx++)
	{
		_pcentroid = ((Centroid*)*(_centroid_list + cent_idx));

		if (_pcentroid->getNumberOfMembers() > 0) {
			_pcentroid->setX(
				_pcentroid->getPoint()->getX() /
				_pcentroid->getNumberOfMembers()
			);

			_pcentroid->setY(
				_pcentroid->getPoint()->getY() /
				_pcentroid->getNumberOfMembers()
			);
		}
	}
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
	for (idx = 0; idx != _tuple_index; idx++)
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