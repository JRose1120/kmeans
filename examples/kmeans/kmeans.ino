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

#include "kMeans.h"

void setup()
{
  Serial.begin(9600);

	// Create instance 9 tuples and 3 centroids
	KMeans* kmeans = new KMeans(9, 3);

	// Set iteration count
	kmeans->setIterationCount(3);

	// Add tuples
	// Number of tuples must be same number as provided on c'tor
	kmeans->addTuple(6, 0);
	kmeans->addTuple(45, 0);
	kmeans->addTuple(32, 0);
	kmeans->addTuple(9, 510);
	kmeans->addTuple(23, 0);
	kmeans->addTuple(1, 0);
	kmeans->addTuple(89, 0);
	kmeans->addTuple(500, 0);
	kmeans->addTuple(510, 0);

	// Add tuples
	// Number of centroids must be same number as provided on c'tor
	kmeans->addCentroid(0, 0);
	kmeans->addCentroid(1, 1);
	kmeans->addCentroid(14, 17);

	// Execute algorithm
	kmeans->run();

	// Get results
	Tuple** pTuples = kmeans->getTuples();
	Centroid** pCentroid = kmeans->getCentroids();

	Serial.printf("Tuples:\n");
	for (int i = 0; i != kmeans->getNumberOfTuples(); i++) {
		printf("X=%f\tY=%f\tCluster=%d\n",
			((Tuple*)pTuples[i])->getPoint()->getX(),
			((Tuple*)pTuples[i])->getPoint()->getY(),
			((Tuple*)pTuples[i])->getClusterId());
	}

	Serial.printf("\nCentroids:\n");
	for (int j = 0; j != kmeans->getNumberOfClusters(); j++) {
		printf("X=%f\tY=%f\tid=%d\n",
			((Centroid*)pCentroid[j])->getPoint()->getX(),
			((Centroid*)pCentroid[j])->getPoint()->getY(),
			((Centroid*)pCentroid[j])->getId());
	}

	// Dispose memory
	kmeans->dispose();
}

void loop() {

}
