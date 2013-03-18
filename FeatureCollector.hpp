#ifndef FEATURE_COLLECTOR_HPP
#define FEATURE_COLLECTOR_HPP

#include <okapi/config.hpp>
#include <okapi/types/basictypes.hpp>
#include "PhMdetector.hpp"
#include "face.hpp"
#include "distance.hpp"

#include <string>
#include <vector>
#include <iostream>

#include <okapi.hpp>
#include <cv.h>
#include <math.h>

class FeatureCollector {

public:
	FeatureCollector();
	~FeatureCollector();

	void writePoseDiff(vector< vector<Face> > store); // create a text file and write the difference bewtween two faces.
	void writeLocaDiff(vector< vector<Face> > store);

	void writeMouthPix(Face cur, const cv::Mat& gray, int clusterNumber, int frameIndex);
	void writeMouthDiff(vector< vector<Face> > store);
	void clear();

private:
	int countMouth(const cv::Mat& gray, int threshold = 40);
	vector< vector<int> > mouthPix;

};


#endif