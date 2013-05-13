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
	FeatureCollector(int clusterNumber);
	~FeatureCollector();

	void writePoseDiff(vector< vector<Face> > store); // create a text file and write the difference bewtween two faces.
	void writeLocaDiff(vector< vector<Face> > store);

	void writeMouthPix(Face cur, const cv::Mat& gray, int clusterNumber, int frameIndex, cv::VideoWriter);
	void writeMouthDiff(vector< vector<Face> > store);
	void clear();

private:
	void writeVideoMouth(cv::Mat mouthRegion, int clusterNumber,  int frameIndex, cv::VideoWriter);
	int countMouth(const cv::Mat& gray, int threshold = 60);
	float getThreshold(cv::Mat face);
	bool curFrameContainsDummy(vector<Face>);

	vector<float> prev_face_pixel;	
	vector<int> prev_mouth;
	vector<float> prev_pose;

	vector<float> thresholds;
	int totalClusterNumber;
	cv::Mat tempFaceFrame;
	int prev_index;
};


#endif