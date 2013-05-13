#ifndef BOOST_HPP
#define BOOST_HPP

#include <okapi/config.hpp>
#include <okapi/types/basictypes.hpp>
#include "PhMdetector.hpp"
#include "GRNN.hpp"
#include "face.hpp"
#include "distance.hpp"
#include "FeatureCollector.hpp"
#include <cstdlib>
#include <ml.h>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <string>
#include <vector>
#include <iostream>
#include <okapi.hpp>
#include <cv.h>
#include <math.h>

class Boost
{
public:
	Boost(char* model);
	~Boost();
	float predict(cv::Mat features);
	bool isSpeaker(Face f, int clusterIndex);
	int getFocus(vector<Face> curFrame, int pre_focus);
	void setSize(int size){nOfFaces = size;};

	void getErrorRate(); //  get classification error rate

private:
	vector<vector<float> > readText(char* filename);
	vector<vector<char> > readTruth(char* filename);
	int nOfFaces; 
	CvBoost boost;
	int maxFrameIndex;
};

#endif  