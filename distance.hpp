#ifndef DISTANCE_HPP
#define DISTANCE_HPP

#include <okapi/config.hpp>
#include <okapi/types/basictypes.hpp>
#include "PhMdetector.hpp"
#include "GRNN.hpp"

#include <string>
#include <vector>
#include <iostream>

#include <okapi.hpp>
#include <cv.h>
#include <math.h>

class Distance
{
public:
	distance();
	~distance();
	Face getFocus(vector<Face> curFrameFaces);
	void printOut(); // for debug use

private:
	float getDistance(Face cur, cv::Point center);
	/* data */
	std::vector<Face> curFrameFaces;
	std::vector<float> sums;
};

#endif