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

#define PI 3.14159265

class Distance
{
public:
	distance();
	~distance();
	Face getFocus(vector<Face> curFrameFaces);
	void printOut(); // for debug use

private:
	float getDistance(Face cur, Face proposedFocus);
	cv::Point2D32f to2D(Face f, int frameWidth = 640);
	/* data */
	std::vector<Face> curFrameFaces;
	std::vector<float> sums;
};

#endif