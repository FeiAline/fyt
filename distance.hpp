#ifndef DISTANCE_HPP
#define DISTANCE_HPP

#include <okapi/config.hpp>
#include <okapi/types/basictypes.hpp>
#include "PhMdetector.hpp"
#include "GRNN.hpp"
#include "face.hpp"

#include <string>
#include <vector>
#include <iostream>

#include <okapi.hpp>
#include <cv.h>
#include <math.h>

#define PI 3.14159265

struct Point
{
	float x;
	float y;
};

class Distance
{
public:
	Distance();
	~Distance();
	Face getFocus(vector<Face> curFrameFaces);
	void printOut(); // for debug use

private:
	float getDistance(Face cur, Face proposedFocus);
	Point to2D(Face f, int frameWidth = 640);
	/* data */
	std::vector<Face> curFaces;
	std::vector<float> sums;
};

#endif