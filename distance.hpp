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

#define DIS = 0
#define NORDIS = 1


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
	float innerProduct(Point a, Point b);
	void  outputFrame(int width = 640, int height = 480, int scale = 30);
	void  myLine( cv::Mat img, Point start, float angle );
	void  myPoint( cv::Mat img, Point center );

	void writeInfo(int clusterNukmber, int frameIndex, float dis, int file);

	/* data */
	std::vector<Face> curFaces;
	std::vector<float> sums;
	cv::VideoWriter cvWriter;
};

#endif