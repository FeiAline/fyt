#ifndef FACELOCATOR_HPP
#define FACELOCATOR_HPP
#include <okapi/config.hpp>
#include <okapi/types/basictypes.hpp>
#include "heat.hpp"
#include "face.hpp"

#include <string>
#include <vector>
#include <iostream>
#include <highgui/highgui.hpp>
#include <core/core.hpp>

#include <okapi.hpp>
#include <cv.h>

class FaceLocator;

class FaceLocator
{
    public:
        FaceLocator(); // 
        ~FaceLocator(); // Destructor

        void addFrame(vector<cv::Rect>); //new image
        void addRect(cv::Rect); //new image
        vector<cv::Rect> getLocations() const;
        vector<cv::Rect> getLocations(vector< vector<Face> > completed, int index) const;
        vector<cv::Rect> getNormal(vector< vector<Face> > completed, int index) const;
        vector<cv::Rect> getIntered(vector< vector<Face> > completed, int index) const;

        // for single cluster
        vector<cv::Rect> getLocations(vector<Face> completed, int index) const;


        // get voted focus
        vector<Face> getVotedFocus(const vector< vector<Face> >& completed, int index);

    private:
        bool checkNear(cv::Rect cur, cv::Rect det, int distance = 100);
        void update(int index, cv::Rect det);
        vector<cv::Rect> faceLocations;
        int count;
};

#endif
