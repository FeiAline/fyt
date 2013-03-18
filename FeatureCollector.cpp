#include "face.hpp"
#include <okapi/types/exception.hpp>
#include <okapi/utilities/string.hpp>
#include <okapi/utilities/ticpp/ticpp.h>
#include <okapi/utilities/timer.hpp>
#include <fstream>
#include <set>
#include <cv.h>
#include "FeatureCollector.hpp"

using namespace std;
using namespace okapi;


/* this file is for data collection for new feature */
FeatureCollector::FeatureCollector(){
}

FeatureCollector::~FeatureCollector(){
}

void FeatureCollector::writePoseDiff(vector< vector<Face> > store){
	ofstream out;
    out.open("features/pose_diff.txt");

    for (int i = 0; i < store.size(); ++i)
    {
    	float prevPose = 0.0;
    	for (int j = 0; j < store[i].size(); ++j)
    	{
    		Face cur = store[i][j];
    		if(cur.isDummy()) {
    			continue;
    		}

    		float poseDiff = cur.pose - prevPose;

    		out<<i<<" "<<j<<" "<<poseDiff<<endl;

    		prevPose = cur.pose;
    	}
    }

    cout<<"write pose Done"<<endl;
    out.close();
}

void FeatureCollector::writeLocaDiff(vector< vector<Face> > store){


    /* write format :
    cluster         frame number        local Diff
    */

	ofstream out;
    out.open("features/loca_diff.txt");

    for (int i = 0; i < store.size(); ++i)
    {
    	// new iteration for a new face
    	cv::Point prevLocation;
    	prevLocation.x = 0;
    	prevLocation.y = 0;
    	for (int j = 0; j < store[i].size(); ++j)
    	{
    		cv::Point cur = store[i][j].center;
            if(cur.x == 0 || cur.y == 0){
                continue;
            }
    		else if(prevLocation.x == 0 && prevLocation.y == 0 && cur.x != 0 && cur.y!=0) {
    			out<<i<<" "<<j<<" "<<0.0<<endl;
    			prevLocation = cur;
    			continue;
    		}

    		float locationDiff = sqrt(pow((cur.x- prevLocation.x),2) + pow((cur.y - prevLocation.y),2));

    		out<<i<<" "<<j<<" "<<locationDiff<<endl;

    		prevLocation = cur;
    	}
    }
    
    cout<<"loca Diff Done"<<endl;
    out.close();

}

void FeatureCollector::writeMouthPix(Face cur, const cv::Mat& gray, int clusterNumber, int frameIndex){
    /* write format :
    cluster          frame number            Mouth Pix
    */
	ofstream out;
    out.open("mouth_pix.txt", fstream::in | fstream::out | fstream::app);

    cout<<"writing new pix"<<endl;

    if(cur.isDummy()){
        out.close();
        cout<<"dummy"<<endl;
        return;
    }


    cv::Mat face = gray(cur.toRect());
    cout<<"face got "<<endl;
    // how to show out the image 
    // taking the lower 1/3 part of the region
    cv::Mat mouthRegion = face(cv::Rect(0, cur.height - cur.height/3 - 1, cur.width, cur.height/3 + 1));

    saveImage("mouth.bmp", mouthRegion);
    saveImage("face.bmp", face);

    cout<<"mouthRegion got "<<endl;
    cout<<mouthRegion.size()<<endl;
    int dark_count = countMouth(mouthRegion);
    out<<clusterNumber<<" "<<frameIndex<<" "<<dark_count<<endl;
/*
    for (int i = 0; i < store.size(); ++i)
    {
        for (int j = 0; j < store[i].size(); ++j)
        {
            Face cur = store[i][j];
            cv::Mat face = gray(cur.toRect());
            // how to show out the image 
            // taking the lower 1/3 part of the region
            cv::Mat mouthRegion = face(cv::Rect(cur.height - cur.height/3, 0, cur.height/3, cur.width));

            int dark_count = countMouth(mouthRegion);

            out<<i<<" "<<j<<" "<<dark_count<<endl;
        }
    }
    */
    cout<<"mouth_pix Done index:"<<frameIndex<<endl;
    out.close();
}

void FeatureCollector::clear(){
    ofstream out;
    out.open("mouth_pix.txt");
    out.close();
    cout<<"cleared"<<endl;
}

void FeatureCollector::writeMouthDiff(vector< vector<Face> > store){
	ofstream out;
    out.open("mouth_diff.txt");



    
    cout<<"mouth_diff Done"<<endl;
    out.close();

}


int FeatureCollector::countMouth(const cv::Mat& mouthRegion, int threshold){
    return cv::countNonZero(mouthRegion <= threshold);
}