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
FeatureCollector::FeatureCollector(int clusterNumber){
    prev_mouth.resize(clusterNumber, -1);
    prev_face_pixel.resize(clusterNumber, -1);
    thresholds.resize(clusterNumber, -1);
    totalClusterNumber = clusterNumber;
    prev_index = 0;
    tempFaceFrame = cv::Mat::zeros(480, 800, CV_8UC3);
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

void FeatureCollector::writeMouthPix(Face cur, const cv::Mat& gray, int clusterNumber, int frameIndex, cv::VideoWriter writer){
    /* write format :
    cluster          frame number            Mouth Pix
    */

    if(cur.isDummy()){
        cout<<"dummy"<<endl;
        return;
    }
	ofstream out;
    ofstream diff_out;
    ofstream face_diff_out;
    ofstream face_out;

    face_diff_out.open("features/face_pix_diff.txt", fstream::in | fstream::out | fstream::app);
    face_out.open("features/face_pix.txt", fstream::in | fstream::out | fstream::app);

    cv::Mat face = gray(cur.toRect());

    float cur_face_pixel = 0;
    for(int i=0; i<face.rows; i++){
        for(int j=0; j<face.cols; j++){
            cur_face_pixel += face.data[face.step[0]*i + face.step[1]* j];
        }
    } 

    // taking the average
    cur_face_pixel = cur_face_pixel / (face.rows*face.cols);

    if(prev_face_pixel[clusterNumber] == -1){
        prev_face_pixel[clusterNumber] = cur_face_pixel;
    }else {
        face_out<<clusterNumber<<" "<<frameIndex<<" "<<cur_face_pixel<<endl;
        face_diff_out<<clusterNumber<<" "<<frameIndex<<" "<<abs(cur_face_pixel - prev_face_pixel[clusterNumber])<<endl;
        prev_face_pixel[clusterNumber] = cur_face_pixel;
    }


    out.open("features/mouth_pix.txt", fstream::in | fstream::out | fstream::app);
    diff_out.open("features/mouth_diff.txt", fstream::in | fstream::out | fstream::app);

    cout<<"writing new pix"<<endl;


    // how to show out the image 
    // taking the lower 1/3 part of the region
    cv::Mat mouthRegion = face(cv::Rect(0, cur.height - cur.height/3 - 1, cur.width, cur.height/3 + 1));

    saveImage("mouth.bmp", mouthRegion);
    saveImage("face.bmp", face);
    if(thresholds[clusterNumber] < 0) {
        // new
        thresholds[clusterNumber] = getThreshold(mouthRegion);
        cout<<thresholds[clusterNumber];
    }
    int dark_count = countMouth(mouthRegion, thresholds[clusterNumber]);
    int diff_count;
    if( prev_mouth[clusterNumber] == -1){
        diff_count = 0;
    }else {
        diff_count = abs(prev_mouth[clusterNumber] - dark_count);
    }

    cout<<"prev:"<<prev_mouth[clusterNumber]<<" dark_count:"<<dark_count<<endl;
    prev_mouth[clusterNumber] = dark_count;

    out<<clusterNumber<<" "<<frameIndex<<" "<<dark_count<<endl;
    diff_out<<clusterNumber<<" "<<frameIndex<<" "<<diff_count<<endl;

    cout<<"mouth_pix Done index:"<<frameIndex<<endl;
    out.close();
    diff_out.close();
    face_diff_out.close();
    face_out.close();


    /* Here I need to show those pixels that is blow the threshod */

    for(int i=0; i<mouthRegion.rows; i++){
        for(int j=0; j<mouthRegion.cols; j++){
            if(mouthRegion.data[mouthRegion.step[0]*i + mouthRegion.step[1]* j + 0] < thresholds[clusterNumber]) 
            {
                mouthRegion.data[mouthRegion.step[0]*i + mouthRegion.step[1]* j + 0] = 255;
                mouthRegion.data[mouthRegion.step[0]*i + mouthRegion.step[1]* j + 1] = 255;
                mouthRegion.data[mouthRegion.step[0]*i + mouthRegion.step[1]* j + 2] = 255;
            }
        }
    } 


    writeVideoMouth(mouthRegion,clusterNumber,frameIndex,writer);
}

void FeatureCollector::clear(){
    ofstream out;
    out.open("features/mouth_pix.txt");
    out.close();
    out.open("features/mouth_diff.txt");
    out.close();
    out.open("features/face_pix_diff.txt");
    out.close();
    cout<<"cleared"<<endl;
}

void FeatureCollector::writeMouthDiff(vector< vector<Face> > store){
    // this function is written in mouthpix as well
	ofstream out;
    out.open("mouth_diff.txt", fstream::in | fstream::out | fstream::app);
/*
    if(cur.isDummy()){
        out.close();
        cout<<"dummy"<<endl;
        return;
    }
    */
    cout<<"mouth_diff Done"<<endl;
    out.close();

}


int FeatureCollector::countMouth(const cv::Mat& mouthRegion, int threshold){
    return cv::countNonZero(mouthRegion <= threshold);
}

void FeatureCollector::writeVideoMouth(cv::Mat mouthRegion, int clusterNumber, int frameIndex, cv::VideoWriter writer){
    if(frameIndex != prev_index){
        // another iteration
        saveImage("last.bmp", tempFaceFrame);
        writer<<tempFaceFrame;        
        tempFaceFrame = cv::Mat::zeros(480, 800, CV_8UC3);
        prev_index = frameIndex;
    }

    cout<<"mouthRegion:"<<mouthRegion.size()<<endl;

    saveImage("mouthRegion_TTT.bmp", mouthRegion);

    cv::Mat tempFace(mouthRegion.cols,mouthRegion.rows,CV_8UC3); 
    cvtColor(mouthRegion,tempFace,CV_GRAY2BGR);

    // normal operations
    cv::Mat dst_roi = tempFaceFrame(cv::Rect((150+mouthRegion.cols) * (clusterNumber + 1), 70 , mouthRegion.cols, mouthRegion.rows));
    tempFace.copyTo(dst_roi);
}

float FeatureCollector::getThreshold(cv::Mat mouthRegion){
    // half of the overall map
    int sum  = 0;

    for (int i = 0; i < mouthRegion.cols; ++i)
    {
        for (int j = 0; j < mouthRegion.rows; ++j)
        {
            sum = sum + mouthRegion.data[mouthRegion.step[0]*i + mouthRegion.step[1]* j + 0];
        }
    }
    return sum / (4*mouthRegion.cols*mouthRegion.rows);
}

