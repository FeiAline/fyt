/* this file is for locating the faces detected */
#include "heat.hpp"
#include <okapi/types/exception.hpp>
#include <okapi/utilities/string.hpp>
#include <okapi/utilities/ticpp/ticpp.h>
#include <okapi/utilities/timer.hpp>
#include <fstream>
#include <set>
#include <cv.h>

using namespace std;
using namespace okapi;

Heat::Heat(cv::Size size, int value){
    heat = cv::Mat(480,640,CV_8UC1,cv::Scalar(0.0,0.0,0.0,0.0)); //  32 bit single channel matrix
}

Heat::~Heat(){} // Destructor do nothing

cv::Size Heat::getSize() const{
    return heat.size();
}

void Heat::setSize(cv::Size){
    // currently don't need 
}
void Heat::saveHeat(){ // output the image of heat

    // create a blank image
    cv::Mat output;
    output = cv::Mat(480,640,CV_8UC1,cv::Scalar(0.0,0.0,0.0,0.0));

    // find the biggest value

    // nomalize the image

    // save the Image
    saveImage("heat.bmp", output);
    saveImage("heat.bmp", heat);
}

void Heat::add(cv::Mat frame){
    cv::Mat gray;
    if (frame.channels() == 1)
        gray = frame;
    else
        cv::cvtColor(frame, gray, CV_RGB2GRAY);
    /*

    printf("heat-size: %i \n", heat.rows);
    printf("gray-size: %i \n", gray.rows);
    
    printf("heat: %i \n", heat.type());
    printf("gray: %i \n", gray.type());

    */

    cv::addWeighted(heat, 1, gray, 1, 0.0, heat);
}
void Heat::add(int x, int y, int width, int height){

    cv::Rect roi = cv::Rect(x,y,width,height);
    cv::Mat roiImg;
    roiImg = heat(roi);

    cv::Mat clone = roiImg.clone();
    cv::Mat one(height,width,CV_8UC1,cv::Scalar(1));
    cv::addWeighted(clone, 1, one, 1, 0.0, clone);
    clone.copyTo(roiImg);

    //cv::Mat roi(heat,cv::Rect(x,y,width,height)); // region of interets
    /*
    for(int i = 0; i < roi.rows; i++)
    {
        uchar* p = roi.ptr(i);
        for(int j= 0; j<roi.cols; j++){
            *p++;
        }
    }
    printf("added one face"):
    */
/*
    //assuming you are using uchar
    uchar* data = square.data;
    for(int i = 0; i < 640*480; i++)
    {
      //operations using *data
      *data++;
    }
*/
    //cv::cvAdd(heat,roi,heat,NULL);
    
    return;
}
