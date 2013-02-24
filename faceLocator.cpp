/* this file is for building a face locator to use past data to locate face */
#include "faceLocator.hpp"
#include <okapi/types/exception.hpp>
#include <okapi/utilities/string.hpp>
#include <okapi/utilities/ticpp/ticpp.h>
#include <okapi/utilities/timer.hpp>
#include <fstream>
#include <set>
#include <cv.h>
#include <math.h>

using namespace std;
using namespace okapi;

FaceLocator::FaceLocator(){
    count = 0;
}

FaceLocator::~FaceLocator(){
    // cur nothing
}

void FaceLocator::addFrame(vector<cv::Rect>){ //new image
    // not used yet
}
void FaceLocator::addRect(cv::Rect det){ //new image
    int cent_x = det.x+ det.width/2;
    int cent_y = det.y+ det.height/2;
    cv::Rect temp;

    for( int i = 0; i < faceLocations.size(); i++){
        temp = faceLocations[i];
        if(checkNear(temp, det)){
            update(i, det);
            return;
        }
    }

    // not match
    faceLocations.push_back(det);
    //cout<<"Added one more item"<<endl;
}

bool FaceLocator::checkNear(cv::Rect cur, cv::Rect det, int distance){
    //cout<<"entered checkNear function"<<endl;
    double curD = sqrt(pow((cur.x-det.x),2) + pow((cur.y - det.y),2));
    if(curD < distance)
        return true;
    else 
        return false;
}

void FaceLocator::update(int index, cv::Rect det){
    //cout<<"entered update function"<<endl;
    /*
    int next_x = int((count*faceLocations[index].x + det.x) / ++count);
    int next_y = int((count*faceLocations[index].y + det.y) / ++count);
    */
    int next_x = int((faceLocations[index].x + det.x)/2);
    int next_y = int((faceLocations[index].y + det.y)/2);
    cv::Rect new_Location(next_x, next_y, det.width, det.height);

    faceLocations[index] = new_Location; 
}

vector<cv::Rect> FaceLocator::getLocations() const{
    return faceLocations;
}
vector<cv::Rect> FaceLocator::getLocations(vector< vector<Face> > completed, int index) const{
    vector<cv::Rect> t;
    for(int i = 0; i< completed.size(); i++) {
        Face cur = completed[i][index];
        if(cur.width == 0 && cur.height == 0)
            continue;
        cv::Rect curL;
        curL = cur.toRect();
        t.push_back(curL);
    }
    return t;
}

vector<cv::Rect> FaceLocator::getNormal(vector< vector<Face> > completed, int index) const{
    vector<cv::Rect> t;
    for(int i = 0; i< completed.size(); i++) {
        if(index >= completed[i].size())
            continue;
        Face cur = completed[i][index];
        if( (cur.width == 0 && cur.height == 0) || cur.intered )
            continue;
        cv::Rect curL;
        curL = cur.toRect();
        t.push_back(curL);
    }
    return t;
}


vector<cv::Rect> FaceLocator::getIntered(vector< vector<Face> > completed, int index) const{
    vector<cv::Rect> t;
    for(int i = 0; i< completed.size(); i++) {
        if(index >= completed[i].size())
            continue;
        Face cur = completed[i][index];
        if((cur.width == 0 && cur.height == 0 ) || !cur.intered )
            continue;
        cv::Rect curL;
        curL = cur.toRect();
        t.push_back(curL);
    }
    return t;
}

vector<cv::Rect> FaceLocator::getLocations(vector<Face> completed, int index) const{
    vector<cv::Rect> t;
    if(index >= completed.size())
        return t;
    Face cur = completed[index];
    if(cur.width == 0 && cur.height == 0)
        return t;
    cv::Rect curL;
    curL = cur.toRect();
    t.push_back(curL);
    return t;
}

vector<Face> FaceLocator::getVotedFocus(const vector< vector<Face> >& completed, int index){
    vector<Face> curFrameFaces;
    cout<<"new iteration"<<endl;
    for(int i = 0; i< completed.size(); i++) {
        if(index >= completed[i].size())
            continue;
        Face cur = completed[i][index];
        if(cur.width == 0 || cur.height == 0){
            continue;
        }else{
            curFrameFaces.push_back(cur);
            cout<<"cur x:"<<cur.center.x<<endl;
        }
    }
    if(curFrameFaces.size() == 0){
        vector<Face> empty;
        return empty;
    }
    vector<int> votes(curFrameFaces.size(), 0);
    //when looking at others, it would also be possible to speak
    for(int i = 0; i< curFrameFaces.size(); i++ ){
        Face cur = curFrameFaces[i];
        if(cur.pose > 0){ // looking right
            for (int j = i; j < curFrameFaces.size(); j++ )
                votes[j]++;
        }else if(cur.pose < 0){ // looking left
            for (int j = 0; j <= i; j++ )
                votes[j]++;
        }
        cout<<"cur pose:"<<cur.pose<<endl;
    }
    for (std::vector<int>::iterator it = votes.begin(); it != votes.end(); ++it)
    std::cout << ' ' << *it;
    std::cout << '\n';

    vector<Face> result;

    // max vote
    int max = *max_element(votes.begin(),votes.end()) ;
    cout << "Max vote is " << max << endl;
    for( int i = 0; i<votes.size(); i++){
        if(votes[i] == max){
            result.push_back(curFrameFaces[i]);
        }
    }
    return result;
}
