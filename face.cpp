/* this file is for locating the faces detected */
#include "face.hpp"
#include <okapi/types/exception.hpp>
#include <okapi/utilities/string.hpp>
#include <okapi/utilities/ticpp/ticpp.h>
#include <okapi/utilities/timer.hpp>
#include <fstream>
#include <set>
#include <cv.h>

using namespace std;
using namespace okapi;


/* For Class FaceStore */

FaceStore::FaceStore(){
}

FaceStore::~FaceStore(){
}

void FaceStore::add(int x, int y, int width, int height, int frameIndex){
    // add one face into the storage
    Face newFace;
    newFace.center.x = x + width / 2.0;
    newFace.center.y = y + height / 2.0;

    newFace.width = width;
    newFace.height = height;
    newFace.frameIndex = frameIndex;

    store.push_back(newFace);

    curIndex = frameIndex;
}

vector<vector<Face> > FaceStore::cluster(){
    //clear history
    totalFrame = curIndex; // store the total number for interation
    clustered.clear();
    cout<<"store size:"<<store.size()<<endl;

    for(int i = 0; i < store.size(); i++) {
        bool matched = false;
        for( int j = 0; j < clustered.size();  j++ ){
            if( near(store[i], clustered[j]) ) {
                clustered[j].push_back(store[i]);
                matched = true;
                break;
            }
        }
        // End the iteration, not near to anyone
        if(!matched){
            vector<Face> newC;
            newC.push_back(store[i]);
            if(clustered.size() == 0){
                clustered.push_back(newC);
            }else{
                int c_i = 0;
                bool ex = false;
                Face dummy = clustered[c_i].back();
                int x_dummy = dummy.center.x;
                while(x_dummy < store[i].center.x){
                    c_i++;
                    if(c_i = clustered.size()){
                        ex = true;
                        break;
                    }
                        
                    Face dummy = clustered[c_i].back();
                    x_dummy = dummy.center.x;
                }
                if(ex){
                    clustered.push_back(newC);
                }else{
                    c_i;
                    clustered.insert(clustered.begin()+c_i, newC);
                }
            }
        }
    }
    reduceNoise(30);

    return clustered;
}

vector<vector<Face> > FaceStore::getClustered() const{
    return clustered;
}
vector<Face> FaceStore::getClustered(int i) const{
    return clustered[i];
}


bool FaceStore::near(Face data, vector<Face> cluster){
    Face last = cluster.back();
    return last.near(data);
}

void FaceStore::printOut(){
    cout<<"cur store size:"<<store.size()<<endl;
    cout<<"cur clusters size:"<<clustered.size()<<endl;
    for( int i = 0; i < clustered.size(); i++ ){
        cout<<"cluster "<<i+1<<" size:"<<clustered[i].size();
        cout<<" loation "<<clustered[i].back().center.x<<endl;
    }
}

void FaceStore::interpolate(){
    cout<<"cluster size:"<<clustered.size()<<endl;
    vector< vector<Face> > newClustered ;
    for(int i = 0; i < clustered.size(); i++) {
        vector<Face> t = fillIn(clustered[i]);
        newClustered.push_back(t);
    }
    clustered = newClustered;
}

void FaceStore::interpolate(int clusterNumber){
    cout<<"cluster size:"<<clustered.size()<<endl;
    vector< vector<Face> > newClustered ;
    vector<Face> t = fillIn(clustered[clusterNumber]);
    newClustered.push_back(t);
    clustered = newClustered;
}


vector<Face> FaceStore::fillIn(vector<Face> input){
    vector<Face> t;
    int preFrameIndex = 0;
    int eIndex;

    /* method one -> using linear interpolation */
    for(eIndex = 0; eIndex < input.size(); eIndex ++) {
        Face cur = input[eIndex];
        int frameIndex = cur.frameIndex;
        /*
    
        cout<<"cur frameIndex:"<<frameIndex<<endl;
        cout<<"pre frameIndex:"<<preFrameIndex<<endl;
        */
        if(preFrameIndex == frameIndex) {
            cout<<"at fillin preIndex == frameIndex"<<endl;
            cur.print();
            input[eIndex-1].print();
        }
            
        if ( preFrameIndex == 0 ) {  // begin from the first face detected
            int i;
            for(i = 0; i < frameIndex; i++) // fill all the frame without face null
            {
                Face dummy;
                dummy.intered = true;
                t.push_back(dummy); // with width and height 0
            }
            cout<<i<<" dummy added "<<endl;
            preFrameIndex = frameIndex;
            t.push_back(input[eIndex]);
            continue;
        }

        double diff = frameIndex - preFrameIndex;

        if( diff > 1) {
        // begining of filling in
            Face prev = input[eIndex - 1];
            Face next = input[eIndex];
            // cout<<"pre:"<<prev.center.x<<" "<<prev.center.y<<' '<<prev.width<<' '<<prev.height<<endl;
            // cout<<diff-1<<"Diff filled"<<endl;
            for ( int index = 1; index < diff; index++) { // begin from 1, because gitting rid of front and back
                Face filled;
                cv::Point nextC = next.center;
                cv::Point prevC = prev.center;
                double xslop = (nextC.x - prevC.x) / diff;
                double yslop = (nextC.y - prevC.y) / diff;
                double wslop = (next.width - prev.width) / diff;
                double hslop = (next.height - prev.height) / diff;

                // cout<<"slops:"<<xslop<<" "<<yslop<<' '<<wslop<<' '<<hslop<<endl;
                
                int curX = xslop * index + prevC.x;
                int curY = yslop * index + prevC.y;
                int curW = wslop * index + prev.width;
                int curH = hslop * index + prev.height;

                // cout<<"cur:"<<curX<<" "<<curY<<' '<<curW<<' '<<curH<<endl;

                filled.center.x = curX;
                filled.center.y = curY;
                filled.width = curW;
                filled.height = curH;
                filled.intered = true;

                t.push_back(filled);
            }
            // cout<<"next:"<<next.center.x<<" "<<next.center.y<<' '<<next.width<<' '<<next.height<<endl;
        }
        // always push back what we have 
        t.push_back(input[eIndex]);
        preFrameIndex = frameIndex;
    }

    /* check if all frames are filled in */
    if(preFrameIndex != totalFrame) {
        for(int i = preFrameIndex; i < totalFrame; i++)
            t.push_back(input[--eIndex]);
    }

    return t;
}

void FaceStore::reduceNoise(int tolerance){
    // Algorithm for reducing noise: 
    // we check if the face is detected every tolernace frames 
    // if there are more than 10 instances of the face, it would be confirmed to have a face. 
    for(int i = 0; i< clustered.size(); i++) {
        if(clustered[i].size() > 10)
            continue;
        else {
            Face cur = clustered[i][0];
            int preIndex = cur.frameIndex;
            for( int j = 1; j < clustered[i].size(); j++) {
                Face cur = clustered[i][j];
                int curIndex = cur.frameIndex;
                // cout<< "cur Difference:"<< curIndex-preIndex<<endl;
                cout<<"At Reduce Noise"<<endl;
                cur.print();
                if (curIndex - preIndex < tolerance){
                    preIndex = curIndex;
                }
                else {
                    // git rid of this cluster
                    // Notice: Don't know about the index problem
                    clustered.erase(clustered.begin()+i);
                    cout<< "cluster erased"<<endl;
                    break;
                }
            }
        }
    }
}

void FaceStore::sort(){
    clustered = quickSort(clustered);
}

vector< vector<Face> >  FaceStore::quickSort(vector< vector<Face> > input ){ // sort based on the last face location
    // using quick sort algorithm
    /*
    if (input.size() == 1)
        return input;
    if (input.size() == 2){
        int y1 = input[0].back.y;
        int y2 = input[1].back.y;

        if(  y1 > y2 ){
            vector<Face> tmp = input[0];
            input[0] = input[1];
            input[1] = tmp;
        }
        return input;
    }

    int pivot = input.back.back.y;
    vector< vector<Face> > fv;
    vector< vector<Face> > bv;

    for(int i = 0; i< input.size()-1; i++){
        if(pivot > input[i].back.y)
            fv.push_back(input[i]);
        else
            bv.push_back(input[i]);
    }

    vector< vector<Face> > fvSorted;
    vector< vector<Face> > bvSorted;
    fvSorted = quickSort(fv);
    bvSorted = quickSort(bv);
            

    fvSorted.push_back(input.back);
    for(int i = 0; i< bvSorted.size(); i++){
        fvSorted.push_back(bvSorted[i]);
    }
    */
    //return fvSorted;
    vector< vector<Face> > tmp;
    return tmp;
}

void FaceStore::generatePoses(PhM::pose_estimator& my_est, cv::Mat& gray, int frameIndex){
    for(int i = 0; i < clustered.size(); i++){
        if(clustered[i].size() < frameIndex)
            continue;

        //cout<<"clustered:"<<clustered[i].size()<<endl;
        cv::Rect cur = clustered[i][frameIndex].toRect();
        if((cur.width == 0 && cur.height == 0 )){
            continue;
        }
		float est=my_est.get_pose(gray, cur); 
        cout<<"est:"<<est<<endl;
        clustered[i][frameIndex].pose = est;
    }
}

vector< vector<int> >  FaceStore::getVotes() const{
    return votes;
}

vector<Face> FaceStore::getFacesAtFrame(int frameIndex) const {
    vector<Face> v;
    for (int i = 0; i < clustered.size(); ++i)
    {
        Face f = clustered[i][frameIndex];
        if(!f.isDummy())
            v.push_back(f);
    }
    return v;
}
