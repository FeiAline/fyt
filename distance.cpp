#include "distance.hpp"

using namespace std;
using namespace okapi;

Distance::Distance(){
// do nothing in constructor
}
Distance::~Distance(){
}

Face Distance::getFocus(vector<Face> curFrameFaces){
	sums = vector(0,curFrameFaces.size());
	
	for (int i = 0; i < curFrameFaces.size(); ++i)
	{
		cv::Point center = curFrameFaces[i].center;
		for (int j = 0; j < curFrameFaces.size(); ++j)
		{
			if(i!=j){
				float dis = getDistance(curFrameFaces[j], center);
				sums[i] += dis;
			}
		}
	}

}

void Distance::printOut(){
	for (int i = 0; i < curFrameFaces.size(); ++i)
	{
		cout<<"center:"<<curFrameFaces[i].center;
		cout<<" sum:"<<sums[i]<<endl;
	}
}

float Distance::getDistance(Face cur, cv::Point center){

}