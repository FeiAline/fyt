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
	float *min = min_element(votes.begin(),votes.end()) ;
    int minIndex = sums.at(distance(sums.begin(),min));
    cout << "Min Sum is " << *min << " at index "<< minIndex << endl;

    return curFrameFaces[minIndex];
}

void Distance::printOut(){
	for (int i = 0; i < curFrameFaces.size(); ++i)
	{
		cout<<"center:"<<curFrameFaces[i].center;
		cout<<" sum:"<<sums[i]<<endl;
	}
}

float Distance::getDistance(Face cur, Face proposedFocus){
	cv::Point2D32f cur2D = to2D(cur);
	float curPose = cur.pose;

	cv::Point2D32f focus2D = to2D(proposedFocus);

	// Method 1:
	// this method directly calculate point to line distance

	// convert degree to pi
	float angle = curPose * PI / 180;
	// line function ax + by + c = 0
	float a = math.tan(angle);
	float b = -1.0;
	float c = cur2D.y - cur2D.x * a;

	float distance = math.abs(a*focus2D.x + b*focus2D.y +c) / math.sqrt(a*a + b*b);

	return distance;
}

cv::Point2D32f Distance::to2D(Face f, int frameWidth){
	int faceWidth = f.width;
	int horX = f.center.x;
	// Assuming that the frame width is 640 px
	float rX = (horX - frameWidth/2)/ faceWidth;

	// relative depth
	float rDepth = 100.0 / width;
	cv::Point2D32f p(rX, rDepth);
	return p;
}