#include "distance.hpp"

using namespace std;
using namespace okapi;

Distance::Distance(){
// do nothing in constructor
}
Distance::~Distance(){
}

Face Distance::getFocus(vector<Face> curFrameFaces){
	sums.reserve(curFrameFaces.size());
	curFaces = curFrameFaces;
	
	for (int i = 0; i < curFrameFaces.size(); ++i)
	{
		sums[i] = 0.0;
		for (int j = 0; j < curFrameFaces.size(); ++j)
		{
			if(i!=j){
				float dis = getDistance(curFrameFaces[j], curFrameFaces[i]);
				sums[i] += dis;
			}
		}
	}
	float min = *min_element(sums.begin(),sums.end()) ;
    int minIndex = distance(sums.begin(),min_element(sums.begin(),sums.end()));
    cout << "Min Sum is " << min << " at index "<< minIndex << endl;

    return curFrameFaces[minIndex];
}

void Distance::printOut(){
	for (int i = 0; i < curFaces.size(); ++i)
	{
		cout<<"center:"<<curFaces[i].center.x<<" "<<curFaces[i].center.y;
		cout<<" sum:"<<sums[i]<<endl;
	}
}

float Distance::getDistance(Face cur, Face proposedFocus){
	Point cur2D = to2D(cur);
	float curPose = cur.pose;

	Point focus2D = to2D(proposedFocus);

	// Method 1:
	// this method directly calculate point to line distance

	// convert degree to pi
	float angle = curPose * PI / 180;
	// line function ax + by + c = 0
	float a = tan(angle);
	float b = -1.0;
	float c = cur2D.y - cur2D.x * a;

	float distance = abs(a*focus2D.x + b*focus2D.y +c) / sqrt(a*a + b*b);

	return distance;
}

Point Distance::to2D(Face f, int frameWidth){
	int faceWidth = f.width;
	int horX = f.center.x;
	// Assuming that the frame width is 640 px
	float rX = (horX - frameWidth/2)/ faceWidth;

	// relative depth
	float rDepth = 100.0 / faceWidth;
	Point p;
	p.x = rX;
	p.y = rDepth;
	return p;
}