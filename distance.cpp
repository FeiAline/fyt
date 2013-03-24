#include "distance.hpp"

using namespace std;
using namespace okapi;

Distance::Distance(){
// do nothing in constructor
	 cvWriter.open("map.avi", CV_FOURCC('D', 'I', 'V', 'X'), 25.0, cv::Size(800,480));
}
Distance::~Distance(){
}

Face Distance::getFocus(vector<Face> curFrameFaces){
	sums.resize(curFrameFaces.size(), 0.0);
	curFaces = curFrameFaces;
	
	for (int i = 0; i < curFrameFaces.size(); ++i)
	{
		if(curFrameFaces[i].isDummy()){
			sums[i] += 1000; //large number so that it could not be chosen
			continue;
		}

		sums[i] = 0.0;
		for (int j = 0; j < curFrameFaces.size(); ++j)
		{
			if(i!=j){
				if(curFrameFaces[j].isDummy()){
					continue;
				}
				float angle = getNormalizedDistance(curFrameFaces[j], curFrameFaces[i]);
				float dis = getDistance(curFrameFaces[j], curFrameFaces[i]);

				// record down information 
				writerDisInfo(i, curFrameFaces[i].frameIndex, dis);
				writerAngleInfo(i, curFrameFaces[i].frameIndex, angle);

				sums[i] += angle;
			}
		}
	}
	vector<float>::const_iterator min = min_element(sums.begin(),sums.end());
    int minIndex = distance(sums.begin(), min_element(sums.begin(),sums.end()));
    cout << "Min Sum is " << *min << " at index "<< minIndex + 1 << endl;

    outputFrame();

    return curFrameFaces[minIndex];
}

void Distance::printOut(){
	for (int i = 0; i < curFaces.size(); ++i)
	{
		cout<<"center:"<<curFaces[i].center.x<<" "<<curFaces[i].center.y;
		cout<<" sum:"<<sums[i]<<endl;
	}
}

float Distance::getNormalizedDistance(Face cur, Face proposedFocus){
	Point cur2D = cur.to2D();
	float curPose = cur.pose;


	Point focus2D = proposedFocus.to2D();

	// Method 1:
	// this method directly calculate point to line distance

	// convert degree to pi
	float angle = curPose * PI / 180;
	// line function ax + by + c = 0
	float a = tan(angle);
	float b = -1.0;
	float c = cur2D.y - cur2D.x * a;

	// One Problem remaining: When the face is at the opposite side, of the viewing angle, we need to make some modification
	// make judgement if the face is at the right side using Inner product
	Point faceDifference;
	faceDifference.x = focus2D.x - cur2D.x;
	faceDifference.y = focus2D.y - cur2D.y;

	Point viewPoseDirection;
	viewPoseDirection.x = a;
	viewPoseDirection.y = 1.0;

	float distance;

	if( innerProduct(faceDifference, viewPoseDirection) > 0) {
		distance = abs(a*focus2D.x + b*focus2D.y +c) / sqrt(a*a + b*b);
		angle = distance / sqrt(pow(faceDifference.x,2) + pow(faceDifference.y,2)); // get the angle 
	}else{
		angle = 1; // maximize the distance 
	}

	return angle;
}


float Distance::getDistance(Face cur, Face proposedFocus){

	Point cur2D = cur.to2D();
	float curPose = cur.pose;


	Point focus2D = proposedFocus.to2D();

	// Method 1:
	// this method directly calculate point to line distance

	// convert degree to pi
	float angle = curPose * PI / 180;
	// line function ax + by + c = 0
	float a = tan(angle);
	float b = -1.0;
	float c = cur2D.y - cur2D.x * a;

	// One Problem remaining: When the face is at the opposite side, of the viewing angle, we need to make some modification
	// make judgement if the face is at the right side using Inner product
	Point faceDifference;
	faceDifference.x = focus2D.x - cur2D.x;
	faceDifference.y = focus2D.y - cur2D.y;

	Point viewPoseDirection;
	viewPoseDirection.x = a;
	viewPoseDirection.y = 1.0;

	float distance;

	if( innerProduct(faceDifference, viewPoseDirection) > 0) {
		distance = abs(a*focus2D.x + b*focus2D.y +c) / sqrt(a*a + b*b);
	}else{
		distance = sqrt(pow(faceDifference.x,2) + pow(faceDifference.y,2)); // get the angle 
	}
	return distance;
}

float Distance::innerProduct(Point a, Point b){
	return a.x * b.x + a.y * b.y;
}

void Distance::outputFrame(int width, int height, int scale){
	cv::Mat coord(height,width,CV_8UC3);
	for (int i = 0; i < curFaces.size(); ++i){
		Point cur = curFaces[i].to2D();
		cur.x = cur.x * scale;
		cur.y = cur.y * scale;
		myPoint(coord, cur);
		myLine(coord, cur, curFaces[i].pose);
	}
	// saveImage("distance.bmp", coord);
	// TODO make it a video with value
	cvWriter<<coord;
}

void Distance::myLine( cv::Mat img, Point start, float pose)
{
	cv::Point cvStart; 
	cv::Point cvEnd; 

	cvStart.x = start.x + img.cols / 2;
	cvStart.y = img.rows - start.y;

	cvEnd.x = cvStart.x + 100 * sin(pose * PI / 180);
	cvEnd.y = cvStart.y + 100 * cos(pose * PI / 180);

  	int thickness = 2;
  	int lineType = 8;
  	line( img,
        cvStart,
        cvEnd,
        cv::Scalar( 0, 0, 0 ),
        thickness,
        lineType );
}

void Distance::myPoint( cv::Mat img, Point center)
{
 	int thickness = -1;
 	int lineType = 8;

	cv::Point cvCenter; 

	cvCenter.x = center.x + img.cols / 2;
	cvCenter.y = img.rows - center.y;

 	circle( img,
 		cvCenter,
        10,
        cv::Scalar( 0, 0, 255 ),
        thickness,
        lineType );
}

void Distance::writeInfo(int clusterNukmber, int frameIndex, float dis, int file ){
	ofstream out;



	out.close();
}


