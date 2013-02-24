#include "./fifconvol_v4.0/RP_transform.hpp"
#include <fstream>
#include <vector>
#include <string>

using namespace okapi;
using namespace PhM;

const int stepsz=2;

struct det_pose
{
	string fname;
	cv::Rect_<float> detection;
	float yaw;
};

void read_detection(ifstream& detf, vector<det_pose>& dets)
{
	string pre_fn=" ";
	while (!detf.eof())
	{
		string str;
		getline(detf, str);
		stringstream ss (stringstream::in | stringstream::out);
		ss<< str;
		char filename[50];
		ss>>filename;string strf(filename);
		float det_x, det_y, det_width, det_height;
		ss>>det_x; ss>>det_y; ss>>det_width; ss>>det_height;
		if (strf.compare(pre_fn)!=0)
		{
		det_pose detnow;
		detnow.fname=strf; pre_fn = strf;
		cv::Rect_<float> rectnow(det_x, det_y, det_width, det_height);
		detnow.detection=rectnow;
		dets.push_back(detnow);
		}
	}
}

int main()
{
/*	string root_path("/home/fjjiang/training_data/FacePix/Images/");
	int subs[]={2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 25, 26, 28, 29, 30};
	int n_subs=sizeof(subs)/sizeof(subs[0]);
	for (int isub=0; isub<1; isub++)//n_subs
	{
		int sub=subs[isub];
		char sub_path[300];
		sprintf(sub_path, "%s/Clip%02d_pose_", root_path.c_str(), sub);
		char filename[300];
		sprintf(filename, "/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/dets/det_FacePix_list_%d.txt", sub);
		ifstream detfile(filename);
		vector<det_pose> rawdets;
		read_detection(detfile, rawdets);
		vector<det_pose> selected;
		int id=0;
		for (int yaw=-90; yaw<=90; yaw+=stepsz)
		{
			char fname[50];
			sprintf(fname, "Clip%02d_pose_%d.png", sub,yaw);
			string strf(fname);
			det_pose detnow=rawdets[id]; id+=stepsz;
			if (strf.compare(detnow.fname)!=0)
			{
			printf("Miss detection\n");
			continue;
			}
			detnow.yaw=yaw;
			selected.push_back(detnow);
			//printf("%s, %f, %f, %f, %f\n", detnow.fname.c_str(), detnow.detection.x, detnow.detection.y,detnow.detection.width, detnow.detection.height);
		}
	}*/

float Inmat[5]={5, 2, 6, 7, 9};
cv::Mat In(1, 5, CV_32FC1, Inmat);
cv::Mat Ind;
cv::sortIdx(In, Ind, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
int* id=Ind.ptr<int>(0);
for (int i=0; i<5; i++)
	printf("%d ", id[i]);
}
