#include "./fifconvol_v5.0/RP_transform.hpp"
#include "GRNN.hpp"
#include <fstream>
#include <vector>
#include <string>

using namespace okapi;
using namespace PhM;

struct det_pose
{
	string fname;
	okapi::MCTDetection detection;
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
			detnow.detection.box=rectnow;
			dets.push_back(detnow);
			}
		}
}

void text_read(ifstream& textf, vector<string>& subs, vector<string>& paths)
{
	string line;
	subs.clear();
	while (std::getline(textf, line))
		{
		std::istringstream is(line);
		char subname[50];
		is>>subname;
		string sub(subname);
		subs.push_back(sub);
		char pathname[100];
		is>>pathname;
		string path(pathname);
		paths.push_back(path);
		}
}

void text_read_pose(ifstream& textf, vector<int>& imgids, vector<float>& yaws)
{
	string line;
	imgids.clear();
	yaws.clear();
	while (std::getline(textf, line))
	{
		std::istringstream is(line);
		int imgid;
		is>>imgid;
		imgids.push_back(imgid);
		float tem;
		is>>tem;
		is>>tem;
		float yaw;
		is>>yaw;
		yaws.push_back(yaw);
	}
}

void text_mat(char* file_name, cv::Mat& mat)
{
	ofstream fb(file_name);
	for (int i=0; i<mat.rows; i++){
		for (int j=0; j<mat.cols; j++)
		{
			char val[10];
			sprintf(val, "%.10f ", mat.at<float>(i, j));
			fb<<val;
		} fb << endl; }
}

int main()
{
	//load the estimator
	//string file_X("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/bin_X_det_datas_sub2_25_ext30_facepix_gabor_forC_b10.txt");
	//string file_y("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/bin_y_rad1.95_det_datas_sub2_25_ext30_facepix_gabor_forC_b10.txt");
	//string file_b("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/bin_b_rad1.95_det_datas_sub2_25_ext30_facepix_gabor_forC_b10.txt");	
	string file_X("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/bin_X_det_datas_sub2_24_ext27.7_sz28_facepix_gabor_forC_b10.txt");
	string file_y("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/bin_y_det_datas_sub2_24_ext27.7_sz28_facepix_gabor_forC_b10.txt");
	string file_b("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/bin_b_det_datas_sub2_24_ext27.7_sz28_facepix_gabor_forC_b10.txt");
	pose_estimator my_est(file_X, file_y, file_b, 1.71f, 60);
	std::vector<string> subs, paths;
	ifstream sublist("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v_det/data_info/sub_list.txt");
	text_read(sublist, subs, paths);
	ofstream estf("tem.txt");//../results/result_60_b10_ext27.7_sz28.txt
	for (int isub=0; isub<subs.size(); isub++)//
		{
		char filename[300];
		sprintf(filename, "/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v_det/faceimgs/dets/det_mydata_list_%s.txt.txt", subs[isub].c_str());
		ifstream detf(filename);
		std::vector<det_pose> dets;
		read_detection(detf, dets);
		sprintf(filename, "%s/%s/abs_yaw_samples_2.txt", paths[isub].c_str(), subs[isub].c_str());
		ifstream annof(filename);
		std::vector<int> imgids;
		std::vector<float> yaws;
		text_read_pose(annof, imgids, yaws);
		//cv::Mat embed_all(dets.size(), 100, CV_32FC1);
		ProgressBar bar(dets.size(), false, cerr);
		int yawid=-1;
		float errnow=0;
		for (int i=0; i<dets.size(); i++)
			{
			bool getit=false;
			//float yawnow;
			while(!getit)
			{
				char imgname[100];
				sprintf(imgname, "L_%d.bmp", imgids[++yawid]);
				string fname(imgname);
				getit=fname.compare(dets[i].fname)==0;
			}
			//yawnow=yaws[yawid];
			char imagepath[300];
			sprintf(imagepath, "%s/%s/Left_sample_yaw/%s", paths[isub].c_str(), subs[isub].c_str(), dets[i].fname.c_str());
			cv::Mat gray=okapi::loadImageGray(imagepath);
			//cv::Mat embed;//=embed_all.row(i);
			//OKAPI_TIMER_START("Pose");
			float est=my_est.get_pose(gray, dets[i].detection); 
			//OKAPI_TIMER_STOP("Pose");
			//estf<<est<<endl;
			estf<<dets[i].fname.c_str()<<" "<<imgids[yawid]<<" "<<est<<" "<<yaws[yawid]<<" "<<abs(est-yaws[yawid])<<endl;
			//cv::Mat tem=embed_all.row(i);
			//embed.copyTo(tem);
			++bar;
			errnow+=abs(est-yaws[yawid]);
			}
		errnow=errnow/(float)dets.size();
		printf("%s, error: %e, number %d\n", subs[isub].c_str(), errnow, dets.size());
		//text_mat("./embed.txt", embed_all);
		}
}
