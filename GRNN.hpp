#ifndef GRNN_HPP
#define GRNN_HPP
#include <okapi.hpp>
#include <cv.h>
#include <highgui.h>
#include <vector>
#include <string>

namespace PhM
{
	const int det_sz=28;//32
	class GRNN
	{
		cv::Mat Train_in;
		cv::Mat Train_out;
		cv::Mat Weights;
		float Sigma;
		int dim_in;
		int dim_out;
		int n_patterns;
	public:
		GRNN(string& file_in, string& file_out, float Sig);
		~GRNN();
		cv::Mat estimate(cv::Mat& V_in, float Sig=0.0f);//row vector
		cv::Mat estimate_K(cv::Mat& V_in, int K, float Sig=0.0f);
	};

	class pose_estimator
	{
		GRNN myGRNN;
		cv::Mat b;
		float Sigma;
		int K;
		float extend_sz;
		
		int dim_embed;
	public:
		pose_estimator(string& file_in, string& file_out, string& file_b, float Sig, int Knn);
		~pose_estimator();
		vector<float> get_pose(cv::Mat& gray, std::vector<okapi::MCTDetection>& dets);
        /* this file is implemented for pure Rect get pose */
		float get_pose(cv::Mat& gray, cv::Rect det);//, cv::Mat& embed
		float get_pose(cv::Mat& ene_map, okapi::MCTDetection& det, double scalenow);
		float get_pose(cv::Mat& gray, okapi::MCTDetection& det);//, cv::Mat& embed
	};
}
#endif
