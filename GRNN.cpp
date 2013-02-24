#include "GRNN.hpp"
#include "./fifconvol_v5.0/RP_transform.hpp"
#include <math.h>
#include <fstream>
#include <typeinfo>
namespace PhM
{
void text_mat(char* file_name, cv::Mat& mat)
{
	ofstream fb(file_name);
	for (int i=0; i<mat.rows; i++){
		for (int j=0; j<mat.cols; j++)
		{
			char val[20];
			sprintf(val, "%.10f ", mat.at<float>(i, j));
			fb<<val;
		} fb << endl; }
}
	GRNN::GRNN(string& file_in, string& file_out, float Sig)
	{
	readbin_matrix_se(file_in.c_str(), Train_in);
	readbin_matrix_se(file_out.c_str(), Train_out);
	Sigma = Sig;
	dim_in=Train_in.cols;
	dim_out=Train_out.cols;
	n_patterns = Train_in.rows;
	Weights.create(1, n_patterns, CV_32FC1);
	}
	
	GRNN::~GRNN()
	{}

	cv::Mat GRNN::estimate(cv::Mat& V_in, float Sig)
	{
	if (Sig!=0.0f)
		Sigma=Sig;
	//float* W=Weights.ptr<float>(0);
	cv::Mat V_out(1, dim_out, CV_32FC1, cv::Scalar(0.0f, 0.0f, 0.0f, 0.0f));
	float sumW=0.0f;
	for (int i=0; i<n_patterns; i++)
		{
		float w=norm(V_in, Train_in.row(i), cv::NORM_L2);
		w=w/Sigma;
		w=exp(-w*w/2);
		V_out=V_out+Train_out.row(i)*w;
		sumW+=w;
		}
	V_out = V_out/sumW;
	return V_out.clone();
	}

	cv::Mat GRNN::estimate_K(cv::Mat& V_in, int K, float Sig)
	{
	if (Sig!=0.0f)
		Sigma=Sig;
	float* W=Weights.ptr<float>(0);
	cv::Mat V_out(1, dim_out, CV_32FC1, cv::Scalar(0.0f, 0.0f, 0.0f, 0.0f));
	//#pragma omp parallel for
	for (int i=0; i<n_patterns; i++)
		{
		float w=norm(V_in, Train_in.row(i), cv::NORM_L2);
		w=w/Sigma;
		w=exp(-w*w/2);
		W[i]=w;
		}

	cv::Mat Idx;//(1, n_patterns, CV_32SC1)
	cv::sortIdx(Weights, Idx, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
	int* inds=Idx.ptr<int>(0);
	float sumW=0.0f;
	for (int i=0; i<K; i++)
		{
		int idnow=inds[i];
		V_out+=Train_out.row(idnow)*W[idnow];
		sumW+=W[idnow];
		}
	V_out = V_out/sumW;
	//printf("%e ", sumW);
	return V_out.clone();
	}

	pose_estimator::pose_estimator(string& file_in, string& file_out, string& file_b, float Sig, int Knn)
	:myGRNN(file_in, file_out, Sig)	
	{
	readbin_matrix_se(file_b.c_str(), b);
	K=Knn;
	extend_sz=0.277f;
	dim_embed=b.rows-1;
	}
	
	pose_estimator::~pose_estimator()
	{}
	
	vector<float> pose_estimator::get_pose(cv::Mat& gray, std::vector<okapi::MCTDetection>& dets)
	{
		int ori_wid=gray.cols;
		int ori_hei=gray.rows;
		vector<float> ests;
		for (int i=0; i<dets.size(); i++)
			{
			float ratio=det_sz/(dets[i].box.width*(1+2*extend_sz));
			int reszw=cvRound(ori_wid*ratio);
			int reszh=cvRound(ori_hei*ratio);
			float boundcut = dets[i].box.width*extend_sz;
			int xnow=cvRound((dets[i].box.x-boundcut)*ratio);
			int ynow=cvRound((dets[i].box.y-boundcut)*ratio);
			cv::Size resz(reszw, reszh);
			cv::Mat ene_map = Gabor_ene_trans(gray, resz);
OKAPI_TIMER_START("Pose est");
			cv::Mat face_ene=ene_map(cv::Rect(xnow-1, ynow-1, det_sz, det_sz));
			cv::Mat vec_face=face_ene.reshape(1, 1);
			cv::Mat embed_face=myGRNN.estimate_K(vec_face, K);
			// linear regression
			double est=embed_face.dot(b.rowRange(0, dim_embed).t());
			est+=b.at<float>(dim_embed, 0);
			ests.push_back(est);
OKAPI_TIMER_START("Pose est");
			}
		return ests;
	}

	float pose_estimator::get_pose(cv::Mat& gray, cv::Rect det)//, cv::Mat& embed
    {
		int ori_wid=gray.cols;
		int ori_hei=gray.rows;
			double ratio=det_sz/(det.width*(1+2*extend_sz));
			int reszw=cvRound(ori_wid*ratio);
			int reszh=cvRound(ori_hei*ratio);
			float boundcut = det.width*extend_sz;
			int xnow=cvRound((det.x-boundcut)*ratio);
			int ynow=cvRound((det.y-boundcut)*ratio);
			cv::Size resz(reszw, reszh);
			cv::Mat ene_map = Gabor_ene_trans(gray, resz);
			double est=0;
if (xnow>0 && ynow>0 && xnow+det_sz<ene_map.cols && ynow+det_sz<ene_map.rows)
{
OKAPI_TIMER_START("Pose est");
			cv::Mat face_ene=ene_map(cv::Rect(xnow-1, ynow-1, det_sz, det_sz)).clone();
			cv::Mat vec_face=face_ene.reshape(1, 1);
			//printf("%d\n", vec_face.cols);
			cv::Mat embed_face=myGRNN.estimate_K(vec_face, K);
			//embed=embed_face.clone();
			// linear regression
			est+=embed_face.dot(b.rowRange(0, dim_embed).t());
			est+=b.at<float>(dim_embed, 0);
OKAPI_TIMER_STOP("Pose est");
}
		return est;
    }

	float pose_estimator::get_pose(cv::Mat& gray, okapi::MCTDetection& det)//, cv::Mat& embed
	{
		int ori_wid=gray.cols;
		int ori_hei=gray.rows;
			double ratio=det_sz/(det.box.width*(1+2*extend_sz));
			int reszw=cvRound(ori_wid*ratio);
			int reszh=cvRound(ori_hei*ratio);
			float boundcut = det.box.width*extend_sz;
			int xnow=cvRound((det.box.x-boundcut)*ratio);
			int ynow=cvRound((det.box.y-boundcut)*ratio);
			cv::Size resz(reszw, reszh);
			cv::Mat ene_map = Gabor_ene_trans(gray, resz);
			double est=0;
if (xnow>0 && ynow>0 && xnow+det_sz<ene_map.cols && ynow+det_sz<ene_map.rows)
{
OKAPI_TIMER_START("Pose est");
			cv::Mat face_ene=ene_map(cv::Rect(xnow-1, ynow-1, det_sz, det_sz)).clone();
			cv::Mat vec_face=face_ene.reshape(1, 1);
			//printf("%d\n", vec_face.cols);
			cv::Mat embed_face=myGRNN.estimate_K(vec_face, K);
			//embed=embed_face.clone();
			// linear regression
			est+=embed_face.dot(b.rowRange(0, dim_embed).t());
			est+=b.at<float>(dim_embed, 0);
OKAPI_TIMER_STOP("Pose est");
}
		return est;
	}

// Gabor feature from the pyramid
	float pose_estimator::get_pose(cv::Mat& ene_map, okapi::MCTDetection& det, double scalenow)//, cv::Mat& embed
	{
			//int ori_wid=gray.cols;
			//int ori_hei=gray.rows;
			//double scalenow=pyr->getScale(level);
			double ratio = 1.0/scalenow;
			//double ratio=det_sz/(det.box.width*(1+2*extend_sz));
			float boundcut = det.box.width*extend_sz;
			int xnow=cvRound((det.box.x-boundcut)*ratio);
			int ynow=cvRound((det.box.y-boundcut)*ratio);
			double est=0;
			char filename[200];
			sprintf(filename, "./textmat_%d_%d_new.txt", xnow, ynow);
			if (xnow>2 && ynow>2 && xnow+det_sz<ene_map.cols && ynow+det_sz<ene_map.rows)
{
OKAPI_TIMER_START("Pose est");
			cv::Mat face_ene=ene_map(cv::Rect(xnow, ynow, det_sz, det_sz)).clone();
			cv::Mat vec_face=face_ene.reshape(1, 1);text_mat(filename, vec_face);
			//printf("%d\n", vec_face.cols);
			cv::Mat embed_face=myGRNN.estimate_K(vec_face, K);
			//embed=embed_face.clone();
			// linear regression
			est+=embed_face.dot(b.rowRange(0, dim_embed).t());
			est+=b.at<float>(dim_embed, 0);
OKAPI_TIMER_STOP("Pose est");
}
		return est;
	}
}
