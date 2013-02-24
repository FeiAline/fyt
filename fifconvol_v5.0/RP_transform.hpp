#include <okapi.hpp>
#include "fifconvol_pyr.hpp"
#include <vector>
#include <iostream>
#include <string>

namespace PhM
{
	class RP_transform
	{
	public:
		RP_transform();
		cv::Mat RP_trans(const cv::Mat& grayin) const;
	};
	class RP_pyramid
	{
	std::vector<fifconvol_pyr*> convs;
	std::vector<cv::Mat> RPs;
	cv::Size input_sz;
	std::vector<double> scales;
	fifconvol_pyr* ori_conv;
	public:
		RP_pyramid(const cv::Size image_sz, std::vector<double>& i_scales);
		~RP_pyramid();
		void build(const cv::Mat& img);
		cv::Mat get_RP(int level) const;
		cv::Size getSize(int level, double scale) const;
		cv::Mat get_ene(int level) const;
		double getScale(int level) const;
	};
	
	cv::Mat Gabor_ene_trans(const cv::Mat& grayin, cv::Size& resz);
	cv::Mat RP_trans(const cv::Mat& grayin, cv::Size& resz);
	void readbin_matrix_se(const char* file_loc, cv::Mat& savemat);
	void savebin_matrix_se(const char* file_loc, cv::Mat& savemat);
}
