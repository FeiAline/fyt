#include "./fifconvol_v5.0/RP_transform.hpp"
#include <fstream>
#include <vector>
#include <string>

using namespace okapi;
using namespace PhM;

std::vector<float> read_row(string line)
{
	std::vector<float> this_row;
	std::istringstream is(line);
	float val;
	while (is>>val) this_row.push_back(val);
	return this_row;
}

cv::Mat read_text_matf(ifstream& matf)
{
	std::string line;
	std::vector<cv::Mat> rows;
	while (std::getline(matf, line))
	{
	std::vector<float> this_row=read_row(line);
	cv::Mat rownow(1, this_row.size(), CV_32FC1);
	for (int i=0; i<this_row.size(); i++)
		rownow.at<float>(0, i)=this_row[i];
	rows.push_back(rownow);
	}
	int nrows=rows.size();
	int ncols=rows[0].cols;
	cv::Mat matout(nrows, ncols, rows[0].type());
	for (int i=0; i<nrows; i++)
		{
		cv::Mat tem=matout.row(i);
		rows[i].copyTo(tem);
		}
	return matout;
}

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


int main()
{
	string file_X("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/text_X_det_datas_sub2_24_ext27.7_sz28_facepix_gabor_forC_b10.txt");
	string file_y("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/text_y_det_datas_sub2_24_ext27.7_sz28_facepix_gabor_forC_b10.txt");
	string file_b("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/text_b_det_datas_sub2_24_ext27.7_sz28_facepix_gabor_forC_b10.txt");
	ifstream fstr_X(file_X.c_str());
	ifstream fstr_y(file_y.c_str());
	ifstream fstr_b(file_b.c_str());
	cv::Mat X=read_text_matf(fstr_X);
	cv::Mat y=read_text_matf(fstr_y);
	cv::Mat b=read_text_matf(fstr_b);
	//savebin_matrix_se("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/bin_X_det_datas_sub2_24_ext27.7_sz28_facepix_gabor_forC_b10.txt", X);
	//savebin_matrix_se("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/bin_y_det_datas_sub2_24_ext27.7_sz28_facepix_gabor_forC_b10.txt", y);
	//savebin_matrix_se("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/bin_b_det_datas_sub2_24_ext27.7_sz28_facepix_gabor_forC_b10.txt", b);
	cv::Mat test;
	//readbin_matrix_se("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/bin_b_det_datas_sub2_24_ext27.7_sz28_facepix_gabor_forC_b10.txt", test);
	text_mat("./tem.txt", b);
}














