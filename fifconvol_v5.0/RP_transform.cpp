#include "RP_transform.hpp"
#include <fstream>

namespace PhM
{
	RP_transform::RP_transform()
	{}


	cv::Mat RP_transform::RP_trans(const cv::Mat& grayin) const
	{
		int rows = grayin.rows;
		int cols = grayin.cols;
		fifconvol_pyr conv(cols, rows);
		//cv::Mat fgray;
		//grayin.convertTo(fgray, CV_32FC1);
		int prows;
		int pcols;
		conv.get_padsz(pcols, prows);
		cv::Mat RP_mat(prows, pcols, CV_16SC1);
		//conv.get_RP5(fgray.ptr<float>(0), RP_mat.ptr<short int>(0));
		conv.convol_RP5(grayin.ptr<unsigned char>(0), RP_mat.ptr<short int>(0));
		return RP_mat(cv::Rect(1, 1, cols-2, rows-2)).clone();//
	}

	RP_pyramid::RP_pyramid(const cv::Size image_sz, std::vector<double>& i_scales)
	{	
		input_sz=image_sz;
		scales = i_scales;
		convs.clear();
		RPs.clear();
		for (int i=0; i< scales.size(); i++)
		{
			cv::Size sznow=getSize(i, scales[i]);

			fifconvol_pyr* conv = new fifconvol_pyr(sznow.width, sznow.height);
			int prows;
			int pcols;
			conv->get_padsz(pcols, prows);
			cv::Mat RPnow(prows, pcols, CV_16SC1);
			RPs.push_back(RPnow);
			convs.push_back(conv);
		}
		ori_conv = NULL;
		if (scales[0]>1.0f)
			ori_conv = new fifconvol_pyr(input_sz.width, input_sz.height);
	}
	RP_pyramid::~RP_pyramid()
	{
		for (int i=0; i< convs.size(); i++)
			delete convs[i];
		if (ori_conv != NULL)
			delete ori_conv;
	}
	cv::Size RP_pyramid::getSize(int level, double scale) const
	{
		return cv::Size(cvFloor(input_sz.width / scale),
        	                cvFloor(input_sz.height / scale));
	}
	void RP_pyramid::build(const cv::Mat& img)
	{
		cv::Mat gray;
    		if (img.channels() == 1)
        		gray = img;
    		else
        		cv::cvtColor(img, gray, CV_RGB2GRAY);
	 	if (scales[0]>1.0f)
			{
			ori_conv->bind_input(gray.ptr<unsigned char>(0));
			convs[0]->convol_RP5_d(RPs[0].ptr<short int>(0), input_sz.width, input_sz.height);
			}		
			else
			convs[0]->convol_RP5(gray.ptr<unsigned char>(0),RPs[0].ptr<short int>(0));
		for (int i=1; i<convs.size(); i++)
			convs[i]->convol_RP5_d(RPs[i].ptr<short int>(0), input_sz.width, input_sz.height);
	}
	cv::Mat RP_pyramid::get_RP(int level) const
	{	
		cv::Size sznow = getSize(level, scales[level]);
		int rows = sznow.height;
		int cols = sznow.width;
		return RPs[level](cv::Rect(1, 1, cols-2, rows-2)).clone();
	}

	cv::Mat RP_pyramid::get_ene(int level) const
	{
		int prows;
		int pcols;
		convs[level]->get_padsz(pcols, prows);
		cv::Size sznow = getSize(level, scales[level]);
		int rows = sznow.height;
		int cols = sznow.width;
		cv::Mat G_ene(prows, pcols, CV_32FC4);
		convs[level]->convol_ene_d(G_ene.ptr<float>(0), input_sz.width, input_sz.height);
		return G_ene(cv::Rect(0, 0, cols, rows)).clone();
	}

	double RP_pyramid::getScale(int level) const
	{
		return scales[level];
	}

	cv::Mat Gabor_ene_trans(const cv::Mat& grayin, cv::Size& resz)
	{
		int rows = grayin.rows;
		int cols = grayin.cols;
		fifconvol_pyr ori_pyr(cols, rows);
		ori_pyr.bind_input(grayin.ptr<unsigned char>(0));
		fifconvol_pyr conv(resz.width, resz.height);
		int pad_w;
		int pad_h;
		conv.get_padsz(pad_w, pad_h);
		cv::Mat G_ene(pad_h, pad_w, CV_32FC4);
		conv.convol_ene_d_thr(G_ene.ptr<float>(0), cols, rows, 10.0f);
		return G_ene(cv::Rect(0, 0, resz.width, resz.height)).clone();
	}

	cv::Mat RP_trans(const cv::Mat& grayin, cv::Size& resz)
	{
		int rows = grayin.rows;
		int cols = grayin.cols;
		fifconvol_pyr ori_pyr(cols, rows);
		ori_pyr.bind_input(grayin.ptr<unsigned char>(0));
		fifconvol_pyr conv(resz.width, resz.height);
		int pad_w;
		int pad_h;
		conv.get_padsz(pad_w, pad_h);
		cv::Mat G_RP(pad_h, pad_w, CV_16SC1);
		conv.convol_RP5_d(G_RP.ptr<short int>(0), cols, rows);
		//conv.convol_ene_d_thr(G_ene.ptr<float>(0), cols, rows, 10.0f);
		return G_RP(cv::Rect(0, 0, resz.width, resz.height)).clone();
	}

 	float chartofloat(char* buffer_loc, int loc)
    	{
	   float val;
	   val=*(float *)(buffer_loc+4*loc);
	//val=*(unsigned int*) (buffer+loc);
	   return val;
     	}

     	void readbin_matrix_se(const char* file_loc, cv::Mat& savemat) 
     	{
      	 	int length;
       		char * buffer;

            std::ifstream is;
       		is.open (file_loc, std::ios::binary );
       		if (is.is_open())
       		{ // get length of file:
       		is.seekg (0, std::ios::end);
       		length = is.tellg();
       		//cout<<length<<endl;
        	is.seekg (0, std::ios::beg);
   
        	// allocate memory:
         	buffer = new char [2*sizeof(float)];
         	is.read(buffer,2*sizeof(float));
         	int mat_row=chartofloat(buffer,0);
         	int mat_col=chartofloat(buffer,1);
         	delete[] buffer;
         	savemat.create(mat_row, mat_col, CV_32FC1);
         	for (int i=0; i<mat_row; i++)
        		{
	  		//allocate memory
	   		buffer=new char[mat_col*sizeof(float)];
	   		is.read(buffer, mat_col*sizeof(float));
	   		for (int j=0;j<mat_col; j++)
           		savemat.at<float>(i,j)=chartofloat(buffer,j);
	   		delete[] buffer;
         		}
        	}
       		else 
		printf("Error opening file\n");
      		is.close(); 
    	}

	void savebin_matrix_se(const char* file_loc, cv::Mat& savemat)
	{
        std::ofstream fb(file_loc,std::ofstream::binary);
    		unsigned int mat_row=savemat.rows;	
		unsigned int mat_col=savemat.cols;
		float* buffer_float=new float[2];
		buffer_float[0]=mat_row;
		buffer_float[1]=mat_col;
		char* buffer_char;
		buffer_char=(char*)buffer_float;
		fb.write(buffer_char,2*sizeof(float));
		delete[] buffer_float;
		//delete buffer_float;
		for (int i=0; i<mat_row; i++)
		{
	    		buffer_float=new float[mat_col];
			for (int j=0; j<mat_col;j++)
				buffer_float[j]=savemat.at<float>(i,j);
			buffer_char=(char*) buffer_float;
			fb.write(buffer_char, mat_col*sizeof(float));
			delete[] buffer_float;
			//delete buffer_float;
		}
		printf("%d chars were saved\n", (2+mat_col*mat_row)*sizeof(float));
		fb.close();
	}
}


