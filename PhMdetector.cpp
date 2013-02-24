// This file is part of the OKAPI library of the Computer Vision for HCI Lab,
// Karlsruhe Institute of Technology
//
// http://cvhci.ira.uka.de/okapi
//
// Copyright (c) 2010, Computer Vision for HCI Lab,
// Karlsruhe Institute of Technology
// All rights reserved.
//
// Authors:
//  Kai Nickel <kai.nickel@kit.edu>
//  Mika Fischer <mika.fischer@kit.edu>
//
// The OKAPI library is free software; you can redistribute it and/or modify it
// according to the terms of the OKAPI license as specified in the LICENSE file
// accompanying this library.

//#include <okapi/detectors/mctdetector.hpp>
#include "PhMdetector.hpp"
#include <okapi/types/exception.hpp>
#include <okapi/utilities/string.hpp>
#include <okapi/utilities/ticpp/ticpp.h>
#include <okapi/utilities/timer.hpp>
#include <fstream>
#include <set>
#include <cv.h>
//#include "PhOP_trans.h"



using namespace ticpp;
using namespace std;
using namespace okapi;

namespace PhM
{
        struct CvPointCompare
        {
            bool operator()(const CvPoint &a, const CvPoint &b)
            {
                if (a.y < b.y)
                    return true;
                else if (b.y < a.y)
                    return false;
                else
                    return a.x < b.x;
            }
        };
    // MCTDetStage //////////////////////////////////////////////////////////////////////

    int* PhMDetStage::generateIndex(int step_size) const
    {
        int* index = new int[classifiers.size()];
        for (unsigned j = 0; j < classifiers.size(); j++)
        {
            const PhMDetClassifier& cj = classifiers[j];
            index[j] = cj.y * step_size + cj.x*2; // MCT pixel size is 2 bytes
        }
        return index;
    }


    // MCTDetector //////////////////////////////////////////////////////////////////////

    PhMDetector::PhMDetector()
    {
        // nothing to do
	rp=RP_transform();
    }

    PhMDetector::PhMDetector(std::string filename)
    {
	    rp=RP_transform();
        printf("RP_transform done \n");
        load(filename);
    }

    PhMDetector::~PhMDetector()
    {
        // nothing to do
    }

    cv::Size PhMDetector::getSize() const
    {
        return cvSize(width, height);
    }

    int PhMDetector::getStages() const
    {
        return stages_phop.size();
    }

    const RP_transform& PhMDetector::getPhOP() const
    {
        return rp;
    }


    void PhMDetector::load(std::string filename)
    {
	stages_phop.clear();
        try
        {
            Document doc;
            printf("doc created \n");
            doc.LoadFile(filename);

            printf("doc loaded \n ");
            loadXMLNode(&doc);
            printf("XML loaded \n ");
        }
        catch (ticpp::Exception& e)
        {
            throw IOException(strprintf("Error loading %s: %s\n", filename.c_str(), e.what()));
        }
    }

    void PhMDetector::loadXMLNode(ticpp::Node* node)
    {
        try
        {
            //Document doc;
            //doc.LoadFile(filename);

            // base settings

            Element* base = node->FirstChildElement("PhMDetector");
            base->GetAttribute("width",  &width);
            base->GetAttribute("height", &height);
	    base->GetAttribute("win_width", &win_width);
	    base->GetAttribute("win_height", &win_height);
	    base->GetAttribute("RIP", &RIP_theta);

            // stages

	    stages_phop.clear();
            Element* el_st = base->FirstChildElement("PhMDetStage");
            while (el_st != NULL)
            {
                PhMDetStage stage_phop;
                el_st->GetAttribute("threshold", &stage_phop.threshold);
                string feature_type;
		feature_type = el_st ->GetAttribute("feature_type");
                // classifiers
/*
                if (feature_type == "MCT" || feature_type == "hybrid")
	{	
		//char file_name[100];
		//sprintf(file_name, "/home/fjiang/CUDA_data/baseline_mct/MCT_fea_loc_stage%d.txt",stages_mct.size() );
		//ofstream f_loc(file_name);
                Element* el_cls = el_st->FirstChildElement("MCTDetClassifier");
                while (el_cls != NULL)
                {
                    PhMDetClassifier cls;
                    el_cls->GetAttribute("x", &cls.x);
                    el_cls->GetAttribute("y", &cls.y);
		//	f_loc<<cls.x<<" "<<cls.y<<endl;

                    string text = el_cls->GetText();
                    istringstream iss(text, istringstream::in);
                    int nr = 0;
                    while (iss >> cls.weights[nr])
                        nr++;

                    if (nr != table_sz)
                        throw IOException("Wrong number of MCTDetector weights.");

                    stage.classifiers.push_back(cls);
                    el_cls = el_cls->NextSiblingElement("MCTDetClassifier", false);
                }
	}
                stages_mct.push_back(stage);*/
		
		//PhMDetStage stage_phop;
		//stage_phop.threshold = stage.threshold;
		
		//phop classifiers
		if (feature_type == "PhOP" || feature_type =="hybrid")
	{
               // char file_name[100];
               // sprintf(file_name, "/home/fjiang/CUDA_data/PKE_det_v2/f_det_config_mag_PKE_2.0_3.0_1.00_1.00_64.xml/PhOP_fea_loc_stage%d.txt",stages_phop.size() );
               // ofstream f_loc(file_name);

		Element* el_cls_1 = el_st->FirstChildElement("PhOPDetClassifier");
                while (el_cls_1 != NULL)
                {
                    PhMDetClassifier cls;
                    el_cls_1->GetAttribute("x", &cls.x);
                    el_cls_1->GetAttribute("y", &cls.y);
		//	f_loc<<cls.x<<" "<<cls.y<<endl;
                    string text = el_cls_1->GetText();
                    istringstream iss(text, istringstream::in);
                    int nr = 0;
                    while (iss >> cls.weights[nr])
                        nr++;

                    if (nr != table_sz)
                        throw IOException("Wrong number of PhOPDetector weights.");

                    stage_phop.classifiers.push_back(cls);
                    el_cls_1 = el_cls_1->NextSiblingElement("PhOPDetClassifier", false);
                }
	}		
		stages_phop.push_back(stage_phop);
		
                el_st = el_st->NextSiblingElement("PhMDetStage", false);
            }

            // User info
            info.clear();
            el_st = base->FirstChildElement("UserInfo", false);
            while (el_st)
            {
                string key, value;
                el_st->GetAttribute("name",  &key);
                el_st->GetAttribute("value", &value);
                info[key] = value;

                el_st = el_st->NextSiblingElement("UserInfo", false);
            }
        }
        catch (ticpp::Exception& e)
        {
            throw IOException(strprintf("Error parsing MCTDetector XML data: %s\n", e.what()));
        }
    }

    void PhMDetector::save(const std::string& filename, const std::string& training_info) const
    {
        try
        {
            Document doc = Document(filename);
            Declaration d("1.0", "ISO-8859-1", "");
            doc.LinkEndChild(&d);

            // base settings

            Element e("PhMDetector");
            Element* base = doc.InsertEndChild(e)->ToElement();
            base->SetAttribute("width",  width);
            base->SetAttribute("height", height);
	    base->SetAttribute("win_width", win_width);
	    base->SetAttribute("win_height", win_height);
	    base->SetAttribute("RIP", RIP_theta);


            if (!training_info.empty())
            {
                Element st("PhMDetTrainingInfo");
                TiXmlText* txt = new TiXmlText(training_info);
                txt->SetCDATA(true);
                st.LinkEndChild(new Text(txt));
                base->InsertEndChild(st);
            }

            // stages

            for (unsigned i = 0; i < stages_phop.size(); i++)
            {
                const PhMDetStage& stage = stages_phop[i];
                Element st("PhMDetStage");
                st.SetAttribute("threshold", stage.threshold);
		//if (stages_mct[i].classifiers.empty())
		st.SetAttribute("feature_type", "PhOP");
		//else if (stages_phop[i].classifiers.empty())
		//st.SetAttribute("feature_type", "MCT");
		//else st.SetAttribute("feature_type", "hybrid");
                // classifiers

                //const PhMDetStage& stage_1 = stages_phop[i];
                for (unsigned j = 0; j < stage.classifiers.size(); j++)
                {
                    const PhMDetClassifier& cj = stage.classifiers[j];
                    Element cl("PhOPDetClassifier");
                    cl.SetAttribute("x", cj.x);
                    cl.SetAttribute("y", cj.y);
                    string text = " ";
                    for (unsigned w = 0; w < table_sz; w++)
                        text += strprintf("%.8f ", cj.weights[w]);
                    cl.SetText(text);
                    st.InsertEndChild(cl);
                }
                
                base->InsertEndChild(st);
            }

            // user info

            map<string,string>::const_iterator it;
            for (it = info.begin(); it != info.end(); ++it)
            {
                Element ui("UserInfo");
                ui.SetAttribute("name",  it->first);
                ui.SetAttribute("value", it->second);
                base->InsertEndChild(ui);
            }

            doc.SaveFile();
        }
        catch (ticpp::Exception& e)
        {
            throw IOException(strprintf("Error saving %s: %s\n", filename.c_str(), e.what()));
        }
    }

    bool PhMDetector::classifyPhMPosition(const cv::Mat& phop_image, int pos_x, int pos_y,
                                          int* n_stages, double* confidence) const
    {
        if (n_stages) *n_stages = 0;
        if (confidence) *confidence = 0.0;
        for (unsigned i = 0; i < stages_phop.size(); i++)
        {
	    //double stagesum = 0;
            //const PhMDetStage& stage = stages_mct[i];
            //stagesum += stage.calcStageSum(mct_image, pos_x, pos_y);
	    const PhMDetStage& stage_1 = stages_phop[i];
	    double stagesum = stage_1.calcStageSum(phop_image, pos_x, pos_y);
           if (stagesum < stage_1.threshold)
                return false;
            if (n_stages)
                *n_stages = *n_stages + 1;
            if (confidence)
                *confidence = stagesum - stage_1.threshold;
        }
        return true;
    }

    bool PhMDetector::classifyPyramidWindow(PhMPyramid& pyramid, const cv::Rect& window,
                                            int* n_stages, double* confidence) const
    {
        // search appropriate scale

        double desired_scale = window.width / (double)this->width;
        int    best_level    = pyramid.findClosestLevel(desired_scale);

        // calculate MCT position

        double scale  = pyramid.getScale(best_level);
        int pos_x = cvRound(window.x/scale) - 1; // -1 to account for MCT border
        int pos_y = cvRound(window.y/scale) - 1;

        // clipping

        //cv::Mat mctimg = pyramid.getMCTImage(best_level);
	cv::Mat phopimg = pyramid.getPhOPImage(best_level);
        if (pos_x + this->width  > phopimg.cols ||
            pos_y + this->height > phopimg.rows ||
            pos_x < 0 || pos_y < 0)
        {
            if (n_stages  ) *n_stages   = 0;
            if (confidence) *confidence = 0.0;
            return false;
        }

        // classification

        return classifyPhMPosition(phopimg, pos_x, pos_y, n_stages, confidence);
    }

    void PhMDetector::scanPyramid(PhMPyramid& pyramid, std::vector<MCTDetection>& detections, int shift_step, std::vector<BinaryPatternRawDetection>* raw_detections) const
    {
        int det_width  = win_width;// 1.414 * (getSize().width+2) -2;
        int det_height = win_height;//1.414 * (getSize().height+2) -2;
        int n_stages   = stages_phop.size();
        for (int i = 0; i < pyramid.getLevels(); i++)
        {
            //cv::Mat mctimg = pyramid.getMCTImage(i);
	    cv::Mat phopimg = pyramid.getPhOPImage(i);
        //std::cout<<"SIZE:"<<phopimg.size()<<endl;
            if (phopimg.cols >= det_width && phopimg.rows >= det_height)
            {
                // create stage indices

                //std::vector<int*> stage_indices_mct;
		std::vector<int*> stage_indices_phop;
                for (int j = 0; j < n_stages; j++)
		{
		//	cout<<j<<endl;
		  //stage_indices_mct.push_back(stages_mct[j].generateIndex(mctimg.step));
			int* index_now;
		  index_now=stages_phop[j].generateIndex(phopimg.step);
		  stage_indices_phop.push_back(index_now);
		}
                // run over image
		//cout<<stage_indices_o.size();
                cv::Size size    = pyramid.getSize(i);
                double   scale_x = double(pyramid.getInputImage().cols) / double(size.width);
                double   scale_y = double(pyramid.getInputImage().rows) / double(size.height);
                int      max_x   = phopimg.cols - det_width;
                int      max_y   = phopimg.rows - det_height;
                //int      pos_inc_mct = shift_step * mctimg.elemSize();
		int      pos_inc_phop = shift_step * phopimg.elemSize();
                //unsigned char* mctdata = mctimg.ptr();
		unsigned char* phopdata = phopimg.ptr();
               // #ifdef _OPENMP
                #pragma omp parallel for
               // #endif
                for (int y = 0; y <= max_y; y += shift_step)
		//for (int y = 1; y < max_y; y += shift_step)
                {
                    unsigned char* ptr_phop = phopdata + y * phopimg.step;
		    //std::vector<unsigned char*> ptr_phop;
		    //for (int o_id=0; o_id<phopdata.size(); o_id++) 			
			//ptr_phop.push_back(phopdata[o_id] + y * phopimg[o_id].step);
		            for (int x = 0; x <= max_x; x += shift_step)
                    //for (int x = 1; x < max_x; x += shift_step)
                    {
			//if (x>15)
			            {
                        int stages_passed = 0;
                        double confidence = 0;
                        do
                        {
			    //double stagesum = 0.0;
                      //      const PhMDetStage& stage = stages_mct[stages_passed];
                      //      stagesum += stage.calcStageSum(ptr_mct, stage_indices_mct[stages_passed]);
			    const PhMDetStage& stage_1 = stages_phop[stages_passed];
			    double stagesum = stage_1.calcStageSum(ptr_phop, stage_indices_phop[stages_passed]);
                            confidence = stagesum - stage_1.threshold;
                            if (stagesum < stage_1.threshold)
                                break;
                            stages_passed += 1;
                        }
                        while (stages_passed != n_stages);
                        if (stages_passed == n_stages)//normal, out FP
			//if (stages_passed<n_stages)//abnormal, out TN
                        {
                            //std::cout<<"face detected"<<endl;
                            MCTDetection r;
#ifdef OKAPI_USE_BROKEN_RECT_TRANSFORM
                            r.box.x      = (x+1) * scale_x; // +1 to correct for MCT border
                            r.box.y      = (y+1) * scale_y;
                            r.box.width  = det_width  * scale_x;
                            r.box.height = det_height * scale_y;
#else
                            //r.box.x      = float((x +1 + 0.5) * scale_x - 0.5); // +1 to correct for MCT border
                            //r.box.y      = float((y +1 + 0.5) * scale_y - 0.5);
			    r.box.x 	   = float( (x + float(det_width-width)/2.0f + 1+ 0.5) * scale_x-0.5);
			    r.box.y	   = float( (y + float(det_height-height)/2.0f + 1+ 0.5) * scale_y-0.5);
			    //r.box.x      = float((x ) * scale_x);
			    //r.box.y      = float((y ) * scale_y);
                            //r.box.width  = float((det_width-1)  * scale_x + 1);
                           // r.box.height = float((det_height-1) * scale_y + 1);
			    r.box.width = float((width -1) * scale_x +1);
			    r.box.height = float((height -1) * scale_y +1);
#endif
                            r.conf = confidence;

                            //#ifdef _OPENMP
                            #pragma omp critical
                            //#endif
                            {
                                if (raw_detections)
                                {
                                    BinaryPatternRawDetection rdet;
                                    rdet.x = x;
                                    rdet.y = y;
                                    rdet.level = i;
                                    rdet.conf = confidence;
                                    raw_detections->push_back(rdet);
                                }

                                detections.push_back(r);
	                     }
                        }
			}
                        //ptr_mct  += pos_inc_mct;
			//for (int o_id=0; o_id<ptr_phop.size(); o_id++)
				ptr_phop += pos_inc_phop;

                    } // for x
                } // for y

                // delete stage indices

                for (int j = 0; j < n_stages; j++)
		{
		  //delete[] stage_indices_mct[j];
		  delete[] stage_indices_phop[j];
		  //delete[] stage_indices_o[j];
		}
            }
        } // for i
    }

    const PhMDetStage& PhMDetector::getStage_phop(size_t i) const
    {
        return stages_phop.at(i);
    }

    PhMDetStage& PhMDetector::getStage_phop(size_t i)
    {
        return stages_phop.at(i);
    }

    bool PhMDetector::hasInfo(const std::string& key) const
    {
        return info.find(key) != info.end();
    }

    string PhMDetector::getInfo(const std::string& key) const
    {
        map<string,string>::const_iterator it = info.find(key);
        if (it != info.end())
            return it->second;
        else
            return "";
    }

    void PhMDetector::setInfo(const std::string& key, const std::string& value)
    {
        info[key] = value;
    }
    

    void PhMDetector::orthogonal_trans()
	{
		RIP_theta = RIP_theta-90;
		//transform the location and look up tables
		for (size_t stage =0; stage < stages_phop.size(); stage ++)
		{
			//transform the locations
			set<CvPoint, CvPointCompare> features;
			for (size_t i=0; i<stages_phop[stage].classifiers.size(); i++)
			{
				int x = stages_phop[stage].classifiers[i].y;
				int y = win_width - stages_phop[stage].classifiers[i].x - 1;
				features.insert(cvPoint(x, y));
			}
			std::vector<PhM::PhMDetClassifier> rot_classifiers;
			for(set<CvPoint, CvPointCompare>::iterator it=features.begin(); it!=features.end(); ++it)
			{
				PhMDetClassifier c;
            			c.x = it->x;
            			c.y = it->y;
				int x0 = win_width - 1 - it->y;
				int y0 = it->x;
				for (size_t i=0; i<stages_phop[stage].classifiers.size(); i++)
				{
				if (stages_phop[stage].classifiers[i].x != x0 || stages_phop[stage].classifiers[i].y != y0)
                    			continue;
                		for (int j=0; j<table_sz; j++)
                    			{
						int h0 = j/125; int d0 = (j%125)/25; int v0 = (j%25)/5; int ds0 = j%5;
						int h1 = v0; if (h1==1 || h1==2) h1=3-h1;
						int d1 = ds0; if (d1==1 || d1==2) d1=3-d1;
						int v1=h0; 
						int ds1=d0; 
						int index1=h1*125+d1*25+v1*5+ds1;
						c.weights[index1] = stages_phop[stage].classifiers[i].weights[j];
					}
				break;
				}
				rot_classifiers.push_back(c);
			}
			stages_phop[stage].classifiers.clear();
			for (size_t i=0; i<rot_classifiers.size(); i++)
			stages_phop[stage].classifiers.push_back(rot_classifiers[i]);
		}
	}

 void PhMDetector::orthogonal_trans_p()
	{
		RIP_theta = RIP_theta+90;
		//transform the location and look up tables
		for (size_t stage =0; stage < stages_phop.size(); stage ++)
		{
			//transform the locations
			set<CvPoint, CvPointCompare> features;
			for (size_t i=0; i<stages_phop[stage].classifiers.size(); i++)
			{
				int y = stages_phop[stage].classifiers[i].x;
				int x = win_width - stages_phop[stage].classifiers[i].y - 1;
				features.insert(cvPoint(x, y));
			}
			std::vector<PhM::PhMDetClassifier> rot_classifiers;
			for(set<CvPoint, CvPointCompare>::iterator it=features.begin(); it!=features.end(); ++it)
			{
				PhMDetClassifier c;
            			c.x = it->x;
            			c.y = it->y;
				int x0 = it->y;
				int y0 = win_width - 1 - it->x;
				for (size_t i=0; i<stages_phop[stage].classifiers.size(); i++)
				{
				if (stages_phop[stage].classifiers[i].x != x0 || stages_phop[stage].classifiers[i].y != y0)
                    			continue;
                		for (int j=0; j<table_sz; j++)
                    			{
						int h0 = j/125; int d0 = (j%125)/25; int v0 = (j%25)/5; int ds0 = j%5;
						int h1 = v0; 
						int d1 = ds0; 
						int v1=h0; if (v1==1 || v1==2) v1=3-v1;
						int ds1=d0; if (ds1==1 || ds1==2) ds1=3-ds1;
						int index1=h1*125+d1*25+v1*5+ds1;
						c.weights[index1] = stages_phop[stage].classifiers[i].weights[j];
					}
				break;
				}
				rot_classifiers.push_back(c);
			}
			stages_phop[stage].classifiers.clear();
			for (size_t i=0; i<rot_classifiers.size(); i++)
			stages_phop[stage].classifiers.push_back(rot_classifiers[i]);
		}
	}

 void PhMDetector::diagonal_trans()
{
	RIP_theta = RIP_theta+45;
	win_width = (height+2) * abs(sin(RIP_theta*PI/180.0f)) + (width+2) * cos(RIP_theta*PI/180.0f)- 2;
	win_height = (height+2) * cos(RIP_theta*PI/180.0f) + (width+2) * abs(sin(RIP_theta*PI/180.0f))- 2;
		//transform the location and look up tables
		for (size_t stage =0; stage < stages_phop.size(); stage ++)
		{
			//transform the locations
			set<CvPoint, CvPointCompare> features;
			std::vector<PhM::PhMDetClassifier> rot_classifiers;
			for (size_t i=0; i<stages_phop[stage].classifiers.size(); i++)
			{
				float fy = cos(RIP_theta*PI/180.0f)*stages_phop[stage].classifiers[i].y+ sin(RIP_theta*PI/180.0f)*stages_phop[stage].classifiers[i].x;
				float fx = win_width/2.0f - sin(RIP_theta*PI/180.0f)*stages_phop[stage].classifiers[i].y + cos(RIP_theta*PI/180.0f)*stages_phop[stage].classifiers[i].x;
				int x = cvRound(fx);
				int y = cvRound(fy);
				features.insert(cvPoint(x, y));
				PhMDetClassifier c;
            			c.x = x;
            			c.y = y;
				for (int j=0; j<table_sz; j++)
                    			{
						int h0 = j/125; int d0 = (j%125)/25; int v0 = (j%25)/5; int ds0 = j%5;
						int h1 = d0; 
						int d1 = v0; 
						int v1=ds0; 
						int ds1=h0; if (ds1==1 || ds1==2) ds1=3-ds1;
						int index1=h1*125+d1*25+v1*5+ds1;
						c.weights[index1] = stages_phop[stage].classifiers[i].weights[j];
					}
				rot_classifiers.push_back(c);
			}
			stages_phop[stage].classifiers.clear();
			for(set<CvPoint, CvPointCompare>::iterator it=features.begin(); it!=features.end(); ++it)
			{
				PhMDetClassifier c;
            			c.x = it->x;
            			c.y = it->y;
				memset(c.weights, 0, sizeof(c.weights));
				for (size_t i=0; i<rot_classifiers.size(); i++)
				{
				if (rot_classifiers[i].x != c.x || rot_classifiers[i].y != c.y)
                    			continue;
 				for (size_t j=0; j<table_sz; ++j)
                    			c.weights[j] += rot_classifiers[i].weights[j];
				}
				stages_phop[stage].classifiers.push_back(c);
			}
		}

}

void PhMDetector::mirror_trans()
{
		//transform the location and look up tables
		for (size_t stage =0; stage < stages_phop.size(); stage ++)
		{
			//transform the locations
			set<CvPoint, CvPointCompare> features;
			std::vector<PhM::PhMDetClassifier> rot_classifiers;
			for (size_t i=0; i<stages_phop[stage].classifiers.size(); i++)
			{
				int x = win_width-1-stages_phop[stage].classifiers[i].x;
				int y = stages_phop[stage].classifiers[i].y;
				features.insert(cvPoint(x, y));
				PhMDetClassifier c;
            			c.x = x;
            			c.y = y;
				for (int j=0; j<table_sz; j++)
                    			{
						int h0 = j/125; int d0 = (j%125)/25; int v0 = (j%25)/5; int ds0 = j%5;
						int h1 = h0; //if (h1==1 || h1==2) h1=3-h1;
						int d1 = ds0; if (d1==1 || d1==2) d1=3-d1;
						int v1=v0; if (v1==1 || v1==2) v1=3-v1;
						int ds1=d0;if (ds1==1 || ds1==2) ds1=3-ds1;
						int index1=h1*125+d1*25+v1*5+ds1;
						c.weights[index1] = stages_phop[stage].classifiers[i].weights[j];
					}
				rot_classifiers.push_back(c);
			}
			stages_phop[stage].classifiers.clear();
			for(set<CvPoint, CvPointCompare>::iterator it=features.begin(); it!=features.end(); ++it)
			{
				PhMDetClassifier c;
            			c.x = it->x;
            			c.y = it->y;
				memset(c.weights, 0, sizeof(c.weights));
				for (size_t i=0; i<rot_classifiers.size(); i++)
				{
				if (rot_classifiers[i].x != c.x || rot_classifiers[i].y != c.y)
                    			continue;
 				for (size_t j=0; j<table_sz; ++j)
                    			c.weights[j] += rot_classifiers[i].weights[j];
				}
				stages_phop[stage].classifiers.push_back(c);
			}
		}

}

	void draw_detections(cv::Mat& img, std::vector<MCTDetection>& gop_dets, std::string& file_name, std::string& foder_path)
	{
		int img_h=img.rows;
    		int img_w=img.cols;
		cv::Mat img_show=img.clone();
    		int win_num=gop_dets.size();
    		for (int i=0; i<win_num; i++)
     	{	
         	if (gop_dets[i].box.x+gop_dets[i].box.width<img_w && gop_dets[i].box.y+gop_dets[i].box.height<img_h)
        	{	
			cv::Rect box_now=gop_dets[i].box;
			img_show(cv::Range(box_now.y, box_now.y+1), cv::Range(box_now.x, box_now.x+box_now.width))=cv::min(img_show(cv::Range(box_now.y, box_now.y+1), cv::Range(box_now.x, box_now.x+box_now.width)),cv::Mat(1,box_now.width, CV_8UC1, cv::Scalar(0.0,0.0,0.0,0.0)));
			img_show(cv::Range(box_now.y, box_now.y+box_now.height), cv::Range(box_now.x, box_now.x+1))=cv::min(img_show(cv::Range(box_now.y, box_now.y+box_now.height), cv::Range(box_now.x, box_now.x+1)),cv::Mat(box_now.height,1,CV_8UC1, cv::Scalar(0.0,0.0,0.0,0.0)));
            		img_show(cv::Range(box_now.y+box_now.height-1, box_now.y+box_now.height), cv::Range(box_now.x, box_now.x+box_now.width))=cv::min(img_show(cv::Range(box_now.y+box_now.height-1, box_now.y+box_now.height), cv::Range(box_now.x, box_now.x+box_now.width)),cv::Mat(1,box_now.width, CV_8UC1, cv::Scalar(0.0,0.0,0.0,0.0)));
            		img_show(cv::Range(box_now.y, box_now.y+box_now.height), cv::Range(box_now.x+box_now.width-1, box_now.x+box_now.width))=cv::min(img_show(cv::Range(box_now.y, box_now.y+box_now.height), cv::Range(box_now.x+box_now.width-1, box_now.x+box_now.width)), cv::Mat(box_now.height,1,CV_8UC1, cv::Scalar(0.0,0.0,0.0,0.0)));
        	}
      	}
            char file_path[200];
	    sprintf(file_path, "%s/dets_%s.jpg", foder_path.c_str(),file_name.c_str());
	    okapi::saveImage(file_path, img_show);
	}

    // MCTPyramid ///////////////////////////////////////////////////////////////////////

    PhMPyramid::PhMPyramid()
    {
      rp=RP_transform();
        // nothing to do
    }

    PhMPyramid::PhMPyramid(const cv::Mat& image, cv::Size det_size,
                           double scale_factor, int min_result_width, int max_result_width)
    {
        // Automatically determine values
        if (min_result_width == -1)
            min_result_width = det_size.width;
        if (max_result_width == -1)
            max_result_width = std::max(min_result_width, image.cols);

        // Check input
        if (image.empty())
            throw InvalidArgumentException("Empty cv::Mat");
        if (image.depth() != CV_8U)
            throw InvalidArgumentException("Expected input image depth CV_8U.");
        if (det_size.width <= 0 || det_size.height <= 0)
            throw InvalidArgumentException("det_size must have positive dimensions.");
        if (scale_factor <= 1)
            throw InvalidArgumentException("scale_factor must be > 1.");
        if (min_result_width > max_result_width)
            throw InvalidArgumentException("min_result_width must be <= max_result_width");
        if (min_result_width == 0)
            throw InvalidArgumentException("min_result_width cannot be 0");

        // Store image
        if (image.channels() != 1)
            cv::cvtColor(image, input_image, CV_RGB2GRAY);
        else
            input_image = image;

        // Convert to scale factors
        double scale_min = min_result_width / double(det_size.width);
        double scale_max = max_result_width / double(det_size.width);

        // Make sure the maximum sized results fit into the image
        if (image.cols / scale_max < det_size.width  + 2)
            scale_max = image.cols / double(det_size.width  + 2);
        if (image.rows / scale_max < det_size.height + 2)
            scale_max = image.rows / double(det_size.height + 2);

        // Compute pyramid scales
        setupLevels(scale_factor, scale_min, scale_max);
    }

    PhMPyramid::PhMPyramid(cv::Size image_size, cv::Size det_size,
                           double scale_factor, int min_result_width, int max_result_width)
    {
        // Automatically determine values
        if (min_result_width == -1)
            min_result_width = det_size.width;
        if (max_result_width == -1)
            max_result_width = std::max(min_result_width, image_size.width);

        // Check input
      /*  if (image.empty())
            throw InvalidArgumentException("Empty cv::Mat");
        if (image.depth() != CV_8U)
            throw InvalidArgumentException("Expected input image depth CV_8U.");
        if (det_size.width <= 0 || det_size.height <= 0)
            throw InvalidArgumentException("det_size must have positive dimensions.");
        if (scale_factor <= 1)
            throw InvalidArgumentException("scale_factor must be > 1.");
        if (min_result_width > max_result_width)
            throw InvalidArgumentException("min_result_width must be <= max_result_width");
        if (min_result_width == 0)
            throw InvalidArgumentException("min_result_width cannot be 0");

        // Store image
        if (image.channels() != 1)
            cv::cvtColor(image, input_image, CV_RGB2GRAY);
        else
            input_image = image;*/
        if (det_size.width <= 0 || det_size.height <= 0)
            throw InvalidArgumentException("det_size must have positive dimensions.");
        if (scale_factor <= 1)
            throw InvalidArgumentException("scale_factor must be > 1.");
        if (min_result_width > max_result_width)
            throw InvalidArgumentException("min_result_width must be <= max_result_width");
        if (min_result_width == 0)
            throw InvalidArgumentException("min_result_width cannot be 0");
        // Convert to scale factors
        double scale_min = min_result_width / double(det_size.width);
        double scale_max = max_result_width / double(det_size.width);

        // Make sure the maximum sized results fit into the image
        if (image_size.width / scale_max < det_size.width  + 2)
            scale_max = image_size.width / double(det_size.width  + 2);
        if (image_size.height / scale_max < det_size.height + 2)
            scale_max = image_size.height / double(det_size.height + 2);

        // Compute pyramid scales
        setupLevels(scale_factor, scale_min, scale_max);
		//initialize the HVoMCTs
//mutex.lock();
		//for (int i=0; i< scales.size(); i++)
		//	{
		//	HVoMCT_transform *tem_trans=new HVoMCT_transform(getSize(i, image_size).width, getSize(i, image_size).height,phop.get_omega(),phop.get_so_prod(), phop.get_offset());
		//	tem_trans->allocate_memo();
		//	HVoMCTs.push_back(tem_trans);		
		//	}
//mutex.unlock();
		//for (int i=0; i< HVoMCTs.size(); i++)
		//	HVoMCTs[i]->allocate_memo();
	//initialize the HVoMCT pyramid
	RP_pyr = new RP_pyramid(image_size, scales);
    }

    PhMPyramid::PhMPyramid(const cv::Mat& image, double scale_factor,
                           double scale_min, double scale_max)
    {
        if (image.depth() != CV_8U)
            throw InvalidArgumentException("Expected input image depth CV_8U.");
        if (image.channels() != 1)
            cv::cvtColor(image, input_image, CV_RGB2GRAY);
        else
            input_image = image;
        if (scale_factor < 1)
            throw InvalidArgumentException("Expected scale_factor >= 1");

        setupLevels(scale_factor, scale_min, scale_max);
    }

    void PhMPyramid::setupLevels(double scale_factor, double scale_min, double scale_max)
    {
        // setup scales array

        scales.clear();
        if (scale_max < scale_min)
            return;
        double fac = scale_min;
        while (fac <= scale_max)
        {
            if (fabs(fac - 1.0) < 0.05) // no resizing for factor of about 1.0
                scales.push_back(1.0);
            else
                scales.push_back(fac);
            fac *= scale_factor;
        }

        // initialize image cache

        gray_imgs.clear();
        //mct_imgs.clear();
	phop_imgs.clear();
        for (unsigned i = 0; i < scales.size(); i++)
        {
            gray_imgs.push_back(cv::Mat());
            //mct_imgs.push_back(cv::Mat());
	    //std::vector<cv::Mat> MG_img_now;
	    //phop_imgs.push_back(MG_img_now);
	    phop_imgs.push_back(cv::Mat());
        }
    }

	void PhMPyramid::build(const cv::Mat& img, bool progressive, bool keep_gray)
    {
    // Check input
        if (img.empty())
            throw InvalidArgumentException("Empty cv::Mat");
        if (img.depth() != CV_8U)
            throw InvalidArgumentException("Expected input image depth CV_8U.");
        // Store image
        if (img.channels() != 1)
            cv::cvtColor(img, input_image, CV_RGB2GRAY);
        else
            input_image = img;
	RP_pyr->build(input_image);
        //cv::Mat gray;
        for (unsigned i = 0; i < scales.size(); i++)
        {	
	    	phop_imgs[i] = RP_pyr->get_RP(i);
        }
    }
    void PhMPyramid::build(bool progressive, bool keep_gray)
    {
        cv::Mat gray;
        for (unsigned i = 0; i < scales.size(); i++)
        {
            if (scales[i] == 1.0)
            {
                // no scaling
                gray = input_image;
            }
            else
            {
                // tries to use smaller version of input image for downscaling
                cv::Mat source = input_image;
                if (progressive && i > 0 && scales[i-1] > 1.0)
                    source = gray;
                cv::Size target_size = getSize(i);
                bool scale_up = source.cols < target_size.width;
                cv::resize(source, gray, target_size, 0, 0, scale_up ? CV_INTER_LINEAR : CV_INTER_AREA);
            }
            //mct_imgs[i] = mct.transform(gray);
	    	//phop_imgs[i] = phop.HVoMCT_CUDA(gray);
		phop_imgs[i] = rp.RP_trans(gray);
            if (keep_gray)
                gray_imgs[i] = gray;
        }
    }

    PhMPyramid::~PhMPyramid()
    {
        // nothing to do
	//if (HVoMCT_pyr)
	delete RP_pyr;
    }

    int PhMPyramid::getLevels() const
    {
        return scales.size();
    }

    double PhMPyramid::getScale(int level) const
    {
        return scales[level];
    }

    cv::Size PhMPyramid::getSize(int level) const
    {
        return cv::Size(cvFloor(input_image.cols / scales[level]),
                        cvFloor(input_image.rows / scales[level]));
    }

    cv::Size PhMPyramid::getSize(int level, cv::Size img_sz) const
    {
        return cv::Size(cvFloor(img_sz.width / scales[level]),
                        cvFloor(img_sz.height / scales[level]));
    }

    int PhMPyramid::findClosestLevel(double desired_scale) const
    {
        int    best_level = -1;
        double best_error = 9999999.9;
        for (unsigned i = 0; i < scales.size(); i++)
        {
            double err = fabs(desired_scale - scales[i]);
            if (err < best_error)
            {
                best_level = i;
                best_error = err;
            }
            else
            {
                // once best_error gets worse, it won't get better again
                break;
            }
        }
        return best_level;
    }

    cv::Mat PhMPyramid::getGrayImage(int level) const
    {
        if (level < 0 || level >= getLevels())
            throw InvalidArgumentException("Invalid level argument");

        // return cached image (if available)
        if (!gray_imgs[level].empty())
            return gray_imgs[level];

        // generate image on the fly
        if (scales[level] == 1.0)
            return input_image; // no scaling required
        else
        {
            cv::Mat result;
            bool scale_up = input_image.cols < getSize(level).width;
            cv::resize(input_image, result, getSize(level), 0, 0, scale_up ? CV_INTER_LINEAR : CV_INTER_AREA);
            return result;
        }
    }

    cv::Mat PhMPyramid::getPhOPImage(int level) 
    {
        if (level < 0 || level >= getLevels())
            throw InvalidArgumentException("Invalid level argument");

        // return cached mct image (if available)
        if (!phop_imgs[level].empty())
            return phop_imgs[level];

        // generate mct image on the fly
        	//phop_imgs[level]=phop.HVoMCT_CUDA(getGrayImage(level));
		phop_imgs[level]=rp.RP_trans(getGrayImage(level));
	return phop_imgs[level];
    }

    cv::Mat PhMPyramid::getInputImage() const
    {
        return input_image;
    }
}
