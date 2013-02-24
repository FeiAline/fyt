#include <okapi.hpp>
#include <okapi-gui.hpp>
#include "PhMdetector.hpp"
#include "GRNN.hpp"

#include <fstream>

using namespace okapi;
using namespace std;

// Detector parameters
float  scale_step           = 1.2f;
size_t shift_step           = 1;
double upscale_factor       = 1.0;
bool   progressive          = false;

// Clustering
int    min_neighbours       = 3;
double neighbour_dist       = 1.4;

// GUI
bool   show_gui             = true;
bool   draw_raw_detections  = true;

// Results
bool   write_results        = false;
bool   show_color           = false;

vector<PhM::PhMDetector *>  mcdets;

static const char help_str[] =
    "\nOkapi detector demonstration\n"
    "\nOptions:\n%s\n";

vector<MCTDetection> mct_detections(cv::Mat img, PhM::pose_estimator& my_est, vector<cv::Rect> *raw_detections=NULL)//
{
	vector<float> ests;
    cv::Mat gray = cvtColorGray(img);

    vector<MCTDetection> mct_dets;
    vector<MCTDetection> mct_dets_clustered;
    mct_dets.clear();
    int min_w = cvRound(mcdets[0]->getSize().width / upscale_factor);
    //OKAPI_TIMER_START("MCTPyramid");
    PhM::PhMPyramid pyramid(gray.size(), mcdets[0]->getSize(), scale_step, min_w, -1);
    pyramid.build(gray);
    //PhM::PhMPyramid pyramid(gray.size(), phm_detector->getSize(), scale_step, min_w, -1, phm_detector->getPhOP());
    //pyramid.build(gray,progressive);
    //OKAPI_TIMER_STOP("MCTPyramid");
    //OKAPI_TIMER_START("MCTscan");
 	for (size_t i=0; i<mcdets.size(); i++)  
 		{
		mcdets[i]->scanPyramid(pyramid, mct_dets, shift_step); //total, scan,
		}
    //OKAPI_TIMER_STOP("MCTscan");
    //OKAPI_TIMER_START("MCTcluster");
    mct_dets_clustered = clusterDetections(mct_dets, min_neighbours, neighbour_dist);
    //OKAPI_TIMER_STOP("MCTcluster");

    if (raw_detections)
    {
        raw_detections->clear();
        for (size_t j=0; j<mct_dets.size(); ++j)
            raw_detections->push_back(mct_dets[j].box);
    }
	//only return the max confidence detection
	if (mct_dets_clustered.size()>1)
	{
	float maxconf=-99999;
	MCTDetection maxdet;
	for (int i=0; i<mct_dets_clustered.size(); i++)
	{
		if (mct_dets_clustered[i].conf>maxconf)
		{
		maxconf = mct_dets_clustered[i].conf;
		maxdet = mct_dets_clustered[i];
		}
	}
	mct_dets_clustered.clear();
	mct_dets_clustered.push_back(maxdet);
	}

	for (unsigned j = 0; j < mct_dets_clustered.size(); j++)
	{
		double scalenow=mct_dets_clustered[j].box.width/mcdets[0]->getSize().width;
		int levelnow=pyramid.findClosestLevel(scalenow);
		cv::Mat G_ene=pyramid.RP_pyr->get_ene(levelnow);
		double scalepyr=pyramid.getScale(levelnow);
		float est=my_est.get_pose(G_ene, mct_dets_clustered[j], scalepyr); 
		ests.push_back(est);
		mct_dets_clustered[j].conf=est;
	}
	
	
    return mct_dets_clustered;
}
void readimagelist(string& file, vector<string>& list)
{
	list.clear();
	ifstream listfile(file.c_str());
	string pre_fn=" ";
	while (!listfile.eof())
	{
	string str;
	getline(listfile, str);
	stringstream ss (stringstream::in | stringstream::out);
	ss<< str;
	char filename[200];
	ss>>filename;
	string strf(filename);
	if (strf.compare(pre_fn)!=0)
	{
	pre_fn = strf;
	list.push_back(strf);
	}
	}
}

int main(int argc, char* argv[])
{
    LibraryInitializer li;
    //ScopedTimer st("main");

    // parse arguments
    CommandLineOptions opt;
    opt.addOption('d', "mctdetector",   true,  "MCT detector file");
    opt.addOption('v', "RIPdetector",   true,  "RIP detector file [optional]");
    opt.addOption('o', "output",        true,  "Output file for detected bounding boxes and confidences");
    opt.addOption('f', "folder-dets",   true,  "folder to save the images with bounding box");
    opt.addOption('m', "mirror",        false, "Mirror the detector around the vertical axis");
    opt.addOption('s', "scale-step",    true,  strprintf("Scale step for MCT pyramid [default: %f]", scale_step));
    opt.addOption(     "shift",         true,  strprintf("Pixel shift of the detector window [default: %d]", (int)shift_step));
    opt.addOption('u', "upscale",       true,  strprintf("Upscale factor for source images [default: %f]", upscale_factor));
    opt.addOption('p', "progressive",   false, "Build MCT Pyramid progressively [use only with scale_step >= 1.2]");
    opt.addOption('n', "neighbors",     true,  strprintf("Number of neighbors for MCT clustering [default: %d]", min_neighbours));
    opt.addOption(     "ndist",         true,  strprintf("Neighbor distance for clustering [default: %d]", neighbour_dist));
    opt.addOption(     "gui",           false, "Display detections in a gui");
    opt.addOption('r', "raw",           false, "Additionally draw raw (unclustered) detections");
    opt.addOption(     "write-results", false, "Write result images next to originals");
    opt.addOption('l', "last-stage",    false, "Report only the stage-sum of the last stage as confidence (the default is to report the whole sum)");
    opt.addOption(     "color",         false, "Show color images");
    opt.addOption('i', "image-list",    true,  "Image lists");

    vector<string> image_files;
    string mctdet_fn, ripdet_fn, output_fn, folder_dets, imglist;
    bool mirror_detector, last_stage_conf;
    try
    {
        // Files
        image_files         = opt.parse(argc, argv);
        mctdet_fn           = opt.getArgument<string>("mctdetector", "");
	ripdet_fn 	    = opt.getArgument<string>("RIPdetector", "");
        output_fn           = opt.getArgument<string>("output", "");
	folder_dets	    = opt.getArgument<string>("folder-dets","");
        mirror_detector     = opt.parameterSet("mirror") > 0;
	imglist		    = opt.getArgument<string>("image-list", "");

        // Detector parameters
        scale_step          = opt.getArgument<float>("scale-step", scale_step);
        shift_step          = opt.getArgument<size_t>("shift", shift_step);
        upscale_factor      = opt.getArgument<double>("upscale", 1.0);
        progressive         = opt.parameterSet("progressive") > 0;
        min_neighbours      = opt.getArgument<int>("neighbors", min_neighbours);
        neighbour_dist      = opt.getArgument<double>("ndist", neighbour_dist);

        // GUI
        show_gui            = opt.parameterSet("gui");
        draw_raw_detections = opt.parameterSet("raw");

        // Results
        write_results       = opt.parameterSet("write-results") > 0;
        show_color          = opt.parameterSet("color") > 0;
        last_stage_conf     = opt.parameterSet("last-stage") > 0;
    }
    catch (CommandLineOptionException e)
    {
        printf(help_str, opt.helpString().c_str());
        exit(1);
    }

printf("loading image names\n");
	//load image list
	readimagelist(imglist,image_files);
printf("loaded image names\n");
    if (image_files.empty())
    {
        printf("Please pass one or more image filenames\n");
        exit(1);
    }

    if (progressive && scale_step < 1.2)
        printf("WARNING: Building MCT pyramid progressively with scale_step < 1.2 may decrease\n"
                "detection performance!\n");

    if (mctdet_fn.empty())
    {
        printf("Please pass a detector file using -d\n");
        exit(1);
    }

    // load detector
    if (mctdet_fn.length() > 0)
    	{
		PhM::PhMDetector * mcdet;
        	try
        	{
        	    mcdet =new PhM::PhMDetector(mctdet_fn);
		    mcdets.push_back(mcdet);
        	}
        	catch (Exception& e)
        	{
        	    printf("%s\n", e.what());
        	    exit(1);
        	}
        	printf("GaborDetector loaded, size: %d x %d\n",
        	        mcdet->getSize().width, mcdet->getSize().height);
    	}
	/*if (mctdet_fn.length() > 0)
    	{
		PhM::PhMDetector * mcdet;
        	try
        	{
        	    mcdet =new PhM::PhMDetector(mctdet_fn);
		    mcdet->diagonal_trans();
		    mcdets.push_back(mcdet);
        	}
        	catch (Exception& e)
        	{
        	    printf("%s\n", e.what());
        	    exit(1);
        	}
        	printf("RIP Detector loaded, size: %d x %d\n",
        	        mcdet->getSize().width, mcdet->getSize().height);
    	}

	if (mctdet_fn.length() > 0)
    	{
		PhM::PhMDetector * mcdet;
        	try
        	{
        	    mcdet =new PhM::PhMDetector(mctdet_fn);
		    mcdet->diagonal_trans();
		    mcdet->orthogonal_trans();
		    mcdets.push_back(mcdet);
        	}
        	catch (Exception& e)
        	{
        	    printf("%s\n", e.what());
        	    exit(1);
        	}
        	printf("RIP Detector loaded, size: %d x %d\n",
        	        mcdet->getSize().width, mcdet->getSize().height);
    	}*/


	if (ripdet_fn.length() > 0)
    	{
		PhM::PhMDetector * mcdet;
        	try
        	{
        	    mcdet =new PhM::PhMDetector(ripdet_fn);
		    mcdets.push_back(mcdet);
        	}
        	catch (Exception& e)
        	{
        	    printf("%s\n", e.what());
        	    exit(1);
        	}
        	printf("Profile Detector loaded, size: %d x %d\n",
        	        mcdet->getSize().width, mcdet->getSize().height);
    	}
	if (ripdet_fn.length() > 0)
    	{
		PhM::PhMDetector * mcdet;
        	try
        	{
        	    mcdet =new PhM::PhMDetector(ripdet_fn);
		    mcdet ->mirror_trans();
		    mcdets.push_back(mcdet);
        	}
        	catch (Exception& e)
        	{
        	    printf("%s\n", e.what());
        	    exit(1);
        	}
        	printf("Profile Detector loaded, size: %d x %d\n",
        	        mcdet->getSize().width, mcdet->getSize().height);
    	}

	string file_X("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/bin_X_det_datas_sub2_24_ext27.7_sz28_facepix_gabor_forC_b10.txt");
	string file_y("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/bin_y_det_datas_sub2_24_ext27.7_sz28_facepix_gabor_forC_b10.txt");
	string file_b("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/bin_b_det_datas_sub2_24_ext27.7_sz28_facepix_gabor_forC_b10.txt");
	PhM::pose_estimator my_est(file_X, file_y, file_b, 1.71f, 60);

    vector<string> names(image_files.size());
    vector<vector<MCTDetection> > detections(image_files.size());
    vector<cv::Mat> gui_images(image_files.size());

    ProgressBar bar(image_files.size(), false, cerr);

//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
    for (int i=0; i<(int)image_files.size(); ++i)
    {
        cv::Mat img;
        try
        {
		char filepath[200];
		sprintf(filepath, "/home/fjjiang/testing_data/Pose_multi_PIE/%s", image_files[i].c_str());
            if (show_color)
                img = loadImageRGB(filepath);
            else
                img = loadImageGray(filepath);
        }
	
        catch (Exception& e)
        {
            printf("%s\n", e.what());
            exit(1);
        }

        string name = filename_strip(image_files[i]);
        names[i] = name;

        vector<cv::Rect> raw_detections;
        detections[i] = mct_detections(img, my_est, draw_raw_detections ? &raw_detections : 0);

        if (show_gui)
        {
            gui_images[i] = cvtColorRGB(img);

            // draw detections
            ImageDeco deco(gui_images[i]);
            if (draw_raw_detections)
            {
                deco.setThickness(1);
                deco.setColor(55, 50, 250);
                for (size_t j=0; j<raw_detections.size(); ++j)
                    deco.drawRect(raw_detections[j]);
            }
            deco.setThickness(2);
            for (size_t j=0; j<detections[i].size(); ++j) {
                deco.setColor(255, 50, 50);
                deco.drawRect(detections[i][j].box);
                deco.setColor(255, 255, 255);
                deco.drawTextDark(stringify(detections[i][j].conf),
                                  detections[i][j].box.x,
                                  detections[i][j].box.y,
                                  0.2);
            }
        }

//#ifdef _OPENMP
//#pragma omp critical
//#endif
        ++bar;
    }

    // Output MCT detections
    ofstream outf;
    ostream* out = &outf;
    if (output_fn.empty() || output_fn == "-")
        out = &cout;
    else
        outf.open(output_fn.c_str());

    if (!out)
        throw IOException("Error writing to " + output_fn);
    for (size_t i=0; i<detections.size(); ++i)
        for (size_t j=0; j<detections[i].size(); ++j)
        {
            const MCTDetection& det = detections[i][j];
            *out << names[i] << " " << det.box.x << " " << det.box.y << " "
                             << det.box.width << " " << det.box.height << " "
                             << det.conf << endl;
        }

    // Display results
    if (show_gui)
    {
        GuiThread thread;
        ImageWindow* imgwin = new ImageWindow("Detections");
        thread.addClient(imgwin);
        thread.start();
        for (size_t i=0; i<gui_images.size(); ++i)
        {
            imgwin->setImage(names[i], gui_images[i]);
            if (write_results)
            {
                string fn = image_files[i] + ".results.png";
                saveImage(fn, gui_images[i]);
            }
        }
        while (imgwin->getWindowState())
            sleep_ms(100);
    }
}
