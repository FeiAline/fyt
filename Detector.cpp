#include "PhMdetector.hpp"
//#include "GRNN.hpp"
#include <okapi.hpp>
#include <okapi-gui.hpp>
#include <okapi-videoio.hpp>



using namespace okapi;
using namespace std;

// scan parameters
float  scale_step           = 1.2f;
int    min_width            = -1;
double upscale_factor       = 1.0;
int    max_width            = -1;
int    shift_step           = 1;
int    min_neighbours       = 3;
double neighbour_dist       = 1.2;
bool   draw_raw_detections  = true;
bool   progressive          = false;
bool   display_fps          = false;

auto_ptr<PhM::PhMDetector>  mcdet;
//auto_ptr<FaceDetector> vjdet;

static const char help_str[] =
    "\nOkapi detector demonstration\n"
    "\nOptions:\n%s\n";

cv::Mat run_detectors(const cv::Mat& img, PhM::PhMPyramid& pyramid)
{
    cv::Mat gray;
    if (img.channels() == 1)
        gray = img;
    else
        cv::cvtColor(img, gray, CV_RGB2GRAY);

	/*CUDA_mat rot_mat(640, 480, PI/4.0f);
	CUDA_mat ori_mat(640, 480);
	cv::Mat f_gray;
	gray.convertTo(f_gray, CV_32FC1);
	ori_mat.feedin_gray(f_gray.ptr<float>(0));
	rot_mat.tobe_rotated(ori_mat.get_data(), 640, 480);
	int r_w, r_h;
	rot_mat.get_size(r_w, r_h);
	cv::Mat rot_gray(r_h, r_w, CV_32FC1);
	rot_mat.get_mat(rot_gray.ptr<float>(0));
	cv::Mat unsig_rot;
	rot_gray.convertTo(unsig_rot, CV_8UC1);*/
	
    vector<MCTDetection> mct_dets;
    vector<MCTDetection> mct_dets_clustered;
    if (mcdet.get())
    {
        mct_dets.clear();
        //int min_w = cvRound(mcdet->getSize().width / upscale_factor);
        OKAPI_TIMER_START("MCTPyramid");
        //PhM::PhMPyramid pyramid(gray, mcdet->getSize(), scale_step, min_w, max_width, mcdet->getPhOP());
		pyramid.build(gray, progressive);
        OKAPI_TIMER_STOP("MCTPyramid");
        OKAPI_TIMER_START("MCTscan");
        mcdet->scanPyramid(pyramid, mct_dets, shift_step);
        OKAPI_TIMER_STOP("MCTscan");
        OKAPI_TIMER_START("MCTcluster");
        mct_dets_clustered = clusterDetections(mct_dets, min_neighbours, neighbour_dist);
        OKAPI_TIMER_STOP("MCTcluster");
    }

    // V&J scan

    /*vector<cv::Rect> vj_dets;
    if (vjdet.get())
    {
        vjdet->setScaleFactor(scale_step);
        vj_dets.clear();
        OKAPI_TIMER_START("Viola&Jones");
        vj_dets = vjdet->detectFaces(gray);
        OKAPI_TIMER_STOP("Viola&Jones");
    }*/

    // draw detection rectangles

    cv::Mat image;
    if (img.channels() != 3)
        cv::cvtColor(img, image, CV_GRAY2RGB);
    else
        image = img;
//	cv::cvtColor(unsig_rot, image, CV_GRAY2RGB);


    ImageDeco deco(image);
    deco.createLayer(5, cvScalar(0,0,0));
    deco.setThickness(1);
    deco.setColor(55,50,250);
    if (draw_raw_detections)
    {
        for (unsigned j = 0; j < mct_dets.size(); j++)
            deco.drawRect(mct_dets[j].box);
    }
//	printf("%d ", mct_dets.size());
    deco.createLayer(5, cvScalar(0,0,0));
    deco.setThickness(2);
    deco.setColor(255,50,50);
    for (unsigned j = 0; j < mct_dets_clustered.size(); j++)
    {
        MCTDetection& det = mct_dets_clustered[j];
        deco.drawRect(det.box);
        deco.drawText(stringify(det.conf), det.box.x + 2, det.box.y + 15);
    }

   /* deco.createLayer(5, cvScalar(0,0,0));
    deco.setColor(50,255,50);
    for (unsigned j = 0; j < vj_dets.size(); j++)
    {
        cv::Rect& r = vj_dets[j];
        deco.drawRect(r);
    }*/

    return deco.mergeLayers();
}

int main(int argc, char* argv[])
{
    LibraryInitializer li;
    //OKAPI_LOG_CONFIGURE_DEFAULT();

    // parse arguments

    CommandLineOptions opt;
    opt.addOption('d', "mctdetector",   true,  "mctdetector file [optional]");
    //opt.addOption('v', "vjdetector",    true,  "OpenCV detector file [optional]");
    opt.addOption('n', "neighbors",     true,  "Number of neighbors for MCT clustering [optional]");
    opt.addOption('r', "no-raw",        false, "Do not draw raw detections [optional]");
    opt.addOption('u', "upscale",       true,  "Upscale factor for source images [optional]");
    opt.addOption('s', "scale-step",    true,  "Scale step for MCT pyramid");
    opt.addOption('p', "progressive",   false, "Build MCT pyramid progressively");
    opt.addOption(     "shift",         true,  "Pixel shift of the detector window");
    opt.addOption(     "ndist",         true,  "Neighbor distance for clustering");
    opt.addOption(     "fps",           false, "Display frames per second");
    if (argc < 3)
    {
        printf(help_str, opt.helpString().c_str());
        exit(1);
    }

    string mctdet_fn, vjdet_fn;
    try
    {
        opt.parse(argc, argv);
        mctdet_fn = opt.getArgument<string>("mctdetector", "");
      //  vjdet_fn  = opt.getArgument<string>("vjdetector", "");
        min_neighbours = opt.getArgument<int>("neighbors", min_neighbours);
        draw_raw_detections = !opt.parameterSet("no-raw");
        upscale_factor = opt.getArgument<double>("upscale", 1.0);
        scale_step = opt.getArgument<float>("scale-step", scale_step);
        progressive = opt.parameterSet("progressive") > 0;
        shift_step = opt.getArgument<size_t>("shift", shift_step);
        neighbour_dist = opt.getArgument<double>("ndist", neighbour_dist);
        display_fps = opt.parameterSet("fps") > 0;
    }
    catch (CommandLineOptionException e)
    {
        printf(help_str, opt.helpString().c_str());
        exit(1);
    }

    if (progressive && scale_step < 1.2)
        printf("WARNING: Building MCT pyramid progressively with scale_step < 1.2 may decrease\n"
                "detection performance!\n");

    // load detectors

    if (mctdet_fn.length() > 0)
    {
        try
        {
            mcdet.reset(new PhM::PhMDetector(mctdet_fn));
        }
        catch (Exception& e)
        {
            printf("%s\n", e.what());
            exit(1);
        }
        printf("MCTDetector loaded, size: %d x %d\n",
                mcdet->getSize().width, mcdet->getSize().height);
    }

    /*if (vjdet_fn.length() > 0)
    {
        try
        {
            vjdet.reset(new FaceDetector(vjdet_fn));
        }
        catch (Exception& e)
        {
            printf("%s\n", e.what());
            exit(1);
        }
        vjdet->setDownscale(false);
        vjdet->setUseCanny(false);
        printf("V&J Detector loaded\n");
    }*/

    VideoSource* src = createDefaultCamera(640, 480);

    GuiThread thread;
    ImageWindow* imgwin = new ImageWindow("Detector Demo (cam)");
    thread.addClient(imgwin);
    thread.start();

    double old_ts = -1;
    double fps = -1;
	int min_w = cvRound(mcdet->getSize().width / upscale_factor);
	min_w = 50;
	max_width = 150;
	PhM::PhMPyramid pyramid(cv::Size(640, 480), mcdet->getSize(), scale_step, min_w, max_width);
	int id=0;
	        printf("Pyramid initialize done\n");
        //cv::Mat img = run_detectors(src->getImage().clone(), pyramid);
	//	saveImage("/home/fjjiang/dets.bmp", img);
    while (imgwin->getWindowState() && src->getNextFrame())
    {
        cv::Mat img = run_detectors(src->getImage().clone(), pyramid);
		//saveImage("/home/fjjiang/dets.bmp", img);
        if (display_fps)
        {
            double new_ts = src->getTimestamp();

            if (old_ts < 0)
                old_ts = new_ts - 0.1;

            // Compute fps
            double new_fps = 1 / (new_ts - old_ts);
            if (fps < 0)
                fps = new_fps;
            else
                fps = 0.95 * fps + 0.05 * new_fps;

            char buf[1000];
            sprintf(buf, "%4.1f fps\n", fps);
            old_ts = new_ts;

            // Draw directly on the cam image :)
            ImageDeco deco(img);
            deco.setAntiAlias(true);
            deco.drawText(buf, 0, 15);
        }

        imgwin->setImage("Cam", img);
    }

    // finished

    printf("Bye.\n");
    return 0;
}
