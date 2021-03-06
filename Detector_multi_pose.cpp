#include "PhMdetector.hpp"
#include "GRNN.hpp"
#include "heat.hpp"
#include "face.hpp"
#include "boost.hpp"
#include "faceLocator.hpp"
#include "distance.hpp"
#include "FeatureCollector.hpp"
#include <okapi.hpp>
#include <okapi-gui.hpp>
#include <okapi-videoio.hpp>
#include <highgui/highgui.hpp>
#include <core/core.hpp>
#include <iostream>
#include <fstream>

using namespace okapi;
using namespace std;

#define SPEAKER_BASED 1;

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

//auto_ptr<PhM::PhMDetector>  mcdet;
std::vector< PhM::PhMDetector *>  mcdets;
//auto_ptr<FaceDetector> vjdet;

// struct for plot
struct posePair{
    float pose;
    int frameIndex;
};


// estimated poses:
vector< posePair > dposes;
vector< posePair > eposes;

// for calculate distance and output result
Distance d;
FeatureCollector* feature;
Boost* boostClassifier;
int preIndex = -1;


void writePoses(const char* filename, vector<posePair> poses){
    ofstream out;
    out.open(filename);
    posePair cur = poses[0];
    int curIndex = 0;
    for(int i = 0; i< poses.size(); i){
        //cout<<"i:"<<i<<endl;
        out<<cur.frameIndex;
        curIndex = cur.frameIndex;
        while(cur.frameIndex == curIndex) {
            out<<" "<<cur.pose;
            cur = poses[i++];
        }
        out<<endl;
    }
    cout<<"write file Done"<<endl;
    out.close();
}

static const char help_str[] =
    "\nOkapi detector demonstration\n"
    "\nOptions:\n%s\n";

// function declare
void collectFace(const cv::Mat& img, PhM::PhMPyramid& pyramid, FaceStore* store, int frameIndex);
vector<cv::Mat> generate(const cv::Mat& img, FaceStore* store, PhM::pose_estimator& my_est, int frameIndex, FaceLocator* locator, cv::VideoWriter);

cv::Mat run_detectors(const cv::Mat& img, PhM::PhMPyramid& pyramid, PhM::pose_estimator& my_est, Heat* heatGraph, FaceLocator* locator, FaceStore* store, int frameIndex)
{
    cv::Mat gray;
    if (img.channels() == 1)
        gray = img;
    else
        cv::cvtColor(img, gray, CV_RGB2GRAY);

    vector<MCTDetection> mct_dets;
    vector<MCTDetection> mct_dets_clustered;
    vector<float> ests;
    //if (mcdet.get())
        mct_dets.clear();
        OKAPI_TIMER_START("MCTPyramid");
		pyramid.build(gray, progressive);
        OKAPI_TIMER_STOP("MCTPyramid");
        OKAPI_TIMER_START("MCTscan");
	for (int ind=0; ind < mcdets.size(); ind++)
        	mcdets[ind]->scanPyramid(pyramid, mct_dets, shift_step);
        OKAPI_TIMER_STOP("MCTscan");
        OKAPI_TIMER_START("MCTcluster");
        mct_dets_clustered = clusterDetections(mct_dets, min_neighbours, neighbour_dist);
        OKAPI_TIMER_STOP("MCTcluster");
	for (unsigned j = 0; j < mct_dets_clustered.size(); j++)
	{
        cv::Rect estBox = mct_dets_clustered[j].box;
        locator->addRect(estBox);
        store->add(estBox.x, estBox.y, estBox.width, estBox.height, frameIndex);
	}
    // cluster those faces
    vector< vector<Face> > curC = store->cluster();
    return cv::Mat();
/*
    vector<cv::Rect> curLocations = locator->getLocations();
    for(unsigned k = 0; k < curLocations.size(); k++){
		float est=my_est.get_pose(gray, curLocations[k]); 
		ests.push_back(est);
    }

    // draw detection rectangles

    cv::Mat image;
    if (img.channels() != 3)
        cv::cvtColor(img, image, CV_GRAY2RGB);
    else
        image = img;
    ImageDeco deco(image);
    deco.createLayer(5, cvScalar(0,0,0));
    deco.setThickness(1);
    deco.setColor(55,50,250);
    if (draw_raw_detections)
    {
        for (unsigned j = 0; j < mct_dets.size(); j++)
            deco.drawRect(mct_dets[j].box);
    }
    deco.createLayer(5, cvScalar(0,0,0));
    deco.setThickness(2);
    deco.setColor(255,50,50);
    for (unsigned j = 0; j < mct_dets_clustered.size(); j++)
    {
        MCTDetection& det = mct_dets_clustered[j];
        deco.drawRect(det.box);
        //deco.drawText(stringify(ests[j]), det.box.x + 2, det.box.y + 15);
        heatGraph->add(det.box.x,det.box.y,det.box.width,det.box.height);
    }

    // this is for face locator
    deco.createLayer(5, cvScalar(0,0,0));
    deco.setColor(50,255,50);
    for (unsigned j = 0; j < curLocations.size(); j++)
    {
        cv::Rect& r = curLocations[j];
        deco.drawRect(r);
        deco.drawText(stringify(ests[j]), r.x + 2, r.y + 15);
    }
    //heatGraph->add(gray);
    heatGraph->saveHeat();

    store->printOut();
    // cout<<"Cur FrameIndex:"<<frameIndex<<endl;

    return deco.mergeLayers();
    */

}

int main(int argc, char* argv[])
{
    LibraryInitializer li;
    //OKAPI_LOG_CONFIGURE_DEFAULT();

    // parse arguments

    CommandLineOptions opt;
    opt.addOption('d', "mctdetector",   true,  "mctdetector file [optional]");
    opt.addOption('m', "multi-viewdetector", true, "multi-view detector file");
    opt.addOption('f', "video-file", true, "video file");
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

    string mctdet_fn,  muldet_fn, vjdet_fn, video_fn;
    try
    {
        opt.parse(argc, argv);
        mctdet_fn = opt.getArgument<string>("mctdetector", "");
	    muldet_fn = opt.getArgument<string>("multi-viewdetector", "");
        video_fn = opt.getArgument<string>("video-file");
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
	    PhM::PhMDetector *mcdet;
        try
        {
            mcdet=new PhM::PhMDetector(mctdet_fn);
        }
        catch (Exception& e)
        {
            printf("Error \n");
            printf("%s\n", e.what());
            exit(1);
        }
        printf("MCTDetector loaded, size: %d x %d\n",
                mcdet->getSize().width, mcdet->getSize().height);
	mcdets.push_back(mcdet);
    }

    if (muldet_fn.length() > 0)
    {
	PhM::PhMDetector *muldet;
        try
        {
            muldet = new PhM::PhMDetector(muldet_fn);
	    //muldet->diagonal_trans();
	   // muldet->save("/home/fjjiang/test_rot.xml");
        }
        catch (Exception& e)
        {
            printf("%s\n", e.what());
            exit(1);
        }

        printf("multiple view Detector loaded\n");
	mcdets.push_back(muldet);
    }

    if (muldet_fn.length() > 0)
    {
	PhM::PhMDetector *muldet;
        try
        {
            muldet = new PhM::PhMDetector(muldet_fn);
	    muldet->mirror_trans();
	   // muldet->save("/home/fjjiang/test_rot.xml");
        }
        catch (Exception& e)
        {
            printf("%s\n", e.what());
            exit(1);
        }

        printf("multiple view Detector loaded\n");
	mcdets.push_back(muldet);
    }

    boostClassifier = new Boost("features/trained_boost.xml");

	//load the estimator
    string file_X("/home/eeuser/bin_X_det_datas_sub2_24_ext27.7_sz28_facepix_gabor_forC_b10.txt");
        //string file_y("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/                  bin_y_det_datas_sub2_24_ext27.7_sz28_facepix_gabor_forC_b10.txt");
    string file_y("/home/eeuser/bin_y_det_datas_sub2_24_ext27.7_sz28_facepix_gabor_forC_b10.txt");
        //        //string file_b("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/                  bin_b_det_datas_sub2_24_ext27.7 _sz28_facepix_gabor_forC_b10.txt");
    string file_b("/home/eeuser/bin_b_det_datas_sub2_24_ext27.7_sz28_facepix_gabor_forC_b10.txt");
	//string file_X("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/bin_X_det_datas_sub2_25_ext30_facepix_gabor_forC.txt");
	//string file_y("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/bin_y_det_datas_sub2_25_ext30_facepix_gabor_forC.txt");
	//string file_b("/home/fjjiang/matlab_work/PoseManifold/gaborfacepix_v2/faceimgs/bin_b_det_datas_sub2_25_ext30_facepix_gabor_forC.txt");
	PhM::pose_estimator my_est(file_X, file_y, file_b, 1.9f, 60);


    // We change the source to video source here
    //VideoSource* src = createDefaultCamera(640, 480);
    VideoSource* src = new FFMPEGVideoSource(video_fn);
    //std::cout<<"Src Read"<<std::endl;

    double old_ts = -1;
    double fps = -1;
	int min_w = cvRound(mcdets[0]->getSize().width / upscale_factor);
	min_w = 50;
	max_width = 150;
	PhM::PhMPyramid pyramid(cv::Size(640, 480), mcdets[0]->getSize(), scale_step, min_w, max_width);
	int id=0;
	//printf("Pyramid initialize done\n");
        //cv::Mat img = run_detectors(src->getImage().clone(), pyramid);
	//	saveImage("/home/fjjiang/dets.bmp", img);
    //
    //
    /* using opencv */
    cv::VideoWriter cvWriter("ouput.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, cv::Size(800,480), 1);
    cv::VideoWriter faceWriter("face.avi", CV_FOURCC('M', 'P', '4', '2'), 25.0, cv::Size(800,480), 1);
   // FFMPEGVideoDestination* fWriter = new FFMPEGVideoDestination("foutput.avi", OKAPI_CODEC_ID_MPEG4, 25.0, cv::Size(640,480), 3800000);

    if(!cvWriter.isOpened())
        std::cout<<"could not open"<<endl;

    Heat* heatGraph = new Heat(cv::Size(640,480));
    FaceLocator* locator = new FaceLocator();
    FaceStore* store = new FaceStore();
    int frameIndex = 0;

    //cout<<"Src Collection Begins"<<endl;
    while (src->getNextFrame())
    {
        collectFace(src->getImage().clone(), pyramid, store, frameIndex++);
    }
    //cout<<"Src Collection Done"<<endl;

    store->setTotalFrame(--frameIndex);

    store->cluster();
    store->printOut();

    store->interpolate();
    store->printOut();

    GuiThread thread;
    ImageWindow* imgwin = new ImageWindow("Detector Demo (cam)");
    thread.addClient(imgwin);
    thread.start();

    delete src;
    // Reload src
    src = new FFMPEGVideoSource(video_fn);
    frameIndex = 0;


    feature = new FeatureCollector(store->getClusterSize());
    feature->clear();
    feature->writePoseDiff(store->getClustered());
    feature->writeLocaDiff(store->getClustered());

    while (imgwin->getWindowState() && src->getNextFrame())
    {
        vector<cv::Mat> imgs = generate(src->getImage().clone(), store, my_est, frameIndex++, locator, faceWriter);
        imgwin->setImage("Cam", imgs[0]);
		saveImage("/home/eeuser/dets.bmp", imgs[0]);
        cv::Mat output;
        cvtColor(imgs[0], output, CV_RGB2BGR);
        cvWriter<<output;
    }
    //cout<<"here"<<endl;
   // cout<<"pose Diff finished"<<endl;

    boostClassifier->getErrorRate();

    writePoses("interpolated.dat", eposes);
    writePoses("detected.dat", dposes);
    /*
    int frameIndex = 0;
    while (imgwin->getWindowState() && src->getNextFrame())
    {
        cv::Mat img = run_detectors(src->getImage().clone(), pyramid, my_est, heatGraph, locator, store, frameIndex++);
		saveImage("/home/eeuser/dets.bmp", img);
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
        if(!writer->writeFrame(img, 25)){
            printf("nothing is written");
        }

        imgwin->setImage("Cam", img);
        cvWriter<<img;
    }

    */
    // finished

    printf("Bye.\n");
    return 0;
}

void collectFace(const cv::Mat& img, PhM::PhMPyramid& pyramid, FaceStore* store, int frameIndex)
{
    cv::Mat gray;
    if (img.channels() == 1)
        gray = img;
    else
        cv::cvtColor(img, gray, CV_RGB2GRAY);

    saveImage("see.bmp", gray);
    vector<MCTDetection> mct_dets;
    vector<MCTDetection> mct_dets_clustered;
    vector<float> ests;
    //if (mcdet.get())
        mct_dets.clear();
        OKAPI_TIMER_START("MCTPyramid");
		pyramid.build(gray, progressive);
        OKAPI_TIMER_STOP("MCTPyramid");
        OKAPI_TIMER_START("MCTscan");
	for (int ind=0; ind < mcdets.size(); ind++)
        	mcdets[ind]->scanPyramid(pyramid, mct_dets, shift_step);
        OKAPI_TIMER_STOP("MCTscan");
        OKAPI_TIMER_START("MCTcluster");
        mct_dets_clustered = clusterDetections(mct_dets, min_neighbours, neighbour_dist);
        OKAPI_TIMER_STOP("MCTcluster");
	for (unsigned j = 0; j < mct_dets_clustered.size(); j++)
	{
        cv::Rect estBox = mct_dets_clustered[j].box;
        store->add(estBox.x, estBox.y, estBox.width, estBox.height, frameIndex);
	}
}

vector<cv::Mat> generate(const cv::Mat& img, FaceStore* store, PhM::pose_estimator& my_est, int frameIndex, FaceLocator* locator, cv::VideoWriter faceWriter){

    /* concert to gray */
    cv::Mat gray;
    if (img.channels() == 1)
        gray = img;
    else
        cv::cvtColor(img, gray, CV_RGB2GRAY);

    /* get locations */

    //for one cluster only
    /*
    vector<Face> collection = store->getClustered();
    vector<cv::Rect> curLocations = locator->getLocations(collection, frameIndex);
    */
    //vector<cv::Rect> curLocations = locator->getLocations(store->getClustered(), frameIndex);
    cout<<"========================"<<endl;
    cout<<"generate Frame Begins"<<endl;
    // try to distinguish interpolated and detected
    store->generatePoses(my_est, gray,frameIndex);
    vector< vector<Face> > collection = store->getClustered();
    //cout<<"collection size:"<<collection.size()<<endl;

    for (int i = 0; i < collection.size(); ++i){
        feature->writeMouthPix(collection[i][frameIndex], gray, i, frameIndex, faceWriter);
    }

    /* With the method of pure voting */
    vector<Face> voted = locator->getVotedFocus(collection, frameIndex);
    //cout<<"voted size"<<voted.size()<<endl;

    /* with ithe method of sum of the distance */
    vector<Face> faces = store->getFacesAtFrame(frameIndex);
    //cout<<"vector size:"<<faces.size()<<endl;
    int mySwitch = 1;
    vector<Face> minDFaces;
    boostClassifier->setSize(faces.size());

    if(faces.size() != 0){
        int index_dummy = d.getFocus(faces);
        // this is for speaker based
        if(mySwitch == 1){
            int index = boostClassifier->getFocus(faces, preIndex);
            preIndex = index;
            if(index != -1) {
                minDFaces.push_back(faces[index]);
                //d.printOut();
            }else{
                cout<<"all dummy"<<endl;
            }
        }else{
            // this is face based
            for (int i = 0; i < faces.size(); ++i)
            {
                if(boostClassifier->isSpeaker(faces[i], i)){ 
                    minDFaces.push_back(faces[i]);
                }
            }
        }
        // this is for face based
        //minDFace.print();
    }else{
        cout<<"distance part: no faces are given"<<endl;
    }
    cout<<"get Min D Faces"<<endl;

    vector<cv::Rect> normal = locator->getNormal(store->getClustered(), frameIndex);
    vector<cv::Rect> intered = locator->getIntered(store->getClustered(), frameIndex);
    //cout<<normal.size()<<endl;
    //cout<<intered.size()<<endl;
    vector<float> ests_normal;
    vector<float> ests_intered;

    for(unsigned k = 0; k < normal.size(); k++){
        //cout<<"size:"<<normal[k]<<endl;
		float est=my_est.get_pose(gray, normal[k]); 
		ests_normal.push_back(est);
    }
    for(unsigned k = 0; k < intered.size(); k++){
		float est=my_est.get_pose(gray, intered[k]); 
		ests_intered.push_back(est);
    }

    // draw detection rectangles

    cv::Mat image;
    if (img.channels() != 3)
        cv::cvtColor(img, image, CV_GRAY2RGB);
    else
        image = img;
    //cout<<"before interpolate"<<endl;

    //cout<<"Image Size:"<<image.size()<<endl;

    /* for interpolated faces */
    ImageDeco deco(image);
    // this is for face locator
    deco.createLayer(5, cvScalar(0,0,0));
    deco.setColor(50,50,250);
    deco.setThickness(1);
    for (unsigned j = 0; j < intered.size(); j++)
    {
        cv::Rect& r = intered[j];
        deco.drawRect(r);
        deco.drawText(stringify(ests_intered[j]), r.x + 2, r.y + 15);
        posePair n;
        n.frameIndex = frameIndex;
        n.pose = ests_intered[j];
        eposes.push_back(n);
    }

    /* for detected faces */
    deco.createLayer(5, cvScalar(0,0,0));
    deco.setThickness(1);
    deco.setColor(50,255,50);
    for (unsigned j = 0; j < normal.size(); j++)
    {
        cv::Rect& r = normal[j];
        deco.drawRect(r);
        deco.drawText(stringify(ests_normal[j]), r.x + 2, r.y + 15);
        posePair n;
        n.frameIndex = frameIndex;
        n.pose = ests_normal[j];
        dposes.push_back(n);
    }
    deco.drawText(stringify(frameIndex),2,15);

    //store->printOut();
    //cout<<"Cur FrameIndex:"<<frameIndex<<endl;
    cv::Mat tmp(480,800,CV_8UC3);

    //return deco.mergeLayers();
    
    // use distance 
    voted = minDFaces;

    for(int i = 0; i< voted.size(); i++) {
        //cout<<"Rect:"<<voted[i].toRect()<<endl;

        //cout<<"here"<<endl;
        cv::Rect face_expend_Rect = cv::Rect(voted[i].center.x  - voted[i].width - 20, voted[i].center.y - voted[i].width- 20 , voted[i].width + 40 , voted[i].height + 40);
        //cout<<face_expend_Rect<<endl;
        //cv::Mat face = img(voted[i].toRect());
        cv::Mat face = img(face_expend_Rect);
        deco.drawText("|", voted[i].center.x, voted[i].center.y - 100);
        deco.drawText("|", voted[i].center.x, voted[i].center.y - 90);
        deco.drawText("|", voted[i].center.x, voted[i].center.y - 80);
        deco.drawText("\\", voted[i].center.x - 8, voted[i].center.y - 80);
        deco.drawText("/", voted[i].center.x + 3, voted[i].center.y - 80);
        cv::Mat tmp_roi = tmp(cv::Rect(660, i*100 + 10, face.cols, face.rows));
        //cout<<"cols:"<<face.cols<<"rows:"<<face.rows<<endl;
        face.copyTo(tmp_roi);
    }

    cv::Mat output = deco.mergeLayers();
    //cout<<output.size()<<endl;

    cv::Mat dst_roi = tmp(cv::Rect(0, 0, output.cols, output.rows));
    output.copyTo(dst_roi);

    vector<cv::Mat> images;
    images.push_back(tmp);

    //cout<<tmp.size()<<endl;

    return images;

}
