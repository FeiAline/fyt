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

/** \file
 *  \ingroup detectors
 */

#ifndef OKAPI_DETECTION_PHMDETECTOR_HPP_
#define OKAPI_DETECTION_PHMDETECTOR_HPP_

#include <okapi/config.hpp>
//#include <okapi/features/mcensustransform.hpp>
#include <okapi/types/basictypes.hpp>

#include <string>
#include <vector>
#include <iostream>

#include "./fifconvol_v5.0/RP_transform.hpp"
#include <okapi.hpp>
#include <cv.h>

using namespace okapi;

namespace ticpp
{
    class Node;
}

namespace PhM
{

    /// \addtogroup detectors
    //@{
    const int table_sz=625;//RP5 625

    class PhMPyramid; // forward declaration

    /** Weak classifier within a MCTDetStage
     * \note All members of this structure should be aligned, so we are using 512
     *       instead of 511 for array size.
     */
    struct PhMDetClassifier
    {
        double weights[table_sz];     /**< Weight table                               */
        int    	x;                /**< Feature location (x)                       */
        int    	y;                /**< Feature location (y)                       */

        /// Set weights to zero
        void zeroWeights()
        {
            memset(weights, 0, sizeof(weights));
        }
    };
    /** Stage of an MCTDetector */
    struct PhMDetStage
    {
        double threshold;       /**< classification threshold                   */
        std::vector<PhM::PhMDetClassifier> classifiers; ///< MCT classifiers
 
        /** Calculate sum of all classifiers in this stage.
         * \param image MCT-transformed image (IPL_DEPTH_16S)
         * \param pos_x,pos_y offset of detector window in image
         */
	inline double calcStageSum(const cv::Mat& image, int pos_x, int pos_y) const;
        
        /** Calculate sum of all classifiers in this stage.
         * This version is faster since it uses the pre-calculated index field
         * in PhMDetClassifier.
         * \param data MCT image data
         * \param index Index array
         * \return scaled integer sum, to be compared with int_threshold
         */
	inline double calcStageSum(const unsigned char* data, const int* index) const;
        /** Generate step-size dependent index to speed up stage-sum calculation.
         * \param step_size MCT image step size
         * \return new created index array, you have to delete it after use.
         */
	int* generateIndex(int step_size) const;
    };

    /** Detector using Modified Census Transform (MCT) features. */
    class PhMDetector
    {
        friend class PhMDetTrainer;
        public:

            /** Constructor */
            PhMDetector();

            /** Constructor an MCTDetector from file.
             * \param[in] filename Detector file
             */
            PhMDetector(std::string filename);

            /** Destructor */
            ~PhMDetector();

            /** Get detector size. */
            cv::Size getSize() const;

            /** Get number of stages. */
            int getStages() const;

            /** Get internal MCensusTransform instance.
             * \return MCT instance to be used for this MCTDetector
             */
	    
	    const RP_transform& getPhOP() const;

            /** Load detector from an XML file.
             * \param[in] filename Detector file
             */
            void load(std::string filename);

            /** Load detector from an XML Node.
             * \param[in] node XML node having an 'MCTDetector' child node
             */
            void loadXMLNode(ticpp::Node* node);

            /** Save detector to an XML file.
             * \param[in] filename Detector file
             * \param[in] training_info Training information to save along with the detector
             */
            void save(const std::string& filename, const std::string& training_info = "") const;

            /** Classify a certain position in an MCT image.
             * Set the detector to the given point and classify the MCT image.
             * This function does no argument/error checks.
             * \param mct_image an MCT image
             * \param pos_x,pos_y window position
             * \param n_stages number of stages passed (if not NULL)
             * \param confidence some confidence value (if not NULL)
             * \return detector result
             */
            bool classifyPhMPosition(const cv::Mat& phop_image, int pos_x, int pos_y,
                                     int* n_stages = NULL, double* confidence = NULL) const;

            /** Classify an arbitrary sized window with the help of a MCTPyramid.
             * This function will find the most appropriate level from the pyramid,
             * translate the window coordinates to the level's scale, and then classify
             * the window.
             * Note that position and size of the window actually being classified
             * will slightly differ from the desired values, if no perfectly fitting
             * scale can be found in the pyramid.
             * \note The aspect ratio of the window must be equal to the aspect
             *       ratio of the detector.
             * \param pyramid the initialized MCTPyramid
             * \param window the window to be classified (coordinates refer to the
             *               pyramid's input image)
             * \param n_stages number of stages passed (if not NULL)
             * \param confidence some confidence value (if not NULL)
             */
            bool classifyPyramidWindow(PhMPyramid& pyramid, const cv::Rect& window,
                                       int* n_stages = NULL, double* confidence = NULL) const;

            /** Scan a multi-scale MCT image pyramid.
             *  The resulting rectangles from different levels will be converted
             *  such that their coordinates fit to the input image of the pyramid.
             * \param pyramid the initialized MCTPyramid
             * \param detections results will be appended to this vector
             * \param shift_step window shift [pixel]
             * \param raw_detections if not NULL, raw detections (in pyramid coordinates) will be appended
             */
            void scanPyramid(PhMPyramid& pyramid, std::vector<MCTDetection>& detections,
                             int shift_step = 1, std::vector<BinaryPatternRawDetection>* raw_detections = NULL) const;

            /** Mirror this instance at the vertical axis.
             */

            /** Mirror this instance at the horizontal axis.
             */

            /** Rotate this instance 90deg clockwise.
             */

            /** Get detector stage.
             * \param i Stage number
             * \return Reference to detector stage
             */
	    const PhMDetStage& getStage_phop(size_t i) const;

            /** Get detector stage.
             * \param i Stage number
             * \return Reference to detector stage
             */
            PhMDetStage& getStage_phop(size_t i);
            /** Check if user info key was defined in detector file.
             * \return Whether user info key was defined in detector file
             */
            bool hasInfo(const std::string& key) const;

            /** Get user info for given key.
             * \param key Key to get user info for
             * \return User info for given key or empty string if key does not exist
             */
            std::string getInfo(const std::string& key) const;

            /** Set user info for given key.
             * \param key Key to set user info for
             * \param value Value to set the user info to
             */
            void setInfo(const std::string& key, const std::string& value);

	    void orthogonal_trans();
	    void orthogonal_trans_p();
	    void diagonal_trans();
	    void mirror_trans();
	      
        protected:

            int width;                                  ///< Width of detector
            int height;                                 ///< Height of detector
	    int win_width;
	    int win_height;
	    float RIP_theta;
	    RP_transform rp;  
            std::vector<PhM::PhMDetStage> stages_phop;    ///< phop stages of detector
                                ///<PhOP transform used by detector
            std::map<std::string,std::string> info;     ///< User-defined info from detector file
    };

    /** Generate a reduced list of detections by clustering neighbours.
     * \param detections detection list
     * \param min_neighbours detections with less neighbours than this value will
     *                       be discarded
     * \param neighbour_dist max. distance to consider detections as neighbours
     *                       (value is relative to detection window size)
     * \return reduced list of detections
     */
    extern OKAPI_API std::vector<MCTDetection> clusterDetections(const std::vector<MCTDetection>& detections,
                                                 int min_neighbours = 3,
                                                 double neighbour_dist = 1.2);

    /** Generate a reduced list of detections by averaging over all detections.
     * \param detections detection list
     * \return average detection
     */
    extern OKAPI_API MCTDetection averageDetections(const std::vector<MCTDetection>& detections);


    void draw_detections(cv::Mat&, std::vector<MCTDetection>&, std::string&, std::string&);
    /** MCTPyramid contains multiple scales of an image. The pyramid
     *  can be searched efficiently with an MCTDetector. */
    class PhMPyramid
    {
        public:

            /** Default constructor. */
	    PhMPyramid();

            /** Create MCT image pyramid.
             * This constructor creates a pyramid such that the given detector can
             * produce detections within the desired size range.
             * \note If you choose min_result_width smaller than the detector width,
             *       the pyramid will use upscaled images on the first level(s).
             *       This may be slow.
             * \param image input image (must not be changed during the use of the pyramid)
             * \param det_size MCTDetector size
             * \param scale_factor scaling factor from one level to the next
             * \param min_result_width minimum desired width of detection (-1 for detector size)
             * \param max_result_width maximum desired width of detection (-1 for image size)
             * \param mct provide a custom MCensusTransform instead of the default one
             */
            PhMPyramid(const cv::Mat& image, cv::Size det_size, double scale_factor = 1.1,
                       int min_result_width = -1, int max_result_width = -1);

            PhMPyramid(const cv::Size image_size, cv::Size det_size, double scale_factor = 1.1,
                       int min_result_width = -1, int max_result_width = -1);

            /** Create MCT image pyramid.
             * This constructor creates a pyramid with a dedicated scale range.
             * When using this pyramid in a detector, you should make sure that
             * all levels' MCT images are >= the detector size.
             * \param image input image (must not be changed during the use of the pyramid)
             * \param scale_factor scaling factor from one level to the next
             * \param scale_min scale to start with
             * \param scale_max scale to stop with (may be exceeded)
             * \param mct provide a custom MCensusTransform instead of the default one
             */
            PhMPyramid(const cv::Mat& image, double scale_factor,
                       double scale_min, double scale_max);

            /** Destructor. */
            ~PhMPyramid();

            /** Return MCensusTranform instance */
            //const MCensusTransform& getMCT() const;

            /** Generate and store all levels' images such they don't need to be (re-)generated
             *  each time you request them.
             *  Calling this method is optional, but it has the advantage that the Pyramid
             *  images can be generated faster using the "progressive" option. The drawback
             *  is that that it may need a lot of memory to keep the entire Pyramid.
             *  \param progressive if true, rescale the image from the previous level instead
             *             of the original image. This is faster than rescaling the original
             *             image, but the result may look slightly worse.
             *  \param keep_gray set this to true if you also want to store the gray images
             *             internally such that they don't need to be regenerated if you request
             *             them later.
             */
            void build(bool progressive = true, bool keep_gray = false);
			void build(const cv::Mat& img, bool progressive = true, bool keep_gray = false);
            /** Get number of levels in the pyramid.
             * \return number of levels
             */
            int getLevels() const;

            /** Get scale factor of a level.
             *  A value of 1.0 represents the input image dimensions, higher values
             *  represent smaller images.
             *  \param level level number
             *  \return level scale factor
             */
            double getScale(int level) const;

            /** Get the (gray) image size of a level.
             *  The according MCT image is 2 pixels smaller (width and height) than the gray image.
             *  \param[in] level The wanted pyramid level
             *  \return image size
             */
            cv::Size getSize(int level) const;
	    cv::Size getSize(int level, cv::Size img_sz) const;

            /** Get gray image of a level.
             *  \note Pre-create the Pyramid using build() with keep_gray set to true if you
             *        don't want the images to be (re-)created on each call.
             *  \param level level number
             *  \return level image
             */
            cv::Mat getGrayImage(int level) const;

            /** Get MCT image of a level.
             *  \note Pre-create the Pyramid using build() if you don't want the images
             *        to be (re-)created on each call.
             *  \param level level number
             *  \return MCT image, 2 pixels smaller (width and height) than the according gray image
             */
            //cv::Mat getMCTImage(int level) const;
	    
	    cv::Mat getPhOPImage(int level);

            /** Finde the level closest to a given scale.
             *  \param desired_scale desired scale factor
             *  \return level number
             */
            int findClosestLevel(double desired_scale) const;

            /** Get the input image from which the MCTPyramid is built
             * \return input image for MCTPyramid
             */
            cv::Mat getInputImage() const;
	    RP_pyramid*		      RP_pyr;
        protected:

            /** Precompute scale factors
             * \param[in] scale_factor scaling factor from one level to the next
             * \param[in] scale_min scale to start with
             * \param[in] scale_max scale to stop with (may be exceeded)
             */
            void setupLevels(double scale_factor, double scale_min, double scale_max);

            cv::Mat                   input_image;  ///< grayscale version of input image
            std::vector<double>       scales;       ///< scale factors of pyramid levels
            std::vector<cv::Mat>      gray_imgs;    ///< grayscale pyramid levels
            std::vector<cv::Mat>      phop_imgs;    ///< PhOP pyramid levels
            RP_transform            rp;         ///< PhOP to be used by pyramid
	    //std::vector<HVoMCT_transform *> HVoMCTs;
	    //HVoMCT_pyramid 	      *HVoMCT_pyr;
	    
    };


    // implementation of inline functions ///////////////////////////////////////////////

    double PhMDetStage::calcStageSum(const cv::Mat& image,
                                     int pos_x, int pos_y) const
    {
        double sum = 0.0;
        int j = classifiers.size();
	if (j>0)
        do
        {
            j -= 1;
            const PhMDetClassifier& cls = classifiers[j];
            int index = image.ptr<GrayPixel16S>(pos_y + cls.y)[pos_x + cls.x];
            sum += cls.weights[index];
        }
        while (j != 0);
        return sum;
    }

   double PhMDetStage::calcStageSum(const unsigned char* data, const int* index) const
    {
        double sum = 0.0;
        int j = classifiers.size();
	if (j>0)
        do
        {
            j -= 1;
            GrayPixel16S* ptr = (GrayPixel16S*)&data[index[j]];
            sum += classifiers[j].weights[*ptr];
        }
        while (j != 0);
        return sum;
    }
    //@}
}

#endif
