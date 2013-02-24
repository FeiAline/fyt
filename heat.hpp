#ifndef HEAT_HPP
#define HEAT_HPP


#include <okapi/config.hpp>
#include <okapi/types/basictypes.hpp>

#include <string>
#include <vector>
#include <iostream>

#include <okapi.hpp>
#include <cv.h>
#include <math.h>

class Heat;
class Heat
{
    public:
        Heat(cv::Size, int value=0); // Constructor, build a 2-D map with zero values
        ~Heat(); // Destructor

        cv::Size getSize() const;
        void setSize(cv::Size);
        void saveHeat(); // output the image of heat
        void add(cv::Mat frame);
        void add(int x, int y, int width, int height);

    private:
        void renderHeat();
        cv::Mat heat;
};

#endif
