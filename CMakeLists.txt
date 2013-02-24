###############################################################################
# Example CMakeLists.txt for using the OKAPI library with CMake               #
###############################################################################

# CMake 2.8 or higher is required
cmake_minimum_required(VERSION 2.8)

# Give your project a suitable name
project("cudagabor")

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake) 
#SET(CMAKE_CXX_FLAGS “${CMAKE_CXX_FLAGS} -fopenmp”)
#SET(CMAKE_C_FLAGS “${CMAKE_C_FLAGS} -fopenmp”)


FIND_PACKAGE(CUDA)

INCLUDE(FindCUDA)
INCLUDE_DIRECTORIES( 
   /home/eeuser/CUDA_SDK/C/common/inc
   /home/fjjiang/CUDA_SDK/C/common/inc
   #/usr/local/cuda/include 
   /home/fjjiang/work/pgrlibdcstereo
   /usr/local/include/dc1394
   /usr/local/include/triclops/include
   /usr/local/include/opencv2
)

include(FindOpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

FIND_PACKAGE(OpenCV REQUIRED)
IF(OpenCV_FOUND)
     LIST(APPEND PROJECT_INCLUDES ${OpenCV_INCLUDE_DIR})
     LIST(APPEND PROJECT_LIBRARY_DIRS ${OpenCV_LIBRARIES})
     INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})
     LINK_DIRECTORIES(${OpenCV_LIBRARIES})
ENDIF(OpenCV_FOUND)

find_package("OKAPI" REQUIRED
    PATHS
        /home/fjjiang/work/okapi_git/okapi/build.release
        /home/fjjiang/work/okapi_git/okapi/build.debug
)
include(${OKAPI_USE_FILE})

#set(SRCS Detector.cpp)
#set(SRCS Detector_pose.cpp)
set(SRCS Detector_multi_pose.cpp)
#set(SRCS Detector_multi_pose_samefilter.cpp)
#set(SRCS main_generate_train.cpp)
#set(SRCS main_generate_fromtext.cpp)
#set (SRCS main_test_pose.cpp)
#set(SRCS RunPhMDetector_list_multiPIE.cpp)

CUDA_ADD_EXECUTABLE(cudaGabor
    faceLocator.cpp
    heat.cpp
    PhMdetector.cpp
    #PhMdettrainer.cpp
    ./fifconvol_v5.0/RP_transform.cpp
    ./fifconvol_v5.0/fifconvol_pyr.cu
    GRNN.cpp
    heat.cpp
    face.cpp
    ${SRCS}
)
LINK_LIBRARIES("-L/home/eeuser/CUDA_SDK/C/lib -lcutil")
LINK_LIBRARIES("-L/usr/local/include/opencv2")
target_link_libraries(cudaGabor okapi okapi-gui okapi-videoio dc1394 /usr/lib/libraw1394.so.11 /usr/local/include/triclops/lib/libpgrlibdcstereo.a)
#target_link_libraries(cudaGabor /usr/local/lib/libavformat.so /usr/local/lib/libavutil.so /usr/local/lib/libavcodec.so)
target_link_libraries(cudaGabor /home/eeuser/triclops/lib/libtriclops.a /home/eeuser/triclops/lib/libpnmutils.a)
target_link_libraries(cudaGabor /usr/local/lib/libopencv_highgui.so.2.2 opencv_imgproc opencv_video)
#target_link_libraries(cudaGabor okapi okapi-gui okapi-videoio dc1394 /usr/lib/libraw1394.so.11 /home/fjjiang/work/pgrlibdcstereo/libpgrlibdcstereo.a)

##target_link_libraries(cudagabor okapi ${CUDA_TARGET_LINK} ${CUDA_CUT_TARGET_LINK})
##target_link_libraries(cudatest okapi  ${CUDA_TARGET_LINK} ${CUDA_CUT_TARGET_LINK})
 

