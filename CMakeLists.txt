cmake_minimum_required(VERSION 3.5)
project(blur_detector)

set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILE main.cpp blur_detector.cpp)
link_libraries(libopencv_core.so libopencv_imgcodecs.so libopencv_imgproc.so libopencv_highgui.so)

add_executable(blur_detector ${SOURCE_FILE})
target_link_libraries(blur_detector libopencv_core.so libopencv_imgcodecs.so libopencv_imgproc.so libopencv_highgui.so)