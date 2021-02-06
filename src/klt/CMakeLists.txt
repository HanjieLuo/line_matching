find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O3 -msse4.2 -mavx")

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/thirdparty/opencv-3.4
)


link_directories(
  ${OpenCV_LIBRARY_DIRS}
)

set(KLT_SOURCES
    klt.cpp
    lk_tracker_invoker_ori.cpp
    lk_tracker_invoker_1d.cpp
    lk_tracker_invoker_2d.cpp
    lk_tracker_invoker_2d_single.cpp
)

add_library(klt SHARED ${KLT_SOURCES})
target_link_libraries(klt ${OpenCV_LIBS})