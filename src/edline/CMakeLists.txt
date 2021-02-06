cmake_minimum_required(VERSION 2.8.3)

find_package(OpenCV REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(
  ${OpenCV_LIBRARY_DIRS}
)

add_library(EDLine SHARED edline_detector.cpp)
target_link_libraries(EDLine ${OpenCV_LIBS})