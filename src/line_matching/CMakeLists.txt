find_package(OpenCV REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/thirdparty/opencv-3.4
)


link_directories(
  ${OpenCV_LIBRARY_DIRS}
)


add_library(line_matching SHARED line_matching.cpp)
target_link_libraries(line_matching klt EDLine ${OpenCV_LIBS})