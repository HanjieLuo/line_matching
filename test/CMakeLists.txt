find_package(OpenCV REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/thirdparty/opencv-3.4
)

link_directories(
  ${OpenCV_LIBRARY_DIRS}
)

add_executable(test_edline_detector test_edline_detector.cpp)
target_link_libraries(test_edline_detector EDLine ${OpenCV_LIBS})

add_executable(test_line_matching test_line_matching.cpp)
target_link_libraries(test_line_matching line_matching ${OpenCV_LIBS})

