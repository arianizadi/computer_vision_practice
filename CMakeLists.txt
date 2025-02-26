cmake_minimum_required(VERSION 3.10)
project(depthai_opencv_example)

find_package(OpenCV REQUIRED)
find_package(depthai CONFIG REQUIRED)

file(GLOB SRC_FILES "src/*.cpp")

foreach(SRC_FILE ${SRC_FILES})
  get_filename_component(EXEC_NAME ${SRC_FILE} NAME_WE)
  add_executable(${EXEC_NAME} ${SRC_FILE})
  target_include_directories(${EXEC_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${EXEC_NAME} PRIVATE depthai::opencv ${OpenCV_LIBS})
endforeach()
