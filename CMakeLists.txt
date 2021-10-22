cmake_minimum_required(VERSION 3.10)
project( FlakeAutoFind )

find_package( OpenCV REQUIRED )
include_directories ( ${OpenCV_INCLUDE_DIRS} )
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")

add_subdirectory ( src )
add_executable( FlakeAutoFind src/flakeautofind.cc )
target_link_libraries( FlakeAutoFind ${OpenCV_LIBS} src)
