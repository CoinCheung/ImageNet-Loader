
cmake_minimum_required (VERSION 2.8)
project (randaug_cpp)

set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

## external lib paths
set (OpenCV_DIR /usr/local/lib/cmake/opencv4)


## use find_package
# opencv
find_package(OpenCV REQUIRED)
list(APPEND LIGHTSTEP_LINK_LIBRARY ${OpenCV_LIBS})


## other link libs
list(APPEND LIGHTSTEP_LINK_LIBRARY glog pthread)


aux_source_directory (. SRC_CPP)
message(${SRC_CPP})
# set (SRC_CPP
#     ./main.cpp
#     ./transforms.cpp
#     ./random.cpp
#     ./rand_aug.cpp
#     ./server.cpp
# )

add_executable (main ${SRC_CPP})
target_link_libraries(main
    ${SUB_LIBRARY}
    ${LIGHTSTEP_LINK_LIBRARY}
)
