
cmake_minimum_required (VERSION 2.8)
project (img_loader)

set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

## external lib paths
set (OpenCV_DIR /usr/local/lib/cmake/opencv4)
set (gRPC_DIR /usr/local/lib/cmake)


## use find_package
# grpc
find_package (gRPC REQUIRED)
list(APPEND LIGHTSTEP_LINK_LIBRARY gRPC::grpc++_unsecure)
# protobuf
find_package (Protobuf REQUIRED)
list(APPEND LIGHTSTEP_LINK_LIBRARY protobuf::libprotobuf)
# opencv
find_package(OpenCV REQUIRED)
list(APPEND LIGHTSTEP_LINK_LIBRARY ${OpenCV_LIBS})


## other link libs
list(APPEND LIGHTSTEP_LINK_LIBRARY glog pthread)

## subdirs
add_subdirectory (comm)

## self libs
list(APPEND SUB_LIBRARY comm)

#
# add_executable (server ./server.cpp ./common.cpp)
# add_executable (client ./client.cpp ./common.cpp)
# target_link_libraries (server msginfo_proto
#     ${LIGHTSTEP_LINK_LIBRARY}
# )
# target_link_libraries (client msginfo_proto
#     ${LIGHTSTEP_LINK_LIBRARY}
# )

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
