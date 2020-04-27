
cmake_minimum_required (VERSION 2.8)
project (randaug_cpp)


if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

## external lib paths
set (OpenCV_DIR /usr/local/lib/cmake/opencv4)


## use find_package
# opencv
find_package(OpenCV REQUIRED)
list(APPEND LIGHTSTEP_LINK_LIBRARY ${OpenCV_LIBS})

## other link libs
list(APPEND LIGHTSTEP_LINK_LIBRARY glog pthread )

add_subdirectory(src)
add_subdirectory(third_party/pybind11)

list(APPEND LIGHTSTEP_LINK_LIBRARY loader)

# add_executable (main ./dataloader.cpp)
# target_link_libraries(main ${LIGHTSTEP_LINK_LIBRARY})

pybind11_add_module(dataloader SHARED wrapper.cpp)
target_link_libraries(dataloader ${LIGHTSTEP_LINK_LIBRARY})
set_target_properties(dataloader
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/cdataloader)
