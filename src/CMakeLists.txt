
## external lib paths
set (OpenCV_DIR /usr/local/lib/cmake/opencv4)


## use find_package
# opencv
find_package(OpenCV REQUIRED)
list(APPEND LIGHTSTEP_LINK_LIBRARY ${OpenCV_LIBS})


## other link libs
list(APPEND LIGHTSTEP_LINK_LIBRARY glog pthread)


aux_source_directory (. SRC_SO)
# set (SRC_SO
#     ./pipeline.cpp
#     ./rand_aug.cpp
#     ./random.cpp
#     ./transforms.cpp
# )
message(${SRC_SO})

# add_library(loader STATIC ${SRC_SO})
# target_link_libraries(loader ${LIGHTSTEP_LINK_LIBRARY})

add_library(loader SHARED ${SRC_SO})
target_link_libraries(loader ${LIGHTSTEP_LINK_LIBRARY})

# set_target_properties(loader
#     PROPERTIES
#     LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

