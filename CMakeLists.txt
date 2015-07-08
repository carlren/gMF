cmake_minimum_required(VERSION 2.8)
project(gMF)
#SET (CMAKE_VERBOSE_MAKEFILE true)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

IF(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
ELSE(MSVC_IDE)
  set(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
  set(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF(MSVC_IDE)

if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
endif(APPLE)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OPENCV_INCLUDE_DIRS})
add_subdirectory(ORUtils)

set(GMF_LIB
gMF_Lib/gMF.h
gMF_Lib/gMF_define.h
gMF_Lib/engines/gMF_fileter_engine.h
gMF_Lib/engines/gMF_fileter_engine.cu
gMF_Lib/engines/gMF_fileter_engine_shared.h
gMF_Lib/engines/gMF_filter_engine_kernel.h
gMF_Lib/engines/gMF_inference_engine.h
gMF_Lib/engines/gMF_inference_engine.cu
gMF_Lib/engines/gMF_inference_engine_kernel.h
gMF_Lib/objects/gMF_BF_info.h
gMF_Lib/objects/gMF_GF_info.h
)

set(GSLIC_LIB
gSLIC_Lib/engines/gSLIC_core_engine.h
gSLIC_Lib/engines/gSLIC_seg_engine.h
gSLIC_Lib/engines/gSLIC_seg_engine_GPU.h
gSLIC_Lib/engines/gSLIC_seg_engine_shared.h
gSLIC_Lib/engines/gSLIC_core_engine.cpp
gSLIC_Lib/engines/gSLIC_seg_engine.cpp
gSLIC_Lib/engines/gSLIC_seg_engine_GPU.cu
gSLIC_Lib/objects/gSLIC_settings.h
gSLIC_Lib/objects/gSLIC_spixel_info.h
gSLIC_Lib/gSLIC_defines.h
gSLIC_Lib/gSLIC.h
)

list(APPEND "-std=c++11 -ftree-vectorize")
SOURCE_GROUP(gMF_Lib FILES ${GMF_LIB})

# libraries
cuda_add_library(gSLIC2_lib
			${GSLIC_LIB}
			OPTIONS -gencode arch=compute_30,code=compute_30)
target_link_libraries(gSLIC2_lib ${CUDA_LIBRARY})


cuda_add_library(gMF_Lib
			${GMF_LIB}
			OPTIONS -gencode arch=compute_30,code=compute_30)
target_link_libraries(gMF_Lib ${CUDA_LIBRARY} gSLIC2_lib)


add_executable(gMF demo.cpp NVTimer.h image_helper.h)
target_link_libraries(gMF gMF_Lib ${OpenCV_LIBS})

add_executable(demo_debug demo_backup.cpp NVTimer.h)
target_link_libraries(demo_debug gMF_Lib ${OpenCV_LIBS})

add_executable(gSLIC demo_gslic.cpp)
target_link_libraries(gSLIC gSLIC2_lib ${OpenCV_LIBS} )

