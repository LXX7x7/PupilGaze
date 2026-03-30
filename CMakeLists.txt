cmake_minimum_required(VERSION 3.10)

project(KappaCalibrate)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/shared_cpp/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors/singleeyefitter")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/shared_cpp/include/zeromq")

#[[
file( GLOB HEADERS 
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/api/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/api/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/calibration/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/calibration/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors/singleeyefitter/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors/singleeyefitter/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors/singleeyefitter/Fit/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors/singleeyefitter/Fit/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors/singleeyefitter/ImageProcessing/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors/singleeyefitter/ImageProcessing/*.hpp"
  )
]]
set(PROJECT_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/geometry.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/state.cpp"
)

file(GLOB MODULE_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/api/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/api/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/calibration/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/calibration/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors/singleeyefitter/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors/singleeyefitter/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors/singleeyefitter/Fit/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors/singleeyefitter/Fit/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors/singleeyefitter/ImageProcessing/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/shared_modules/pupil_detectors/singleeyefitter/ImageProcessing/*.cpp"
)

add_executable(KappaCalibrate ${PROJECT_SRCS} ${MODULE_SRCS})

target_link_libraries(KappaCalibrate ${Boost_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${LAPACK_LIBRARIES})
