cmake_minimum_required (VERSION 3.8)
project (mnist_test_adapter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(processImages 
  processImages.cpp 
  connectedComponents/connectedComponents.cpp
)

include_directories(.)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -w -std=c++17")
SET(CMAKE_USE_RELATIVE_PATHS ON)

find_package(OpenCV REQUIRED)
target_link_libraries(processImages ${OpenCV_LIBS})
