cmake_minimum_required(VERSION 3.20)
project("21120511")

# Set C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV and make executables
include_directories(${OPENCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB HEADERS "${PROJECT_SOURCE_DIR}/include/*.hpp")
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Set source path
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(SOURCE_PATH="${PROJECT_SOURCE_DIR}/")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(SOURCE_PATH="")
endif()