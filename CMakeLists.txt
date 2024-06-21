cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ComputerGraphics C CXX)

# Enable vector instructions
if(MSVC)
	add_definitions( /arch:AVX2)
endif()

if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/framework")
	# Create framework library and include CMake scripts (compiler warnings, sanitizers and static analyzers).
	add_subdirectory("framework")
	set(REFERENCE_MODE OFF)
else()
	# During development the framework lives in parent folder.
	add_subdirectory("../../../framework/" "${CMAKE_BINARY_DIR}/framework/")
	set(REFERENCE_MODE ON)
endif()

set(embree_DIR "C:/embree-4.3.1/lib/cmake/embree-4.3.1/") # Replace with directory to local Embree installation
find_package(embree REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED)

add_library(CGFinProjLib INTERFACE)
enable_sanitizers(CGFinProjLib)
set_project_warnings(CGFinProjLib)
include(${CMAKE_CURRENT_LIST_DIR}/src/CMakeLists.txt)
target_include_directories(CGFinProjLib INTERFACE "${CMAKE_CURRENT_LIST_DIR}/src/")
target_link_libraries(CGFinProjLib INTERFACE CGFramework OpenGL::GLU OpenMP::OpenMP_CXX embree)
target_compile_features(CGFinProjLib INTERFACE cxx_std_20)

# Main executable config
add_executable(SeminarImpl	"src/main.cpp")
enable_sanitizers(SeminarImpl)
set_project_warnings(SeminarImpl)
target_link_libraries(SeminarImpl PUBLIC CGFinProjLib)
target_compile_features(SeminarImpl PUBLIC cxx_std_20)

# Preprocessor definitions for path(s)
target_compile_definitions(CGFinProjLib INTERFACE
	"-DDATA_DIR=\"${CMAKE_CURRENT_LIST_DIR}/data/\""
	"-DRENDERS_DIR=\"${CMAKE_CURRENT_LIST_DIR}/renders/\"")
