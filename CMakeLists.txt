cmake_minimum_required(VERSION 3.16.3)
project(VkFFTBackend)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(VkFFTBackend_LIBRARIES VkFFTBackend)

if(NOT ITK_SOURCE_DIR)
  include(itk-module-init.cmake)
endif()

## When this module is loaded by an app, load OpenCL too.
set(VkFFTBackend_EXPORT_CODE_INSTALL "
set(OpenCL_DIR \"${OpenCL_DIR}\")
find_package(OpenCL REQUIRED)
")
set(VkFFTBackend_EXPORT_CODE_BUILD "
if(NOT ITK_BINARY_DIR)
  set(OpenCL_DIR \"${OpenCL_DIR}\")
  find_package(OpenCL REQUIRED)
endif()
")
set(VkFFTBackend_SYSTEM_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS})
get_filename_component(OpenCL_LIB_DIR ${OpenCL_LIBRARY} DIRECTORY)
set(VkFFTBackend_SYSTEM_LIBRARY_DIRS ${OpenCL_LIB_DIR})

message(STATUS "==== Show all variables (CMakeLists.txt begin) ===")
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
message(STATUS "==== Show all variables (CMakeLists.txt end) ===")

## VkFFT dependency
include(FetchContent)
set(VKFFT_BACKEND 3 CACHE STRING "0 - Vulkan, 1 - CUDA, 2 - HIP, 3 - OpenCL")
add_definitions(-DVKFFT_BACKEND=${VKFFT_BACKEND} -DCL_TARGET_OPENCL_VERSION=120)
set(vulkan_GIT_REPOSITORY "https://github.com/DTolm/VkFFT") # original source
set(vulkan_GIT_TAG        "7402aeb0302fcacad09fe428bd7010ea21f00997")
# set(vulkan_GIT_REPOSITORY "https://github.com/Leengit/VkFFT") # Kitware copy of original source
# set(vulkan_GIT_TAG        "0c5ddfffe1af9ac96c21d4c38941f9b5481b73c3")
FetchContent_Declare(
  vulkan_lib
  GIT_REPOSITORY ${vulkan_GIT_REPOSITORY}
  GIT_TAG ${vulkan_GIT_TAG}
  )
FetchContent_MakeAvailable(vulkan_lib)
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Remove this list of disabled warnings when VkFFT has been updated
  message("Adding compile options: -Wno-format-overflow")
  add_compile_options(-Wno-format-overflow)
  target_compile_options(VkFFT INTERFACE -Wno-format-overflow)
endif()
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # Remove this list of disabled warnings when VkFFT has been updated
  message("Adding compile options: /wd4146 /wd4244 /wd4996")
  #   C4146: unary minus operator applied to unsigned type, result still unsigned
  #   C4244: 'argument': conversion from 'double' to 'uint64_t', possible loss of data
  #   C4996: 'sprintf': This function or variable may be unsafe. Consider using sprintf_s instead.
  add_compile_options(/wd4146 /wd4244 /wd4996)
  target_compile_options(VkFFT INTERFACE /wd4146 /wd4244 /wd4996)
endif()

include_directories(SYSTEM ${vulkan_lib_SOURCE_DIR}/vkFFT)

if(NOT ITK_SOURCE_DIR)
  find_package(ITK REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})
  include(ITKModuleExternal)
else()
  set(ITK_DIR ${CMAKE_BINARY_DIR})
  itk_module_impl()
endif()
