cmake_minimum_required(VERSION 3.16.3)
project(VkFFTBackend)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(VkFFTBackend_LIBRARIES VkFFTBackend)

if(NOT ITK_SOURCE_DIR)
  include(itk-module-init.cmake)
endif()

#### Set up VkFFT flags ####

set(VKFFT_BACKEND 3 CACHE STRING "0 - Vulkan, 1 - CUDA, 2 - HIP, 3 - OpenCL")
add_compile_definitions(VKFFT_BACKEND=${VKFFT_BACKEND})

if(VKFFT_BACKEND EQUAL 1)
  # pass
elseif(VKFFT_BACKEND EQUAL 3)
  add_compile_definitions(CL_TARGET_OPENCL_VERSION=120)
  
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
  list(APPEND VkFFTBackend_SYSTEM_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS})
  get_filename_component(OpenCL_LIB_DIR ${OpenCL_LIBRARY} DIRECTORY)
  set(VkFFTBackend_SYSTEM_LIBRARY_DIRS ${OpenCL_LIB_DIR})
else()
  message(WARNING "ITKVkFFTBackend currently supports only CUDA or OpenCL backends.")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Remove this list of disabled warnings when VkFFT has been updated
  message("Adding compile options: -Wno-format-overflow")
  add_compile_options(-Wno-format-overflow)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # Remove this list of disabled warnings when VkFFT has been updated
  message("Adding compile options: /wd4146 /wd4244 /wd4996")
  #   C4146: unary minus operator applied to unsigned type, result still unsigned
  #   C4244: 'argument': conversion from 'double' to 'uint64_t', possible loss of data
  #   C4996: 'sprintf': This function or variable may be unsafe. Consider using sprintf_s instead.
  add_compile_options(/wd4146 /wd4244 /wd4996)
endif()

#### Populate VkFFT dependency ###

# VkFFT is a header-only library with build target(s)
# for FFT benchmarking and applications.
# By default we populate only the header library file `VkFFT.h`
# without building other VkFFT targets.

set(VkFFT_GIT_TAG "v1.2.31")
set(VkFFT_GIT_REPOSITORY "https://github.com/DTolm/VkFFT")
set(VkFFT_HEADER_URL "https://raw.githubusercontent.com/DTolm/VkFFT/${VkFFT_GIT_TAG}/vkFFT/vkFFT.h")

include(FetchContent)
option(BUILD_VKFFT OFF)
if(BUILD_VKFFT)
  # Fetch the full VkFFT repo with the header-only library and build targets
  FetchContent_Declare(
    vkfft_lib
    GIT_REPOSITORY ${VkFFT_GIT_REPOSITORY}
    GIT_TAG ${VkFFT_GIT_TAG}
    )
  FetchContent_MakeAvailable(vkfft_lib)

  set(vkfft_INCLUDE_DIR "${vkfft_lib_SOURCE_DIR}/vkFFT")
else()
  # Fetch header-only VkFFT library
  set(vkfft_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/include")
  FetchContent_Declare(
    vkfft_header_only
    URL ${VkFFT_HEADER_URL}
    DOWNLOAD_DIR "${vkfft_INCLUDE_DIR}"
    DOWNLOAD_NO_EXTRACT TRUE
  )
  FetchContent_MakeAvailable(vkfft_header_only)
endif()
list(APPEND VkFFTBackend_SYSTEM_INCLUDE_DIRS ${vkfft_INCLUDE_DIR})

#### Configure ITK module ####

if(NOT ITK_SOURCE_DIR)
  find_package(ITK REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})
  include(ITKModuleExternal)
else()
  set(ITK_DIR ${CMAKE_BINARY_DIR})
  itk_module_impl()
endif()

