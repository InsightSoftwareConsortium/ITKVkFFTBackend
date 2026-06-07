cmake_minimum_required(VERSION 3.16.3)
project(VkFFTBackend)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(VkFFTBackend_LIBRARIES VkFFTBackend)

if(NOT ITK_SOURCE_DIR)
  include(itk-module-init.cmake)
endif()

include(FetchContent)

#### Set up VkFFT flags ####

set(VKFFT_BACKEND 3 CACHE STRING "0 - Vulkan, 1 - CUDA, 2 - HIP, 3 - OpenCL, 4 - Level Zero, 5 - Metal")
add_compile_definitions(VKFFT_BACKEND=${VKFFT_BACKEND})

if(VKFFT_BACKEND EQUAL 1)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND VkFFTBackend_SYSTEM_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
elseif(VKFFT_BACKEND EQUAL 3)
  set(CL_TARGET_OPENCL_VERSION 300 CACHE STRING "OpenCL API version to target (e.g. 120, 300)")
  add_compile_definitions(CL_TARGET_OPENCL_VERSION=${CL_TARGET_OPENCL_VERSION})

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
elseif(VKFFT_BACKEND EQUAL 4)
  # oneAPI Level Zero loader (libze_loader) + headers (<level_zero/ze_api.h>).
  # Installed by Intel's "level-zero" package on Linux/Windows; also shipped
  # with the oneAPI Base Toolkit.
  find_path(LevelZero_INCLUDE_DIR
    NAMES level_zero/ze_api.h
    HINTS ENV LEVEL_ZERO_ROOT ENV CMPLR_ROOT
    PATH_SUFFIXES include)
  find_library(LevelZero_LIBRARY
    NAMES ze_loader
    HINTS ENV LEVEL_ZERO_ROOT ENV CMPLR_ROOT
    PATH_SUFFIXES lib lib64 lib/x64)
  if(NOT LevelZero_INCLUDE_DIR OR NOT LevelZero_LIBRARY)
    message(FATAL_ERROR "VKFFT_BACKEND=4 (Level Zero) requires the oneAPI Level Zero loader (ze_loader) and headers (level_zero/ze_api.h). Install the 'level-zero' package or set LEVEL_ZERO_ROOT.")
  endif()
  # VkFFT includes <ze_api.h> bare; this module uses <level_zero/ze_api.h> — both dirs needed.
  list(APPEND VkFFTBackend_SYSTEM_INCLUDE_DIRS
    ${LevelZero_INCLUDE_DIR}
    ${LevelZero_INCLUDE_DIR}/level_zero)
  list(APPEND VkFFTBackend_SYSTEM_LIBRARIES ${LevelZero_LIBRARY})

  set(VkFFTBackend_EXPORT_CODE_INSTALL "
  find_path(LevelZero_INCLUDE_DIR NAMES level_zero/ze_api.h)
  find_library(LevelZero_LIBRARY NAMES ze_loader)
  ")
  set(VkFFTBackend_EXPORT_CODE_BUILD "${VkFFTBackend_EXPORT_CODE_INSTALL}")
elseif(VKFFT_BACKEND EQUAL 5)
  if(NOT APPLE)
    message(FATAL_ERROR "VKFFT_BACKEND=5 (Metal) requires Apple platforms (macOS/iOS).")
  endif()

  # metal-cpp requires C++17.
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED True)

  # Apple's metal-cpp single-header bundle. Mirrored at bkaradzic/metal-cpp
  # for stable FetchContent (the official Apple zip URL is versioned per
  # macOS release and rotates).
  set(metal_cpp_GIT_TAG "metal-cpp_macOS15_iOS18")
  FetchContent_Declare(
    metal_cpp
    GIT_REPOSITORY "https://github.com/bkaradzic/metal-cpp.git"
    GIT_TAG ${metal_cpp_GIT_TAG}
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(metal_cpp)
  list(APPEND VkFFTBackend_SYSTEM_INCLUDE_DIRS ${metal_cpp_SOURCE_DIR})

  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
  list(APPEND VkFFTBackend_SYSTEM_LIBRARIES
    ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK})

  set(VkFFTBackend_EXPORT_CODE_INSTALL "
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
  ")
  set(VkFFTBackend_EXPORT_CODE_BUILD "${VkFFTBackend_EXPORT_CODE_INSTALL}")
else()
  message(WARNING "ITKVkFFTBackend currently supports only CUDA, OpenCL, Level Zero, or Metal backends.")
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

set(VkFFT_GIT_TAG "v1.3.4")
set(VkFFT_GIT_REPOSITORY "https://github.com/DTolm/VkFFT")
set(VkFFT_HEADER_URL "https://raw.githubusercontent.com/DTolm/VkFFT/${VkFFT_GIT_TAG}/vkFFT/vkFFT.h")

# v1.3.x split vkFFT.h into a multi-file tree, so the single-header URL
# fetch no longer suffices; force the full-repo path when not overridden.
option(BUILD_VKFFT "Fetch the full VkFFT repo (required for v1.3+ multi-file layout)" ON)
if(BUILD_VKFFT)
  # Fetch the full VkFFT repo with the header-only library and build targets
  FetchContent_Declare(
    vkfft_lib
    GIT_REPOSITORY ${VkFFT_GIT_REPOSITORY}
    GIT_TAG ${VkFFT_GIT_TAG}
    )
  FetchContent_GetProperties(vkfft_lib)
  if(NOT vkfft_lib_POPULATED)
    FetchContent_Populate(vkfft_lib)
    add_subdirectory(${vkfft_lib_SOURCE_DIR} ${vkfft_lib_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()

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

if(VKFFT_BACKEND EQUAL 4)
  target_link_libraries(VkFFTBackend PUBLIC ${LevelZero_LIBRARY})
  target_include_directories(VkFFTBackend SYSTEM PUBLIC ${LevelZero_INCLUDE_DIR})

  # Probe at configure time whether a Level Zero driver implementation is
  # actually present on this host (the loader can be installed without any
  # underlying GPU runtime, e.g. CPU-only servers or NVIDIA-only hosts). If
  # no driver is discovered, FFT tests would all fail at zeInit with
  # ZE_RESULT_ERROR_UNINITIALIZED — disable them so the dashboard signal
  # stays clean.
  if(BUILD_TESTING)
    set(_probe_src "${CMAKE_CURRENT_BINARY_DIR}/level_zero_probe.cxx")
    file(WRITE "${_probe_src}" [=[
#include <level_zero/ze_api.h>
int main(){
  if (zeInit(0) != ZE_RESULT_SUCCESS) return 1;
  uint32_t n = 0;
  if (zeDriverGet(&n, nullptr) != ZE_RESULT_SUCCESS) return 1;
  return n > 0 ? 0 : 1;
}
]=])
    try_run(VkFFTBackend_LEVEL_ZERO_RUNTIME_RUN
            VkFFTBackend_LEVEL_ZERO_RUNTIME_COMPILE
            "${CMAKE_CURRENT_BINARY_DIR}/level_zero_probe"
            SOURCES "${_probe_src}"
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${LevelZero_INCLUDE_DIR};${LevelZero_INCLUDE_DIR}/level_zero"
            LINK_LIBRARIES ${LevelZero_LIBRARY})
    if(NOT VkFFTBackend_LEVEL_ZERO_RUNTIME_COMPILE OR NOT VkFFTBackend_LEVEL_ZERO_RUNTIME_RUN EQUAL 0)
      message(STATUS "VkFFTBackend: no working Level Zero driver detected (zeInit/zeDriverGet probe failed); FFT tests will be DISABLED.")
      set(VkFFTBackend_LEVEL_ZERO_RUNTIME_AVAILABLE FALSE CACHE INTERNAL "")
    else()
      message(STATUS "VkFFTBackend: Level Zero runtime probe found at least one driver.")
      set(VkFFTBackend_LEVEL_ZERO_RUNTIME_AVAILABLE TRUE CACHE INTERNAL "")
    endif()
  endif()
endif()


if(VKFFT_BACKEND EQUAL 5)
  target_link_libraries(VkFFTBackend PUBLIC
    ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK})
endif()

