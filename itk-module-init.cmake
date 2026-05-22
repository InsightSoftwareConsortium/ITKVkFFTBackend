#
# Find the packages required by this module
#
set(VKFFT_BACKEND 3 CACHE STRING "1 - CUDA, 3 - OpenCL, 4 - Level Zero, 5 - Metal")
if(${VKFFT_BACKEND} EQUAL 1)
  find_package(CUDA 9.0 REQUIRED)
  find_library(CUDA_NVRTC_LIB libnvrtc nvrtc HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" "/usr/lib64" "/usr/local/cuda/lib64")
elseif(${VKFFT_BACKEND} EQUAL 3)
  find_package(OpenCL REQUIRED)
elseif(${VKFFT_BACKEND} EQUAL 4)
  find_path(LevelZero_INCLUDE_DIR
    NAMES level_zero/ze_api.h
    HINTS ENV LEVEL_ZERO_ROOT ENV CMPLR_ROOT
    PATH_SUFFIXES include)
  find_library(LevelZero_LIBRARY
    NAMES ze_loader
    HINTS ENV LEVEL_ZERO_ROOT ENV CMPLR_ROOT
    PATH_SUFFIXES lib lib64 lib/x64)
  if(NOT LevelZero_INCLUDE_DIR OR NOT LevelZero_LIBRARY)
    message(FATAL_ERROR "VKFFT_BACKEND=4 (Level Zero) requires the oneAPI Level Zero loader (ze_loader) and headers (level_zero/ze_api.h).")
  endif()
endif()
