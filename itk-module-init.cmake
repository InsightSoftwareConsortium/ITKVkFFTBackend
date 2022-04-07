#
# Find the packages required by this module
#
set(VKFFT_BACKEND 3 CACHE STRING "1 - CUDA, 3 - OpenCL")
if(${VKFFT_BACKEND} EQUAL 1)
  find_package(CUDA 9.0 REQUIRED)
  find_library(CUDA_NVRTC_LIB libnvrtc nvrtc HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" "/usr/lib64" "/usr/local/cuda/lib64")
elseif(${VKFFT_BACKEND} EQUAL 3)
  find_package(OpenCL REQUIRED)
endif()
