set(VkFFTBackend_SRCS
  itkVkCommon.cxx
  itkVkGlobalConfiguration.cxx
  )

itk_module_add_library(VkFFTBackend ${VkFFTBackend_SRCS})

if(${VKFFT_BACKEND} EQUAL 1)
  find_package(CUDA 9.0 REQUIRED)
  find_library(CUDA_NVRTC_LIB libnvrtc nvrtc HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBNVRTC_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)
	target_link_libraries(VkFFTBackend PUBLIC ${CUDA_LIBRARIES} cuda ${CUDA_NVRTC_LIB} VkFFT half)
elseif(${VKFFT_BACKEND} EQUAL 3)
target_link_libraries(VkFFTBackend PUBLIC ${OpenCL_LIBRARY})
endif()
