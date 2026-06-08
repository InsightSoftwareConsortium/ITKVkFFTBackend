set(VkFFTBackend_SRCS
  itkVkCommon.cxx
  itkVkGlobalConfiguration.cxx
  itkVkFFTImageFilterInitFactory.cxx
  )

itk_module_add_library(VkFFTBackend ${VkFFTBackend_SRCS})

if(${VKFFT_BACKEND} EQUAL 1)
  # itkVkCommon.cxx is host code; VkFFT JIT-compiles CUDA kernels at runtime
  # via nvrtc, so no nvcc device compilation is required here.
  target_link_libraries(VkFFTBackend PUBLIC ${CUDA_LIBRARIES} CUDA::cuda_driver ${CUDA_NVRTC_LIB} VkFFT half)
elseif(${VKFFT_BACKEND} EQUAL 3)
  target_link_libraries(VkFFTBackend PUBLIC ${OpenCL_LIBRARY})
endif()
