set(
  VkFFTBackend_SRCS
  itkVkCommon.cxx
  itkVkGlobalConfiguration.cxx
  itkVkFFTImageFilterInitFactory.cxx
)

itk_module_add_library(VkFFTBackend ${VkFFTBackend_SRCS})

if(${VKFFT_BACKEND} EQUAL 1)
  # itkVkCommon.cxx is host code; VkFFT JIT-compiles CUDA kernels at runtime
  # via nvrtc, so no nvcc device compilation is required here. VkFFT and half
  # are header-only; their include dirs arrive via VkFFTBackend_SYSTEM_INCLUDE_DIRS
  # (as for the OpenCL backend), so only the CUDA libraries are linked.
  target_link_libraries(
    VkFFTBackend
    PUBLIC
      ${CUDA_LIBRARIES}
      CUDA::cuda_driver
      ${CUDA_NVRTC_LIB}
  )
elseif(${VKFFT_BACKEND} EQUAL 3)
  target_link_libraries(VkFFTBackend PUBLIC ${OpenCL_LIBRARY})
endif()
