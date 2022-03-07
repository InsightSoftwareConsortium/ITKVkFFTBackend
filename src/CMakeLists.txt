set(VkFFTBackend_SRCS
  itkVkCommon.cxx
  itkVkGlobalConfiguration.cxx
  itkVkFFTImageFilterInitFactory.cxx
  )

itk_module_add_library(VkFFTBackend ${VkFFTBackend_SRCS})

if(${VKFFT_BACKEND} EQUAL 1)
	target_link_libraries(VkFFTBackend PUBLIC ${CUDA_LIBRARIES} cuda ${CUDA_NVRTC_LIB} VkFFT half)
  if(MSVC)
  else()
    set_source_files_properties(itkVkCommon.cpp PROPERTIES LANGUAGE CUDA)
  endif()
  target_compile_options(VkFFTBackend PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:SHELL
    -DVKFFT_BACKEND=${VKFFT_BACKEND}
    -gencode arch=compute_35,code=compute_35
    -gencode arch=compute_60,code=compute_60
    -gencode arch=compute_70,code=compute_70
    -gencode arch=compute_75,code=compute_75
    -gencode arch=compute_80,code=compute_80
    -gencode arch=compute_86,code=compute_86>")
  set_target_properties(VkFFTBackend PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(VkFFTBackend PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
elseif(${VKFFT_BACKEND} EQUAL 3)
  target_link_libraries(VkFFTBackend PUBLIC ${OpenCL_LIBRARY})
endif()
