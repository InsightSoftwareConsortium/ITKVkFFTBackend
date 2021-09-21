set(VkFFTBackend_SRCS
  itkVkCommon.cxx
  )

itk_module_add_library(VkFFTBackend ${VkFFTBackend_SRCS})
target_link_libraries(VkFFTBackend PUBLIC ${OpenCL_LIBRARY})
