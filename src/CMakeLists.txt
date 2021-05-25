set(VkFFTBackend_SRCS
  itkVkCommon.cxx
  itkMinimalStandardRandomVariateGenerator.cxx
  )

itk_module_add_library(VkFFTBackend ${VkFFTBackend_SRCS})
