set(ITK_WRAP_COMPLEX_FLOAT OFF)
if("CF" IN_LIST WRAP_ITK_COMPLEX_REAL)
  set(ITK_WRAP_COMPLEX_FLOAT ON)
endif()

set(ITK_WRAP_COMPLEX_DOUBLE OFF)
if("CD" IN_LIST WRAP_ITK_COMPLEX_REAL)
  set(ITK_WRAP_COMPLEX_DOUBLE ON)
endif()

itk_wrap_module("VkFFTBackend")
itk_auto_load_submodules()
itk_end_wrap_module()
