itk_module_test()

set(VkFFTBackendTests
  itkVkComplexToComplexFFTImageFilterTest.cxx
  itkVkForwardInverseFFTImageFilterTest.cxx
  itkVkHalfHermitianFFTImageFilterTest.cxx
  )

include_directories(${VkFFTBackend_INCLUDE_DIRS})

CreateTestDriver(VkFFTBackend
  "${VkFFTBackend-Test_LIBRARIES};${OpenCL_LIBRARY}"
  "${VkFFTBackendTests}"
  # itkOpenCLTestHelper.cxx
  )

message(STATUS "==== Show all variables (tests/CMakeLists.txt begin) ===")
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
message(STATUS "==== Show all variables (tests/CMakeLists.txt end) ===")

# itk_add_test(NAME itkVkComplexToComplexFFTImageFilterTest
#   COMMAND VkFFTBackendTestDriver
#     --compare
#     DATA{Baseline/itkVkComplexToComplexFFTImageFilterTestOutput.mha}
#     ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplexFFTImageFilterTestOutput.mha
#   itkVkComplexToComplexFFTImageFilterTest
#     ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplexFFTImageFilterTestOutput.mha
#   )

# itk_add_test(NAME itkVkForwardInverseFFTImageFilterTest
#   COMMAND VkFFTBackendTestDriver itkVkForwardInverseFFTImageFilterTest
#   )

# itk_add_test(NAME itkVkHalfHermitianFFTImageFilterTest
#   COMMAND VkFFTBackendTestDriver itkVkHalfHermitianFFTImageFilterTest
#   )
