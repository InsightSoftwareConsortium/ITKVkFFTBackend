itk_module_test()

if(DEFINED OpenCL_INCLUDE_DIR)
  include_directories(SYSTEM ${OpenCL_INCLUDE_DIR})
endif()

set(VkFFTBackendTests
  itkMinimalStandardRandomVariateGeneratorTest.cxx
  itkVkComplexToComplexFFTImageFilterTest.cxx
  )

CreateTestDriver(VkFFTBackend "${VkFFTBackend-Test_LIBRARIES}" "${VkFFTBackendTests}")

itk_add_test(NAME itkMinimalStandardRandomVariateGeneratorTest
  COMMAND VkFFTBackendTestDriver itkMinimalStandardRandomVariateGeneratorTest
  )

# Restore this test when underlying code is fixed.
# itk_add_test(NAME itkVkComplexToComplexFFTImageFilterTest
#   COMMAND VkFFTBackendTestDriver
#     --compare
#     DATA{Baseline/itkVkComplexToComplexFFTImageFilterTestOutput.mha}
#     ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplexFFTImageFilterTestOutput.mha
#   itkVkComplexToComplexFFTImageFilterTest
#     ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplexFFTImageFilterTestOutput.mha
#   )
