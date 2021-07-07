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
  )

itk_add_test(NAME itkVkComplexToComplexFFTImageFilterTest
  COMMAND VkFFTBackendTestDriver
    --compare
    DATA{Baseline/itkVkComplexToComplexFFTImageFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplexFFTImageFilterTestOutput.mha
  itkVkComplexToComplexFFTImageFilterTest
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplexFFTImageFilterTestOutput.mha
  )

itk_add_test(NAME itkVkForwardInverseFFTImageFilterTest
  COMMAND VkFFTBackendTestDriver itkVkForwardInverseFFTImageFilterTest
  )

itk_add_test(NAME itkVkHalfHermitianFFTImageFilterTest
  COMMAND VkFFTBackendTestDriver itkVkHalfHermitianFFTImageFilterTest
  )
