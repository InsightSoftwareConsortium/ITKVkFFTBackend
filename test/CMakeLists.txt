itk_module_test()

set(VkFFTBackendTests
  itkMinimalStandardRandomVariateGeneratorTest.cxx
  itkVkComplexToComplexFFTImageFilterTest.cxx
  )

CreateTestDriver(VkFFTBackend "${VkFFTBackend-Test_LIBRARIES}" "${VkFFTBackendTests}")

itk_add_test(NAME itkMinimalStandardRandomVariateGeneratorTest
  COMMAND VkFFTBackendTestDriver itkMinimalStandardRandomVariateGeneratorTest
  )

itk_add_test(NAME itkVkComplexToComplexFFTImageFilterTest
  COMMAND VkFFTBackendTestDriver
    --compare
    DATA{Baseline/itkVkComplexToComplexFFTImageFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplexFFTImageFilterTestOutput.mha
  itkVkComplexToComplexFFTImageFilterTest
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplexFFTImageFilterTestOutput.mha
  )
