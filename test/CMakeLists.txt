itk_module_test()

set(VkFFTTests
  itkMinimalStandardRandomVariateGeneratorTest.cxx
  itkVkComplexToComplexFFTImageFilterTest.cxx
  )

CreateTestDriver(VkFFT "${VkFFT-Test_LIBRARIES}" "${VkFFTTests}")

itk_add_test(NAME itkMinimalStandardRandomVariateGeneratorTest
  COMMAND VkFFTTestDriver itkMinimalStandardRandomVariateGeneratorTest
  )

itk_add_test(NAME itkVkComplexToComplexFFTImageFilterTest
  COMMAND VkFFTTestDriver
    --compare
    DATA{Baseline/itkVkComplexToComplexFFTImageFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplexFFTImageFilterTestOutput.mha
  itkVkComplexToComplexFFTImageFilterTest
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplexFFTImageFilterTestOutput.mha
  )
