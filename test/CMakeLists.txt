itk_module_test()

set(VkFFTTests
  itkMinimalStandardRandomVariateGeneratorTest.cxx
  itkVkComplexToComplexImageFilterTest.cxx
  )

CreateTestDriver(VkFFT "${VkFFT-Test_LIBRARIES}" "${VkFFTTests}")

itk_add_test(NAME itkMinimalStandardRandomVariateGeneratorTest
  COMMAND VkFFTTestDriver itkMinimalStandardRandomVariateGeneratorTest
  )

itk_add_test(NAME itkVkComplexToComplexImageFilterTest
  COMMAND VkFFTTestDriver
    --compare
    DATA{Baseline/itkVkComplexToComplexImageFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplexImageFilterTestOutput.mha
  itkVkComplexToComplexImageFilterTest
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplexImageFilterTestOutput.mha
  )
