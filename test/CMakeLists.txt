itk_module_test()

set(VkFFTBackendTests
  itkVkComplexToComplexFFTImageFilterTest.cxx
  itkVkComplexToComplex1DFFTImageFilterBaselineTest.cxx
  itkVkComplexToComplex1DFFTImageFilterSizesTest.cxx
  itkVkDiscreteGaussianImageFilterTest.cxx
  itkVkFFTImageFilterFactoryTest.cxx
  itkVkForwardInverseFFTImageFilterTest.cxx
  itkVkForwardInverse1DFFTImageFilterTest.cxx
  itkVkForward1DFFTImageFilterBaselineTest.cxx
  itkVkGlobalConfigurationTest.cxx
  itkVkHalfHermitianFFTImageFilterTest.cxx
  itkVkInverse1DFFTImageFilterBaselineTest.cxx
  itkVkMultiResolutionPyramidImageFilterTest.cxx
  itkVkMultiResolutionPyramidImageFilterFactoryTest.cxx
  )

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
   
 itk_add_test(NAME itkVkComplexToComplex1DFFTImageFilterSizesTest
   COMMAND VkFFTBackendTestDriver
     --compare
     DATA{Baseline/itkVkComplexToComplexFFTImageFilterTestOutput.mha}
     ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplex1DFFTImageFilterSizesTestOutput.mha
   itkVkComplexToComplex1DFFTImageFilterSizesTest
     ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplex1DFFTImageFilterSizesTestOutput.mha
   )

   itk_add_test(NAME itkVkComplexToComplex1DFFTImageFilterBaselineTest
  COMMAND VkFFTBackendTestDriver
  --compare
    DATA{Input/TreeBarkTexture.png}
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplex1DFFTImageFilterBaselineTestOutput.mha
  itkVkComplexToComplex1DFFTImageFilterBaselineTest
      DATA{Input/itkForward1DFFTImageFilterTestBaselineRealFull.mhd,itkForward1DFFTImageFilterTestBaselineRealFull.raw}
      DATA{Input/itkForward1DFFTImageFilterTestBaselineImaginaryFull.mhd,itkForward1DFFTImageFilterTestBaselineImaginaryFull.raw}
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplex1DFFTImageFilterBaselineTestOutput.mha
    )

itk_add_test(NAME itkVkForward1DFFTImageFilterBaselineTest
  COMMAND VkFFTBackendTestDriver
  --compare
    DATA{Input/itkForward1DFFTImageFilterTestBaselineRealFull.mhd,itkForward1DFFTImageFilterTestBaselineRealFull.raw}
    ${ITK_TEST_OUTPUT_DIR}/itkVkForward1DFFTImageFilterTestBaselineOutputReal.mha
  --compare
    DATA{Input/itkForward1DFFTImageFilterTestBaselineImaginaryFull.mhd,itkForward1DFFTImageFilterTestBaselineImaginaryFull.raw}
    ${ITK_TEST_OUTPUT_DIR}/itkVkForward1DFFTImageFilterTestBaselineOutputImaginary.mha
  itkVkForward1DFFTImageFilterBaselineTest
    DATA{Input/TreeBarkTexture.png}
    ${ITK_TEST_OUTPUT_DIR}/itkVkForward1DFFTImageFilterTestBaselineOutput
    )
    
itk_add_test( NAME itkVkInverse1DFFTImageFilterBaselineTest
  COMMAND VkFFTBackendTestDriver
  --compare
    DATA{Input/TreeBarkTexture.png}
    ${ITK_TEST_OUTPUT_DIR}/itkVkInverse1DFFTImageFilterBaselineTest.mhd
  itkVkInverse1DFFTImageFilterBaselineTest
    DATA{Input/itkForward1DFFTImageFilterTestBaselineRealFull.mhd,itkForward1DFFTImageFilterTestBaselineRealFull.raw}
    DATA{Input/itkForward1DFFTImageFilterTestBaselineImaginaryFull.mhd,itkForward1DFFTImageFilterTestBaselineImaginaryFull.raw}
    ${ITK_TEST_OUTPUT_DIR}/itkVkInverse1DFFTImageFilterBaselineTest.mhd
    )

 itk_add_test(NAME itkVkForwardInverseFFTImageFilterTest
   COMMAND VkFFTBackendTestDriver itkVkForwardInverseFFTImageFilterTest
   )

itk_add_test(NAME itkVkForwardInverse1DFFTImageFilterTest
  COMMAND VkFFTBackendTestDriver itkVkForwardInverse1DFFTImageFilterTest
  )

 itk_add_test(NAME itkVkHalfHermitianFFTImageFilterTest
   COMMAND VkFFTBackendTestDriver itkVkHalfHermitianFFTImageFilterTest
   )

itk_add_test(NAME itkVkFFTImageFilterFactoryTest
  COMMAND VkFFTBackendTestDriver
  itkVkFFTImageFilterFactoryTest
   )

itk_add_test(NAME itkVkGlobalConfigurationTest
  COMMAND VkFFTBackendTestDriver
  itkVkGlobalConfigurationTest)

itk_add_test(NAME itkVkMultiResolutionPyramidImageFilterTest
  COMMAND VkFFTBackendTestDriver
  --compare
    DATA{Baseline/itkVkMultiResolutionPyramidImageFilterTest0.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTest0.mha
  --compare
    DATA{Baseline/itkVkMultiResolutionPyramidImageFilterTest1.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTest1.mha
  --compare
    DATA{Baseline/itkVkMultiResolutionPyramidImageFilterTest2.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTest2.mha
  --compare
    DATA{Baseline/itkVkMultiResolutionPyramidImageFilterTest3.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTest3.mha
  --compare
    DATA{Baseline/itkVkMultiResolutionPyramidImageFilterTest4.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTest4.mha
  itkVkMultiResolutionPyramidImageFilterTest
  DATA{Input/TreeBarkTexture.png}
  ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTest
  10 # kernelRadiusThreshold dim 0
  12 # kernelRadiusThreshold dim 1
  1  # threshold dimension
  0 # useShrinkFilter
  5 # numLevels
)

itk_add_test(NAME itkVkMultiResolutionPyramidImageFilterFactoryTest
  COMMAND VkFFTBackendTestDriver
  itkVkMultiResolutionPyramidImageFilterFactoryTest
   )

itk_add_test(NAME itkVkDiscreteGaussianImageFilterTest
  COMMAND VkFFTBackendTestDriver
  --compare
    DATA{Baseline/itkVkDiscreteGaussianImageFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkDiscreteGaussianImageFilterTestOutput.mha    
  itkVkDiscreteGaussianImageFilterTest
    DATA{Input/TreeBarkTexture.png}    
    ${ITK_TEST_OUTPUT_DIR}/itkVkDiscreteGaussianImageFilterTestOutput.mha
    0   # Expect spatial
    8.0 # Threshold for FFT to run
    2.0 # Sigma
  )
itk_add_test(NAME itkVkDiscreteGaussianImageFilterTest2
  COMMAND VkFFTBackendTestDriver
  --compare
    DATA{Baseline/itkVkDiscreteGaussianImageFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkDiscreteGaussianImageFilterTestOutput2.mha
  itkVkDiscreteGaussianImageFilterTest
    DATA{Input/TreeBarkTexture.png}    
    ${ITK_TEST_OUTPUT_DIR}/itkVkDiscreteGaussianImageFilterTestOutput2.mha
    1   # Expect FFT
    2.0 # Lower threshold for FFT to run
    2.0 # Sigma
  )
