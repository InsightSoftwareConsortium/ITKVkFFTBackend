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

# Helper: register one ctest case per <Float|Double> precision for each
# logical test. Each test driver entry accepts a leading "float" or
# "double" argument that selects the pixel-precision instantiation.
#
# Apple Silicon GPUs have no FP64 hardware, so VkFFT shader compilation
# fails for double-precision plans regardless of backend (OpenCL or
# Metal). Mark double-precision tests DISABLED on arm64-Darwin to keep
# the test run green there; CI on Linux/x86_64 still exercises them.
set(VkFFTBackend_DOUBLE_PRECISION_GPU_UNSUPPORTED FALSE)
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(VkFFTBackend_DOUBLE_PRECISION_GPU_UNSUPPORTED TRUE)
endif()

function(_vkfft_disable_on_unsupported_fp64 test_name)
  if(VkFFTBackend_DOUBLE_PRECISION_GPU_UNSUPPORTED)
    set_tests_properties(${test_name} PROPERTIES DISABLED TRUE)
  endif()
endfunction()

# -----------------------------------------------------------------------------
# ComplexToComplexFFTImageFilterTest
# -----------------------------------------------------------------------------
# TODO: upload float baselines to ITKData and re-add --compare DATA{...Float.mha}
# (initial Float ctest entry runs the FFT pipeline and relies on the test
# body's internal assertions rather than pixel-level baseline comparison).
itk_add_test(NAME itkVkComplexToComplexFFTImageFilterTestFloat
  COMMAND VkFFTBackendTestDriver
  itkVkComplexToComplexFFTImageFilterTest float
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplexFFTImageFilterTestOutputFloat.mha
  )
itk_add_test(NAME itkVkComplexToComplexFFTImageFilterTestDouble
  COMMAND VkFFTBackendTestDriver
    --compare
    DATA{Baseline/itkVkComplexToComplexFFTImageFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplexFFTImageFilterTestOutputDouble.mha
  itkVkComplexToComplexFFTImageFilterTest double
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplexFFTImageFilterTestOutputDouble.mha
  )
_vkfft_disable_on_unsupported_fp64(itkVkComplexToComplexFFTImageFilterTestDouble)

# -----------------------------------------------------------------------------
# ComplexToComplex1DFFTImageFilterSizesTest
# -----------------------------------------------------------------------------
# TODO: upload float baselines to ITKData and re-add --compare DATA{...Float.mha}.
itk_add_test(NAME itkVkComplexToComplex1DFFTImageFilterSizesTestFloat
  COMMAND VkFFTBackendTestDriver
  itkVkComplexToComplex1DFFTImageFilterSizesTest float
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplex1DFFTImageFilterSizesTestOutputFloat.mha
  )
itk_add_test(NAME itkVkComplexToComplex1DFFTImageFilterSizesTestDouble
  COMMAND VkFFTBackendTestDriver
    --compare
    DATA{Baseline/itkVkComplexToComplexFFTImageFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplex1DFFTImageFilterSizesTestOutputDouble.mha
  itkVkComplexToComplex1DFFTImageFilterSizesTest double
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplex1DFFTImageFilterSizesTestOutputDouble.mha
  )
_vkfft_disable_on_unsupported_fp64(itkVkComplexToComplex1DFFTImageFilterSizesTestDouble)

# -----------------------------------------------------------------------------
# ComplexToComplex1DFFTImageFilterBaselineTest
# -----------------------------------------------------------------------------
itk_add_test(NAME itkVkComplexToComplex1DFFTImageFilterBaselineTestFloat
  COMMAND VkFFTBackendTestDriver
    --compare
    DATA{Input/TreeBarkTexture.png}
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplex1DFFTImageFilterBaselineTestOutputFloat.mha
  itkVkComplexToComplex1DFFTImageFilterBaselineTest float
    DATA{Input/itkForward1DFFTImageFilterTestBaselineRealFull.mhd,itkForward1DFFTImageFilterTestBaselineRealFull.raw}
    DATA{Input/itkForward1DFFTImageFilterTestBaselineImaginaryFull.mhd,itkForward1DFFTImageFilterTestBaselineImaginaryFull.raw}
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplex1DFFTImageFilterBaselineTestOutputFloat.mha
  )
itk_add_test(NAME itkVkComplexToComplex1DFFTImageFilterBaselineTestDouble
  COMMAND VkFFTBackendTestDriver
    --compare
    DATA{Input/TreeBarkTexture.png}
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplex1DFFTImageFilterBaselineTestOutputDouble.mha
  itkVkComplexToComplex1DFFTImageFilterBaselineTest double
    DATA{Input/itkForward1DFFTImageFilterTestBaselineRealFull.mhd,itkForward1DFFTImageFilterTestBaselineRealFull.raw}
    DATA{Input/itkForward1DFFTImageFilterTestBaselineImaginaryFull.mhd,itkForward1DFFTImageFilterTestBaselineImaginaryFull.raw}
    ${ITK_TEST_OUTPUT_DIR}/itkVkComplexToComplex1DFFTImageFilterBaselineTestOutputDouble.mha
  )
_vkfft_disable_on_unsupported_fp64(itkVkComplexToComplex1DFFTImageFilterBaselineTestDouble)

# -----------------------------------------------------------------------------
# Forward1DFFTImageFilterBaselineTest
# -----------------------------------------------------------------------------
itk_add_test(NAME itkVkForward1DFFTImageFilterBaselineTestFloat
  COMMAND VkFFTBackendTestDriver
    --compare
    DATA{Input/itkForward1DFFTImageFilterTestBaselineRealFull.mhd,itkForward1DFFTImageFilterTestBaselineRealFull.raw}
    ${ITK_TEST_OUTPUT_DIR}/itkVkForward1DFFTImageFilterTestBaselineOutputFloatReal.mha
    --compare
    DATA{Input/itkForward1DFFTImageFilterTestBaselineImaginaryFull.mhd,itkForward1DFFTImageFilterTestBaselineImaginaryFull.raw}
    ${ITK_TEST_OUTPUT_DIR}/itkVkForward1DFFTImageFilterTestBaselineOutputFloatImaginary.mha
  itkVkForward1DFFTImageFilterBaselineTest float
    DATA{Input/TreeBarkTexture.png}
    ${ITK_TEST_OUTPUT_DIR}/itkVkForward1DFFTImageFilterTestBaselineOutputFloat
  )
itk_add_test(NAME itkVkForward1DFFTImageFilterBaselineTestDouble
  COMMAND VkFFTBackendTestDriver
    --compare
    DATA{Input/itkForward1DFFTImageFilterTestBaselineRealFull.mhd,itkForward1DFFTImageFilterTestBaselineRealFull.raw}
    ${ITK_TEST_OUTPUT_DIR}/itkVkForward1DFFTImageFilterTestBaselineOutputDoubleReal.mha
    --compare
    DATA{Input/itkForward1DFFTImageFilterTestBaselineImaginaryFull.mhd,itkForward1DFFTImageFilterTestBaselineImaginaryFull.raw}
    ${ITK_TEST_OUTPUT_DIR}/itkVkForward1DFFTImageFilterTestBaselineOutputDoubleImaginary.mha
  itkVkForward1DFFTImageFilterBaselineTest double
    DATA{Input/TreeBarkTexture.png}
    ${ITK_TEST_OUTPUT_DIR}/itkVkForward1DFFTImageFilterTestBaselineOutputDouble
  )
_vkfft_disable_on_unsupported_fp64(itkVkForward1DFFTImageFilterBaselineTestDouble)

# -----------------------------------------------------------------------------
# Inverse1DFFTImageFilterBaselineTest
# -----------------------------------------------------------------------------
itk_add_test(NAME itkVkInverse1DFFTImageFilterBaselineTestFloat
  COMMAND VkFFTBackendTestDriver
    --compare
    DATA{Input/TreeBarkTexture.png}
    ${ITK_TEST_OUTPUT_DIR}/itkVkInverse1DFFTImageFilterBaselineTestFloat.mhd
  itkVkInverse1DFFTImageFilterBaselineTest float
    DATA{Input/itkForward1DFFTImageFilterTestBaselineRealFull.mhd,itkForward1DFFTImageFilterTestBaselineRealFull.raw}
    DATA{Input/itkForward1DFFTImageFilterTestBaselineImaginaryFull.mhd,itkForward1DFFTImageFilterTestBaselineImaginaryFull.raw}
    ${ITK_TEST_OUTPUT_DIR}/itkVkInverse1DFFTImageFilterBaselineTestFloat.mhd
  )
itk_add_test(NAME itkVkInverse1DFFTImageFilterBaselineTestDouble
  COMMAND VkFFTBackendTestDriver
    --compare
    DATA{Input/TreeBarkTexture.png}
    ${ITK_TEST_OUTPUT_DIR}/itkVkInverse1DFFTImageFilterBaselineTestDouble.mhd
  itkVkInverse1DFFTImageFilterBaselineTest double
    DATA{Input/itkForward1DFFTImageFilterTestBaselineRealFull.mhd,itkForward1DFFTImageFilterTestBaselineRealFull.raw}
    DATA{Input/itkForward1DFFTImageFilterTestBaselineImaginaryFull.mhd,itkForward1DFFTImageFilterTestBaselineImaginaryFull.raw}
    ${ITK_TEST_OUTPUT_DIR}/itkVkInverse1DFFTImageFilterBaselineTestDouble.mhd
  )
_vkfft_disable_on_unsupported_fp64(itkVkInverse1DFFTImageFilterBaselineTestDouble)

# -----------------------------------------------------------------------------
# ForwardInverseFFTImageFilterTest (round-trip, no --compare)
# -----------------------------------------------------------------------------
itk_add_test(NAME itkVkForwardInverseFFTImageFilterTestFloat
  COMMAND VkFFTBackendTestDriver itkVkForwardInverseFFTImageFilterTest float
  )
itk_add_test(NAME itkVkForwardInverseFFTImageFilterTestDouble
  COMMAND VkFFTBackendTestDriver itkVkForwardInverseFFTImageFilterTest double
  )
_vkfft_disable_on_unsupported_fp64(itkVkForwardInverseFFTImageFilterTestDouble)

# pocl (CPU OpenCL) computes VkFFT's size-19 Bluestein inverse incorrectly, so
# the hosted pocl-backed leg runs the round-trip tests capped at size 16
# (radix-2/3/5/7 plus Bluestein primes 11/13). Real GPUs run the full sweep above.
itk_add_test(NAME itkVkForwardInverseFFTImageFilterTestFloatPoclSafe
  COMMAND VkFFTBackendTestDriver itkVkForwardInverseFFTImageFilterTest float 16
  )
itk_add_test(NAME itkVkForwardInverseFFTImageFilterTestDoublePoclSafe
  COMMAND VkFFTBackendTestDriver itkVkForwardInverseFFTImageFilterTest double 16
  )
_vkfft_disable_on_unsupported_fp64(itkVkForwardInverseFFTImageFilterTestDoublePoclSafe)

# -----------------------------------------------------------------------------
# ForwardInverse1DFFTImageFilterTest
# -----------------------------------------------------------------------------
itk_add_test(NAME itkVkForwardInverse1DFFTImageFilterTestFloat
  COMMAND VkFFTBackendTestDriver itkVkForwardInverse1DFFTImageFilterTest float
  )
itk_add_test(NAME itkVkForwardInverse1DFFTImageFilterTestDouble
  COMMAND VkFFTBackendTestDriver itkVkForwardInverse1DFFTImageFilterTest double
  )
_vkfft_disable_on_unsupported_fp64(itkVkForwardInverse1DFFTImageFilterTestDouble)

itk_add_test(NAME itkVkForwardInverse1DFFTImageFilterTestFloatPoclSafe
  COMMAND VkFFTBackendTestDriver itkVkForwardInverse1DFFTImageFilterTest float 16
  )
itk_add_test(NAME itkVkForwardInverse1DFFTImageFilterTestDoublePoclSafe
  COMMAND VkFFTBackendTestDriver itkVkForwardInverse1DFFTImageFilterTest double 16
  )
_vkfft_disable_on_unsupported_fp64(itkVkForwardInverse1DFFTImageFilterTestDoublePoclSafe)

# -----------------------------------------------------------------------------
# HalfHermitianFFTImageFilterTest
# -----------------------------------------------------------------------------
itk_add_test(NAME itkVkHalfHermitianFFTImageFilterTestFloat
  COMMAND VkFFTBackendTestDriver itkVkHalfHermitianFFTImageFilterTest float
  )
itk_add_test(NAME itkVkHalfHermitianFFTImageFilterTestDouble
  COMMAND VkFFTBackendTestDriver itkVkHalfHermitianFFTImageFilterTest double
  )
_vkfft_disable_on_unsupported_fp64(itkVkHalfHermitianFFTImageFilterTestDouble)

itk_add_test(NAME itkVkHalfHermitianFFTImageFilterTestFloatPoclSafe
  COMMAND VkFFTBackendTestDriver itkVkHalfHermitianFFTImageFilterTest float 16
  )
itk_add_test(NAME itkVkHalfHermitianFFTImageFilterTestDoublePoclSafe
  COMMAND VkFFTBackendTestDriver itkVkHalfHermitianFFTImageFilterTest double 16
  )
_vkfft_disable_on_unsupported_fp64(itkVkHalfHermitianFFTImageFilterTestDoublePoclSafe)

# -----------------------------------------------------------------------------
# FFTImageFilterFactoryTest (instantiation only — runs on all platforms)
# -----------------------------------------------------------------------------
itk_add_test(NAME itkVkFFTImageFilterFactoryTestFloat
  COMMAND VkFFTBackendTestDriver
  itkVkFFTImageFilterFactoryTest float
   )
itk_add_test(NAME itkVkFFTImageFilterFactoryTestDouble
  COMMAND VkFFTBackendTestDriver
  itkVkFFTImageFilterFactoryTest double
   )

# -----------------------------------------------------------------------------
# GlobalConfigurationTest
# -----------------------------------------------------------------------------
itk_add_test(NAME itkVkGlobalConfigurationTestFloat
  COMMAND VkFFTBackendTestDriver
  itkVkGlobalConfigurationTest float)
itk_add_test(NAME itkVkGlobalConfigurationTestDouble
  COMMAND VkFFTBackendTestDriver
  itkVkGlobalConfigurationTest double)

# -----------------------------------------------------------------------------
# MultiResolutionPyramidImageFilterTest
# -----------------------------------------------------------------------------
itk_add_test(NAME itkVkMultiResolutionPyramidImageFilterTestFloat
  COMMAND VkFFTBackendTestDriver
  --compare
    DATA{Baseline/itkVkMultiResolutionPyramidImageFilterTest0.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTestFloat0.mha
  --compare
    DATA{Baseline/itkVkMultiResolutionPyramidImageFilterTest1.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTestFloat1.mha
  --compare
    DATA{Baseline/itkVkMultiResolutionPyramidImageFilterTest2.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTestFloat2.mha
  --compare
    DATA{Baseline/itkVkMultiResolutionPyramidImageFilterTest3.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTestFloat3.mha
  --compare
    DATA{Baseline/itkVkMultiResolutionPyramidImageFilterTest4.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTestFloat4.mha
  itkVkMultiResolutionPyramidImageFilterTest float
  DATA{Input/TreeBarkTexture.png}
  ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTestFloat
  10 # kernelRadiusThreshold dim 0
  12 # kernelRadiusThreshold dim 1
  1  # threshold dimension
  0  # useShrinkFilter
  5  # numLevels
)
itk_add_test(NAME itkVkMultiResolutionPyramidImageFilterTestDouble
  COMMAND VkFFTBackendTestDriver
  --compare
    DATA{Baseline/itkVkMultiResolutionPyramidImageFilterTest0.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTestDouble0.mha
  --compare
    DATA{Baseline/itkVkMultiResolutionPyramidImageFilterTest1.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTestDouble1.mha
  --compare
    DATA{Baseline/itkVkMultiResolutionPyramidImageFilterTest2.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTestDouble2.mha
  --compare
    DATA{Baseline/itkVkMultiResolutionPyramidImageFilterTest3.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTestDouble3.mha
  --compare
    DATA{Baseline/itkVkMultiResolutionPyramidImageFilterTest4.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTestDouble4.mha
  itkVkMultiResolutionPyramidImageFilterTest double
  DATA{Input/TreeBarkTexture.png}
  ${ITK_TEST_OUTPUT_DIR}/itkVkMultiResolutionPyramidImageFilterTestDouble
  10 # kernelRadiusThreshold dim 0
  12 # kernelRadiusThreshold dim 1
  1  # threshold dimension
  0  # useShrinkFilter
  5  # numLevels
)
_vkfft_disable_on_unsupported_fp64(itkVkMultiResolutionPyramidImageFilterTestDouble)

# -----------------------------------------------------------------------------
# MultiResolutionPyramidImageFilterFactoryTest
# -----------------------------------------------------------------------------
itk_add_test(NAME itkVkMultiResolutionPyramidImageFilterFactoryTestFloat
  COMMAND VkFFTBackendTestDriver
  itkVkMultiResolutionPyramidImageFilterFactoryTest float
   )
itk_add_test(NAME itkVkMultiResolutionPyramidImageFilterFactoryTestDouble
  COMMAND VkFFTBackendTestDriver
  itkVkMultiResolutionPyramidImageFilterFactoryTest double
   )

# -----------------------------------------------------------------------------
# DiscreteGaussianImageFilterTest (2 entries: expect-spatial and expect-FFT)
# -----------------------------------------------------------------------------
itk_add_test(NAME itkVkDiscreteGaussianImageFilterTestFloat
  COMMAND VkFFTBackendTestDriver
  --compare
    DATA{Baseline/itkVkDiscreteGaussianImageFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkDiscreteGaussianImageFilterTestOutputFloat.mha
  itkVkDiscreteGaussianImageFilterTest float
    DATA{Input/TreeBarkTexture.png}
    ${ITK_TEST_OUTPUT_DIR}/itkVkDiscreteGaussianImageFilterTestOutputFloat.mha
    0   # Expect spatial
    8.0 # Threshold for FFT to run
    2.0 # Sigma
  )
itk_add_test(NAME itkVkDiscreteGaussianImageFilterTestDouble
  COMMAND VkFFTBackendTestDriver
  --compare
    DATA{Baseline/itkVkDiscreteGaussianImageFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkDiscreteGaussianImageFilterTestOutputDouble.mha
  itkVkDiscreteGaussianImageFilterTest double
    DATA{Input/TreeBarkTexture.png}
    ${ITK_TEST_OUTPUT_DIR}/itkVkDiscreteGaussianImageFilterTestOutputDouble.mha
    0   # Expect spatial
    8.0 # Threshold for FFT to run
    2.0 # Sigma
  )
_vkfft_disable_on_unsupported_fp64(itkVkDiscreteGaussianImageFilterTestDouble)

itk_add_test(NAME itkVkDiscreteGaussianImageFilterTest2Float
  COMMAND VkFFTBackendTestDriver
  --compare
    DATA{Baseline/itkVkDiscreteGaussianImageFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkDiscreteGaussianImageFilterTestOutput2Float.mha
  itkVkDiscreteGaussianImageFilterTest float
    DATA{Input/TreeBarkTexture.png}
    ${ITK_TEST_OUTPUT_DIR}/itkVkDiscreteGaussianImageFilterTestOutput2Float.mha
    1   # Expect FFT
    2.0 # Lower threshold for FFT to run
    2.0 # Sigma
  )
itk_add_test(NAME itkVkDiscreteGaussianImageFilterTest2Double
  COMMAND VkFFTBackendTestDriver
  --compare
    DATA{Baseline/itkVkDiscreteGaussianImageFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkVkDiscreteGaussianImageFilterTestOutput2Double.mha
  itkVkDiscreteGaussianImageFilterTest double
    DATA{Input/TreeBarkTexture.png}
    ${ITK_TEST_OUTPUT_DIR}/itkVkDiscreteGaussianImageFilterTestOutput2Double.mha
    1   # Expect FFT
    2.0 # Lower threshold for FFT to run
    2.0 # Sigma
  )
_vkfft_disable_on_unsupported_fp64(itkVkDiscreteGaussianImageFilterTest2Double)

# Disable every FFT-touching test when VKFFT_BACKEND=4 (Level Zero) is selected
# but no Level Zero driver was detected at configure time (see the probe in the
# top-level CMakeLists.txt). The lightweight factory / global-configuration
# tests still run because they exercise no GPU code.
if(VKFFT_BACKEND EQUAL 4 AND NOT VkFFTBackend_LEVEL_ZERO_RUNTIME_AVAILABLE)
  foreach(_stem
    itkVkComplexToComplexFFTImageFilterTest
    itkVkComplexToComplex1DFFTImageFilterSizesTest
    itkVkComplexToComplex1DFFTImageFilterBaselineTest
    itkVkForward1DFFTImageFilterBaselineTest
    itkVkInverse1DFFTImageFilterBaselineTest
    itkVkForwardInverseFFTImageFilterTest
    itkVkForwardInverse1DFFTImageFilterTest
    itkVkHalfHermitianFFTImageFilterTest
    itkVkMultiResolutionPyramidImageFilterTest
    itkVkDiscreteGaussianImageFilterTest
    itkVkDiscreteGaussianImageFilterTest2
  )
    foreach(_suffix Float Double FloatPoclSafe DoublePoclSafe)
      if(TEST ${_stem}${_suffix})
        set_tests_properties(${_stem}${_suffix} PROPERTIES DISABLED TRUE)
      endif()
    endforeach()
  endforeach()
endif()
