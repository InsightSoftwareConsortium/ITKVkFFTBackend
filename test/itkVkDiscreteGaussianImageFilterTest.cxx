/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include <iostream>

#include "itkConstantBoundaryCondition.h"
#include "itkVkDiscreteGaussianImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTestingMacros.h"
#include "itkZeroFluxNeumannBoundaryCondition.h"
#include "itkImage.h"

int
itkVkDiscreteGaussianImageFilterTest(int argc, char * argv[])
{
  if (argc < 3)
  {
    std::cerr << "Missing parameters." << std::endl;
    std::cerr << "Usage:" << std::endl;
    std::cerr << itkNameOfTestExecutableMacro(argv)
              << " inputFilename outputFilename [expectFFT] [metricThreshold] [sigma] [kernelError] [kernelWidth] "
              << std::endl;
    return EXIT_FAILURE;
  }

  constexpr unsigned int ImageDimension = 2;
  using ImageType = typename itk::Image<float, ImageDimension>;

  bool         expectFFT = (argc > 3) ? (std::atoi(argv[3]) == 1) : false;
  float        fftThreshold = (argc > 4) ? std::stof(argv[4]) : 8.0f;
  float        sigma = (argc > 5) ? std::stof(argv[5]) : 0.0;
  float        kernelError = (argc > 6) ? std::stof(argv[6]) : 0.01;
  unsigned int kernelWidth = (argc > 7) ? std::stoi(argv[7]) : 32;

  typename ImageType::Pointer inputImage = itk::ReadImage<ImageType>(argv[1]);

  using FilterType = itk::VkDiscreteGaussianImageFilter<ImageType, ImageType>;
  auto filter = FilterType::New();
  ITK_EXERCISE_BASIC_OBJECT_METHODS(filter, VkDiscreteGaussianImageFilter, DiscreteGaussianImageFilter);

  // Test setting inputs

  filter->SetInput(inputImage);

  filter->SetAnticipatedPerformanceMetricThreshold(fftThreshold);
  ITK_TEST_SET_GET_VALUE(fftThreshold, filter->GetAnticipatedPerformanceMetricThreshold());

  filter->SetSigma(sigma);
  for (auto & param : filter->GetSigmaArray())
  {
    ITK_TEST_EXPECT_EQUAL(sigma, param);
  }
  for (auto & param : filter->GetVariance())
  {
    double tolerance = 1e-6;
    ITK_TEST_EXPECT_TRUE(std::fabs((sigma * sigma) - param) < tolerance);
  }

  filter->SetMaximumError(kernelError);
  for (size_t dim = 0; dim < ImageType::ImageDimension; ++dim)
  {
    ITK_TEST_SET_GET_VALUE(kernelError, filter->GetMaximumError()[dim]);
  }

  filter->SetMaximumKernelWidth(kernelWidth);
  ITK_TEST_SET_GET_VALUE(kernelWidth, filter->GetMaximumKernelWidth());

  filter->SetFilterDimensionality(ImageType::ImageDimension);
  ITK_TEST_SET_GET_VALUE(ImageType::ImageDimension, filter->GetFilterDimensionality());

  // Test with default input boundary conditions

  // Check metric value and filter to use
  filter->UpdateOutputInformation();
  filter->GetOutput()->SetRequestedRegionToLargestPossibleRegion();
  ITK_TEST_EXPECT_EQUAL(expectFFT, filter->GetUseFFT());

  // Run convolution
  ITK_TRY_EXPECT_NO_EXCEPTION(filter->Update());

  // Verify selected filter matched expectation
  ITK_TEST_EXPECT_EQUAL(filter->GetLastRunUsedFFT(), filter->GetUseFFT());
  ITK_TEST_EXPECT_EQUAL(expectFFT, filter->GetLastRunUsedFFT());
  filter->Print(std::cout);

  itk::WriteImage(filter->GetOutput(), argv[2], true);

  return EXIT_SUCCESS;
}
