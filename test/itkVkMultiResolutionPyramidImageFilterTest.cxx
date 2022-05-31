/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include <iostream>
#include <iomanip>

#include "itkVkMultiResolutionPyramidImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkMath.h"
#include "itkTestingMacros.h"

namespace
{
// The following three classes are used to support callbacks
// on the filter in the pipeline that follows later
class ShowProgressObject
{
public:
  ShowProgressObject(itk::ProcessObject * o) { m_Process = o; }
  void
  ShowProgress()
  {
    std::cout << "Progress " << m_Process->GetProgress() << std::endl;
  }
  itk::ProcessObject::Pointer m_Process;
};
} // namespace

int
itkVkMultiResolutionPyramidImageFilterTest(int argc, char * argv[])
{
  if (argc < 3)
  {
    std::cerr << "Missing Parameters." << std::endl;
    std::cerr << "Usage: " << itkNameOfTestExecutableMacro(argv);
    std::cerr << " inputImage outputImage <threshold0> [threshold1] [kernelThresholdDimension] [useShrinkFilter] "
                 "[numLevels] [expectedFFTLevelCount]"
              << std::endl;
    std::cerr << std::flush;
    return EXIT_FAILURE;
  }

  constexpr unsigned int ImageDimension = 2;
  using InputPixelType = float;
  using ImageType = itk::Image<InputPixelType, ImageDimension>;

  auto inputImage = itk::ReadImage<ImageType>(argv[1]);

  using PyramidType = itk::VkMultiResolutionPyramidImageFilter<ImageType, ImageType>;
  using ScheduleType = typename PyramidType::ScheduleType;
  using KernelSizeType = typename PyramidType::KernelSizeType;

  KernelSizeType kernelRadiusThreshold;
  if (argc == 4)
  {
    kernelRadiusThreshold.Fill(std::atoi(argv[3]));
  }
  else if (argc > 4)
  {
    kernelRadiusThreshold[0] = std::atoi(argv[3]);
    kernelRadiusThreshold[1] = std::atoi(argv[4]);
  }
  else
  {
    kernelRadiusThreshold.Fill(10);
  }

  auto         kernelThresholdDimension = (argc > 5 ? std::atoi(argv[5]) : 1);
  bool         useShrinkFilter = (argc > 6 && std::atoi(argv[6]) == 1);
  unsigned int numLevels = (argc > 7 ? std::atoi(argv[7]) : 3);
  int          expectedFFTCount = (argc > 8 ? std::atoi(argv[8]) : -1); // only test if specified

  // Set up multi-resolution pyramid
  auto pyramidFilter = PyramidType::New();
  pyramidFilter->SetInput(inputImage);

  pyramidFilter->SetUseShrinkImageFilter(useShrinkFilter);
  ITK_TEST_SET_GET_VALUE(pyramidFilter->GetUseShrinkImageFilter(), useShrinkFilter);

  // Verify metric threshold
  // Tune performance based on expected image sizes, underlying acceleration hardware, etc
  // If a desired threshold value is known then allow the user to set it directly.
  // This value can be obtained from user benchmarking applied to their specific use case,
  // i.e. specific hardware, expected image sizes, pyramid kernel sizes, etc
  pyramidFilter->SetMetricThreshold(5.2f);
  ITK_TEST_SET_GET_VALUE(pyramidFilter->GetMetricThreshold(), 5.2f);

  // Tune performance to an expected data profile
  // Here we specify that, for any input that is the size of our input image, if a pyramid level
  // uses a Gaussian kernel of radius greater than 6 then FFT smoothing should be used,
  // otherwise spatial smoothing should be used
  pyramidFilter->SetMetricThreshold(inputImage->GetLargestPossibleRegion().GetSize(), { 6, 6 });
  ITK_TEST_EXPECT_TRUE(itk::Math::FloatAlmostEqual(pyramidFilter->GetMetricThreshold(), 5.41497f, 4, 1e-5f));

  // Use default schedule for testing
  pyramidFilter->SetNumberOfLevels(numLevels);

  // Verify kernel variance and radius match expectations for default schedule
  KernelSizeType radius, prevRadius;
  unsigned int   fftCount = 0;
  float          sizeMetric;
  for (unsigned int level = 0; level < numLevels; ++level)
  {
    auto schedule = pyramidFilter->GetSchedule();
    auto variance = pyramidFilter->GetVariance(level);
    radius = pyramidFilter->GetKernelRadius(level);
    sizeMetric = pyramidFilter->ComputeMetricValue(inputImage->GetLargestPossibleRegion().GetSize(), radius);
    auto useFFT = pyramidFilter->GetUseFFT(radius);

    std::cout << "FFT will " << (useFFT ? "" : "not ") << "be used for level " << level << " with radius " << radius
              << " and metric value " << sizeMetric << std::setprecision(3) << std::endl;
    if (useFFT)
      ++fftCount;

    for (unsigned int dim = 0; dim < ImageDimension; ++dim)
    {
      // Verify variance output
      ITK_TEST_EXPECT_TRUE(itk::Math::AlmostEquals(variance[dim], itk::Math::sqr(0.5 * schedule[level][dim])));

      // Verify kernel radius output
      // Full calculations for default Gaussian size are outside the scope of this test
      // so just test that radius decreases with level
      if (level > 0)
      {
        ITK_TEST_EXPECT_TRUE(radius[dim] == 1 || prevRadius[dim] == 1 || radius[dim] < prevRadius[dim]);
      }
      else
      {
        prevRadius = radius;
      }
    }
  }

  if (expectedFFTCount != -1)
  {
    // Test number of levels for FFT smoothing matches expectations
    ITK_TEST_EXPECT_EQUAL(fftCount, expectedFFTCount);
  }

  ITK_EXERCISE_BASIC_OBJECT_METHODS(
    pyramidFilter, VkMultiResolutionPyramidImageFilter, MultiResolutionPyramidImageFilter);

  // Run the filter and track progress
  ShowProgressObject                                    progressWatch(pyramidFilter);
  itk::SimpleMemberCommand<ShowProgressObject>::Pointer command;
  command = itk::SimpleMemberCommand<ShowProgressObject>::New();
  command->SetCallbackFunction(&progressWatch, &ShowProgressObject::ShowProgress);
  pyramidFilter->AddObserver(itk::ProgressEvent(), command);
  pyramidFilter->Update();

  for (unsigned int ilevel = 0; ilevel < numLevels; ++ilevel)
  {
    itk::WriteImage(pyramidFilter->GetOutput(ilevel), argv[2] + std::to_string(ilevel) + ".mha");
  }

  return EXIT_SUCCESS;
}
