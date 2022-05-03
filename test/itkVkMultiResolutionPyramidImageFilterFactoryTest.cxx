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

#include <complex>
#include <string>

#include "itkMultiResolutionPyramidImageFilter.h"
#include "itkVkMultiResolutionPyramidImageFilter.h"

#include "itkVkMultiResolutionPyramidImageFilterFactory.h"
#include "itkTestingMacros.h"

// Verify MultiResolutionPyramidImageFilter can be overriden
// with spatial+FFT implementation through object factory override

int
itkVkMultiResolutionPyramidImageFilterFactoryTest(int, char *[])
{
  using PixelType = double;
  constexpr unsigned int Dimension{ 2 };
  using ImageType = itk::Image<PixelType, Dimension>;
  using BaseFilterType = itk::MultiResolutionPyramidImageFilter<ImageType, ImageType>;
  using VkSubclassType = itk::VkMultiResolutionPyramidImageFilter<ImageType, ImageType>;

  // Verify default is non-accelerated implementation
  typename BaseFilterType::Pointer baseFilter = BaseFilterType::New();
  VkSubclassType *                 derivedFilter = dynamic_cast<VkSubclassType *>(baseFilter.GetPointer());
  ITK_TEST_EXPECT_TRUE(derivedFilter == nullptr);
  ITK_EXERCISE_BASIC_OBJECT_METHODS(baseFilter, MultiResolutionPyramidImageFilter, ImageToImageFilter);

  // Register factory and verify override
  itk::VkMultiResolutionPyramidImageFilterFactory::RegisterOneFactory();

  baseFilter = BaseFilterType::New();
  derivedFilter = dynamic_cast<VkSubclassType *>(baseFilter.GetPointer());
  ITK_TEST_EXPECT_TRUE(derivedFilter != nullptr);
  ITK_EXERCISE_BASIC_OBJECT_METHODS(
    derivedFilter, VkMultiResolutionPyramidImageFilter, MultiResolutionPyramidImageFilter);

  return EXIT_SUCCESS;
}
