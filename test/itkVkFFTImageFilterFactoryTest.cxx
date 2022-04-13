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

#include "itkComplexToComplex1DFFTImageFilter.h"
#include "itkVkComplexToComplex1DFFTImageFilter.h"
#include "itkVnlComplexToComplex1DFFTImageFilter.h"
#include "itkVkFFTImageFilterInitFactory.h"

#include "itkFFTImageFilterFactory.h"
#include "itkTestingMacros.h"

// Verify FFT interface classes can be instantiated with
// VkFFT backends through ITK object factory override methods

int
itkVkFFTImageFilterFactoryTest(int, char *[])
{
  using PixelType = double;
  const unsigned int Dimension = 2;
  using ComplexImageType = itk::Image<std::complex<PixelType>, Dimension>;
  using FFTBaseType = itk::ComplexToComplex1DFFTImageFilter<ComplexImageType>;
  using FFTDefaultSubclassType = itk::VnlComplexToComplex1DFFTImageFilter<ComplexImageType>;
  using FFTVkSubclassType = itk::VkComplexToComplex1DFFTImageFilter<ComplexImageType>;

  // Verify default is non-accelerated implementation
  typename FFTBaseType::Pointer fft{ FFTBaseType::New() };
  FFTDefaultSubclassType *      vnlFFT = dynamic_cast<FFTDefaultSubclassType *>(fft.GetPointer());
  ITK_TEST_EXPECT_TRUE(vnlFFT != nullptr);
  ITK_EXERCISE_BASIC_OBJECT_METHODS(vnlFFT, VnlComplexToComplex1DFFTImageFilter, ComplexToComplex1DFFTImageFilter);

  // Register factory and verify override
  using FactoryType = itk::FFTImageFilterFactory<itk::VkComplexToComplex1DFFTImageFilter>;
  typename FactoryType::Pointer factory{ FactoryType::New() };
  itk::ObjectFactoryBase::RegisterFactory(factory, itk::ObjectFactoryEnums::InsertionPosition::INSERT_AT_FRONT);

  fft = FFTBaseType::New();
  FFTVkSubclassType * vkFFT = dynamic_cast<FFTVkSubclassType *>(fft.GetPointer());
  ITK_TEST_EXPECT_TRUE(vkFFT != nullptr);
  ITK_EXERCISE_BASIC_OBJECT_METHODS(vkFFT, VkComplexToComplex1DFFTImageFilter, ComplexToComplex1DFFTImageFilter);

  // Verify factory can be removed
  itk::ObjectFactoryBase::UnRegisterFactory(factory);

  fft = FFTBaseType::New();
  vnlFFT = dynamic_cast<FFTDefaultSubclassType *>(fft.GetPointer());
  ITK_TEST_EXPECT_TRUE(vnlFFT != nullptr);
  ITK_EXERCISE_BASIC_OBJECT_METHODS(vnlFFT, VnlComplexToComplex1DFFTImageFilter, ComplexToComplex1DFFTImageFilter);

  // Verify factory initialization successfully registers factories
  using FactoryInitializerType = itk::VkFFTImageFilterInitFactory;
  FactoryInitializerType::RegisterFactories();

  fft = FFTBaseType::New();
  vkFFT = dynamic_cast<FFTVkSubclassType *>(fft.GetPointer());
  ITK_TEST_EXPECT_TRUE(vkFFT != nullptr);
  ITK_EXERCISE_BASIC_OBJECT_METHODS(vkFFT, VkComplexToComplex1DFFTImageFilter, ComplexToComplex1DFFTImageFilter);


  return EXIT_SUCCESS;
}
