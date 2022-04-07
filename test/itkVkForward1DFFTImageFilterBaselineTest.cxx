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

#include "itkComplexToImaginaryImageFilter.h"
#include "itkComplexToRealImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkVkForward1DFFTImageFilter.h"
#include "itkTestingMacros.h"

template <typename FFTType>
int
doTest(const char * inputImage, const char * outputImagePrefix)
{
  using ImageType = typename FFTType::InputImageType;
  using ComplexImageType = typename FFTType::OutputImageType;

  using ReaderType = itk::ImageFileReader<ImageType>;
  using RealFilterType = itk::ComplexToRealImageFilter<ComplexImageType, ImageType>;
  using ImaginaryFilterType = itk::ComplexToImaginaryImageFilter<ComplexImageType, ImageType>;
  using WriterType = itk::ImageFileWriter<ImageType>;

  typename ReaderType::Pointer          reader{ ReaderType::New() };
  typename FFTType::Pointer             fft{ FFTType::New() };
  typename RealFilterType::Pointer      realFilter{ RealFilterType::New() };
  typename ImaginaryFilterType::Pointer imaginaryFilter{ ImaginaryFilterType::New() };
  typename WriterType::Pointer          writer{ WriterType::New() };

  reader->SetFileName(inputImage);
  fft->SetInput(reader->GetOutput());
  realFilter->SetInput(fft->GetOutput());
  imaginaryFilter->SetInput(fft->GetOutput());

  writer->SetInput(realFilter->GetOutput());
  writer->SetFileName(std::string(outputImagePrefix) + "Real.mha");

  ITK_TRY_EXPECT_NO_EXCEPTION(writer->Update());

  writer->SetInput(imaginaryFilter->GetOutput());
  writer->SetFileName(std::string(outputImagePrefix) + "Imaginary.mha");

  ITK_TRY_EXPECT_NO_EXCEPTION(writer->Update());

  return EXIT_SUCCESS;
}

int
itkVkForward1DFFTImageFilterBaselineTest(int argc, char * argv[])
{
  if (argc < 3)
  {
    std::cerr << "Missing Parameters." << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " inputImage outputImagePrefix [backend]" << std::endl;
    std::cerr << std::flush;
    return EXIT_FAILURE;
  }

  using PixelType = double;
  const unsigned int Dimension = 2;

  using ImageType = itk::Image<PixelType, Dimension>;
  using FFTForwardType = itk::VkForward1DFFTImageFilter<ImageType>;

  // Instantiate a filter to exercise basic object methods
  typename FFTForwardType::Pointer fft{ FFTForwardType::New() };
  ITK_EXERCISE_BASIC_OBJECT_METHODS(fft, VkForward1DFFTImageFilter, Forward1DFFTImageFilter);

  return doTest<FFTForwardType>(argv[1], argv[2]);
}
