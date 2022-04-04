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

#include "itkComposeImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkVkInverse1DFFTImageFilter.h"
#include "itkTestingMacros.h"

template <typename FFTType>
int
doTest(const char * inputRealFullImage, const char * inputImaginaryFullImage, const char * outputImage)
{
  using ImageType = typename FFTType::OutputImageType;
  using ComplexImageType = typename FFTType::InputImageType;

  using ReaderType = itk::ImageFileReader<ImageType>;
  using JoinFilterType = itk::ComposeImageFilter<ImageType, ComplexImageType>;
  using WriterType = itk::ImageFileWriter<ImageType>;

  typename ReaderType::Pointer     readerReal{ ReaderType::New() };
  typename ReaderType::Pointer     readerImag{ ReaderType::New() };
  typename FFTType::Pointer        fft{ FFTType::New() };
  typename JoinFilterType::Pointer joinFilter{ JoinFilterType::New() };
  typename WriterType::Pointer     writer{ WriterType::New() };

  readerReal->SetFileName(inputRealFullImage);
  readerImag->SetFileName(inputImaginaryFullImage);
  joinFilter->SetInput1(readerReal->GetOutput());
  joinFilter->SetInput2(readerImag->GetOutput());
  fft->SetInput(joinFilter->GetOutput());
  writer->SetInput(fft->GetOutput());
  writer->SetFileName(outputImage);

  ITK_TRY_EXPECT_NO_EXCEPTION(writer->Update());

  return EXIT_SUCCESS;
}

int
itkVkInverse1DFFTImageFilterBaselineTest(int argc, char * argv[])
{
  if (argc < 3)
  {
    std::cerr << "Missing Parameters." << std::endl;
    std::cerr << "Usage: " << itkNameOfTestExecutableMacro(argv);
    std::cerr << " inputImagePrefix outputImage [backend]" << std::endl;
    std::cerr << std::flush;
    return EXIT_FAILURE;
  }

  using PixelType = double;
  const unsigned int Dimension = 2;

  using ComplexImageType = itk::Image<std::complex<PixelType>, Dimension>;
  using FFTInverseType = itk::VkInverse1DFFTImageFilter<ComplexImageType>;
  using ImageType = typename FFTInverseType::OutputImageType;

  // Instantiate a filter to exercise basic object methods
  typename FFTInverseType::Pointer fft{ FFTInverseType::New() };
  ITK_EXERCISE_BASIC_OBJECT_METHODS(fft, VkInverse1DFFTImageFilter, Inverse1DFFTImageFilter);

  return doTest<FFTInverseType>(argv[1], argv[2], argv[3]);
}
