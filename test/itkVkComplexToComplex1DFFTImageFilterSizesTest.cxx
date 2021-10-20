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

#include "itkVkComplexToComplex1DFFTImageFilter.h"

#include <complex.h>

#include "itkCommand.h"
#include "itkImageFileWriter.h"
#include "itkTestingMacros.h"

namespace
{
class ShowProgress : public itk::Command
{
public:
  itkNewMacro(ShowProgress);

  void
  Execute(itk::Object * caller, const itk::EventObject & event) override
  {
    Execute((const itk::Object *)caller, event);
  }

  void
  Execute(const itk::Object * caller, const itk::EventObject & event) override
  {
    if (!itk::ProgressEvent().CheckEvent(&event))
    {
      return;
    }
    const auto * processObject = dynamic_cast<const itk::ProcessObject *>(caller);
    if (!processObject)
    {
      return;
    }
    std::cout << " " << processObject->GetProgress();
  }
};
} // namespace

int
itkVkComplexToComplex1DFFTImageFilterSizesTest(int argc, char * argv[])
{
  int testNumber{ 0 };
  {
    if (argc != 2)
    {
      std::cerr << "Missing parameters." << std::endl;
      std::cerr << "Usage: " << itkNameOfTestExecutableMacro(argv);
      std::cerr << " outputImage";
      std::cerr << std::endl;
      return EXIT_FAILURE;
    }
    const char * outputImageFileName = argv[1];

    constexpr unsigned int Dimension{ 2 };
    using ComplexType = std::complex<double>;
    using ComplexImageType = itk::Image<ComplexType, Dimension>;

    using FilterType = itk::VkComplexToComplex1DFFTImageFilter<ComplexImageType>;
    typename FilterType::Pointer filter{ FilterType::New() };
    filter->SetDeviceID(0);

    ITK_EXERCISE_BASIC_OBJECT_METHODS(filter, VkComplexToComplex1DFFTImageFilter, ComplexToComplex1DFFTImageFilter);

    // Create input image to avoid test dependencies.
    typename ComplexImageType::SizeType size;
    size.Fill(128);
    typename ComplexImageType::Pointer image{ ComplexImageType::New() };
    image->SetRegions(size);
    image->Allocate();

    typename ShowProgress::Pointer showProgress{ ShowProgress::New() };

    for (int i = 0; i < 2; ++i)
    {
      image->FillBuffer(1.2f);
      image->FillBuffer(1.1f);
      filter->AddObserver(itk::ProgressEvent(), showProgress);
      filter->SetInput(image);

      using WriterType = itk::ImageFileWriter<ComplexImageType>;
      typename WriterType::Pointer writer{ WriterType::New() };
      writer->SetFileName(outputImageFileName);
      writer->SetInput(filter->GetOutput());
      writer->SetUseCompression(true);

      ITK_TRY_EXPECT_NO_EXCEPTION(writer->Update());
      std::cout << std::endl << "Test " << ++testNumber << "... passed." << std::endl;
    }
  }

  bool testsPassed{ true };
  {
    constexpr unsigned int Dimension{ 1 };
    using RealType = float;
    using ComplexType = std::complex<RealType>;
    using ComplexImageType = itk::Image<ComplexType, Dimension>;
    typename ComplexImageType::SizeType size;
    // Skip trivial case where 1D image of size 1 fails.
    for (int mySize = 2; mySize <= 20; ++mySize)
    {
      // We expect that anything evenly divisible by a prime number greater than 13
      // will succeed with Bluestein's Algorithm implementation in VkFFT, though
      // with less precision.

      float valueTolerance = (mySize == 17 || mySize == 19) ? 1e-5 : 1e-6;

      size.Fill(0);
      size[0] = mySize;
      typename ComplexImageType::Pointer image{ ComplexImageType::New() };
      image->SetRegions(size);
      image->Allocate();
      const ComplexType zeroValue{ 0.0, 0.0 };
      image->FillBuffer(zeroValue);
      typename ComplexImageType::IndexType index;
      index.Fill(0);
      const ComplexType someValue{ 1.23, 4.567 };
      image->SetPixel(index, someValue);

      using FilterType = itk::VkComplexToComplex1DFFTImageFilter<ComplexImageType>;
      typename FilterType::Pointer filter{ FilterType::New() };
      filter->SetDeviceID(0);
      filter->SetInput(image);

      ITK_TRY_EXPECT_NO_EXCEPTION(filter->Update());
      typename ComplexImageType::Pointer        output{ filter->GetOutput() };
      bool                                      thisTestPassed{ true };
      const typename ComplexImageType::SizeType outputSize{ output->GetLargestPossibleRegion().GetSize() };
      if (outputSize[0] != mySize)
      {
        thisTestPassed = false;
      }
      for (int i = 0; i < mySize; ++i)
      {
        index[0] = i;

        if (std::abs(output->GetPixel(index) - someValue) > valueTolerance)
        {
          thisTestPassed = false;
        }
      }
      std::cout << "Test " << ++testNumber << " (forward, size=" << mySize << ") ... "
                << (thisTestPassed ? "passed." : "failed.") << std::endl;
      testsPassed &= thisTestPassed;

      filter->SetTransformDirection(FilterType::TransformDirectionType::INVERSE);
      filter->SetInput(output);
      filter->Update();
      typename ComplexImageType::Pointer output2{ filter->GetOutput() };
      thisTestPassed = true;
      const typename ComplexImageType::SizeType output2Size{ output2->GetLargestPossibleRegion().GetSize() };
      if (output2Size[0] != mySize)
      {
        std::cout << "Size is " << output2Size[0] << " but should be " << mySize << "." << std::endl;
        thisTestPassed = false;
      }
      index[0] = 0;
      if (std::abs(output2->GetPixel(index) - someValue) > valueTolerance)
      {
        std::cout << "|difference| = " << std::abs(output2->GetPixel(index) - someValue) << std::endl;
        thisTestPassed = false;
      }
      for (int i = 1; i < mySize; ++i)
      {
        index[0] = i;
        if (std::abs(output2->GetPixel(index) - zeroValue) > valueTolerance)
        {
          std::cout << "|difference| = " << std::abs(output2->GetPixel(index) - zeroValue) << std::endl;
          thisTestPassed = false;
        }
      }
      std::cout << std::endl;
      std::cout << "Test " << ++testNumber << " (inverse, size=" << mySize << ") ... "
                << (thisTestPassed ? "passed." : "failed.") << std::endl;
      testsPassed &= thisTestPassed;
    }
  }

  if (testsPassed)
  {
    std::cout << "All tests passed." << std::endl;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}
