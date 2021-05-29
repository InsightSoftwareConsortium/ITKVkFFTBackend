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

#include "itkVkComplexToComplexFFTImageFilter.h"

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
itkVkComplexToComplexFFTImageFilterTest(int argc, char * argv[])
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

    constexpr unsigned int Dimension = 2;
    using PixelType = std::complex<double>;
    using ImageType = itk::Image<PixelType, Dimension>;

    using FilterType = itk::VkComplexToComplexFFTImageFilter<ImageType>;
    typename FilterType::Pointer filter = FilterType::New();

    ITK_EXERCISE_BASIC_OBJECT_METHODS(filter, VkComplexToComplexFFTImageFilter, ComplexToComplexFFTImageFilter);

    // Create input image to avoid test dependencies.
    typename ImageType::SizeType size;
    size.Fill(128);
    typename ImageType::Pointer image = ImageType::New();
    image->SetRegions(size);
    image->Allocate();
    image->FillBuffer(1.1f);

    typename ShowProgress::Pointer showProgress = ShowProgress::New();
    filter->AddObserver(itk::ProgressEvent(), showProgress);
    filter->SetInput(image);

    using WriterType = itk::ImageFileWriter<ImageType>;
    typename WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(outputImageFileName);
    writer->SetInput(filter->GetOutput());
    writer->SetUseCompression(true);

    ITK_TRY_EXPECT_NO_EXCEPTION(writer->Update());
    std::cout << std::endl << "Test " << ++testNumber << "... passed." << std::endl;
  }

  bool testsPassed{ true };
  {
    constexpr unsigned int Dimension = 1;
    using RealType = float;
    using ComplexType = std::complex<RealType>;
    using PixelType = ComplexType;
    using ImageType = itk::Image<PixelType, Dimension>;
    typename ImageType::SizeType size;
    for (int mySize = 1; mySize <= 20; ++mySize)
    {
      size.Fill(mySize);
      typename ImageType::Pointer image = ImageType::New();
      image->SetRegions(size);
      image->Allocate();
      image->FillBuffer(0.0);
      typename ImageType::IndexType index;
      index.Fill(0);
      const ComplexType someValue{ 1.23, 4.567 };
      const ComplexType zeroValue{ 0.0, 0.0 };
      image->SetPixel(index, someValue);

      using FilterType = itk::VkComplexToComplexFFTImageFilter<ImageType>;
      typename FilterType::Pointer filter = FilterType::New();
      filter->SetInput(image);
      if (mySize == 1 || mySize == 17 || mySize == 19)
      {
        // Anything evenly divisible by a prime number greater than 13 is expected to fail.  A size of 1
        // fails too, for reasons that aren't clear.
        ITK_TRY_EXPECT_EXCEPTION(filter->Update());
        std::cout << std::flush;
        std::cerr << std::flush;
        std::cout << "Test " << ++testNumber << " (forward, size=" << mySize << ") ... passed." << std::endl;

        filter->SetTransformDirection(FilterType::TransformDirectionEnum::INVERSE);
        ITK_TRY_EXPECT_EXCEPTION(filter->Update());
        std::cout << std::flush;
        std::cerr << std::flush;
        std::cout << "Test " << ++testNumber << " (inverse, size=" << mySize << ") ... passed." << std::endl;
      }
      else
      {
        filter->Update();
        typename ImageType::Pointer        output = filter->GetOutput();
        bool                               thisTestPassed{ true };
        const typename ImageType::SizeType outputSize = output->GetLargestPossibleRegion().GetSize();
        if (outputSize[0] != mySize)
          thisTestPassed = false;
        for (int i = 0; i < mySize; ++i)
        {
          index[0] = i;
          if (output->GetPixel(index) != someValue)
            thisTestPassed = false;
        }
        std::cout << "Test " << ++testNumber << " (forward, size=" << mySize << ") ... "
                  << (thisTestPassed ? "passed." : "failed.") << std::endl;
        testsPassed &= thisTestPassed;

        filter->SetTransformDirection(FilterType::TransformDirectionEnum::INVERSE);
        filter->SetInput(output);
        filter->Update();
        typename ImageType::Pointer output2 = filter->GetOutput();
        thisTestPassed = true;
        const typename ImageType::SizeType output2Size = output2->GetLargestPossibleRegion().GetSize();
        if (output2Size[0] != mySize)
        {
          std::cout << "Size is " << output2Size[0] << " but should be " << mySize << "." << std::endl;
          thisTestPassed = false;
        }
        index[0] = 0;
        if (std::abs(output2->GetPixel(index) - someValue) > 1e-6)
        {
          std::cout << "|difference| = " << std::abs(output2->GetPixel(index) - someValue) << std::endl;
          thisTestPassed = false;
        }
        for (int i = 1; i < mySize; ++i)
        {
          index[0] = i;
          if (std::abs(output2->GetPixel(index) - zeroValue) > 1e-6)
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
  }

  if (testsPassed)
  {
    std::cout << "All tests passed." << std::endl;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}
