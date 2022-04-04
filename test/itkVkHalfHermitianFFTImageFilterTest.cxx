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

#include "itkVkRealToHalfHermitianForwardFFTImageFilter.h"
#include "itkVkHalfHermitianToRealInverseFFTImageFilter.h"

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
itkVkHalfHermitianFFTImageFilterTest(int argc, char * argv[])
{
  int  testNumber{ 0 };
  bool testsPassed{ true };
  {
    if (argc != 1)
    {
      std::cerr << "Missing parameters." << std::endl;
      std::cerr << "Usage: " << itkNameOfTestExecutableMacro(argv);
      std::cerr << std::endl;
      return EXIT_FAILURE;
    }

    constexpr unsigned int Dimension{ 1 };
    using RealType = float;
    using ComplexType = std::complex<RealType>;
    using RealImageType = itk::Image<RealType, Dimension>;
    using ComplexImageType = itk::Image<ComplexType, Dimension>;
    typename RealImageType::SizeType  size;
    typename RealImageType::IndexType index;
    bool                              firstPass{ true };
    // Skip trivial case where 1D image of size 1 fails.
    for (int mySize = 2; mySize <= 20; ++mySize, firstPass = false)
    {
      // We expect that anything evenly divisible by a prime number greater than 13
      // will succeed with Bluestein's Algorithm implementation in VkFFT, though
      // with less precision.

      float valueTolerance{ (mySize == 17 || mySize == 19) ? 1e-5f : 1e-6f };

      size.Fill(0);
      size[0] = mySize;
      bool xIsOdd{ mySize % 2 == 1 };

      typename RealImageType::Pointer realImage{ RealImageType::New() };
      realImage->SetRegions(size);
      realImage->Allocate();
      const RealType realZeroValue{ 0.0 };
      realImage->FillBuffer(realZeroValue);
      index.Fill(0);
      const RealType realSomeValue{ 4.567 };
      realImage->SetPixel(index, realSomeValue);
      using ForwardFilterType = itk::VkRealToHalfHermitianForwardFFTImageFilter<RealImageType>;
      typename ForwardFilterType::Pointer forwardFilter{ ForwardFilterType::New() };
      if (firstPass)
      {
        ITK_EXERCISE_BASIC_OBJECT_METHODS(
          forwardFilter, VkRealToHalfHermitianForwardFFTImageFilter, RealToHalfHermitianForwardFFTImageFilter);
      }
      forwardFilter->SetDeviceID(0);
      forwardFilter->SetInput(realImage);

      typename ComplexImageType::Pointer complexImage{ ComplexImageType::New() };
      complexImage->SetRegions(size);
      complexImage->Allocate();
      const ComplexType complexZeroValue{ 0.0, 0.0 };
      complexImage->FillBuffer(complexZeroValue);
      index.Fill(0);
      const ComplexType complexSomeValue{ 4.567, 0.0 };
      complexImage->SetPixel(index, complexSomeValue);
      using InverseFilterType = itk::VkHalfHermitianToRealInverseFFTImageFilter<ComplexImageType>;
      typename InverseFilterType::Pointer inverseFilter{ InverseFilterType::New() };
      if (firstPass)
      {
        ITK_EXERCISE_BASIC_OBJECT_METHODS(
          inverseFilter, VkHalfHermitianToRealInverseFFTImageFilter, HalfHermitianToRealInverseFFTImageFilter);
      }
      inverseFilter->SetDeviceID(0);
      inverseFilter->SetInput(complexImage);

      bool thisTestPassed{ true };
      forwardFilter->Update();
      typename ComplexImageType::Pointer        output{ forwardFilter->GetOutput() };
      const typename ComplexImageType::SizeType outputSize{ output->GetLargestPossibleRegion().GetSize() };
      const int                                 desiredOutputSize{ mySize / 2 + 1 };
      if (outputSize[0] != desiredOutputSize)
      {
        std::cout << "Size is " << outputSize[0] << " but should be " << desiredOutputSize << "." << std::endl;
        thisTestPassed = false;
      }
      for (int i = 0; i < desiredOutputSize; ++i)
      {
        index[0] = i;
        if (std::abs(output->GetPixel(index) - complexSomeValue) > valueTolerance)
        {
          std::cout << output->GetPixel(index)
                    << ": |difference| = " << std::abs(output->GetPixel(index) - complexSomeValue) << std::endl;
          thisTestPassed = false;
        }
      }
      std::cout << "Test " << ++testNumber << " (forward, size=" << mySize << ") ... "
                << (thisTestPassed ? "passed." : "failed.") << std::endl;
      testsPassed &= thisTestPassed;

      thisTestPassed = true;
      inverseFilter->SetInput(output);
      inverseFilter->SetActualXDimensionIsOdd(xIsOdd);
      inverseFilter->Update();
      typename RealImageType::Pointer        output2{ inverseFilter->GetOutput() };
      const typename RealImageType::SizeType output2Size{ output2->GetLargestPossibleRegion().GetSize() };
      if (output2Size[0] != mySize)
      {
        std::cout << "Size is " << output2Size[0] << " but should be " << mySize << "." << std::endl;
        thisTestPassed = false;
      }
      index[0] = 0;
      if (std::abs(output2->GetPixel(index) - realSomeValue) > valueTolerance)
      {
        std::cout << output2->GetPixel(index)
                  << ": |difference| = " << std::abs(output2->GetPixel(index) - realSomeValue) << std::endl;
        thisTestPassed = false;
      }
      for (int i = 1; i < mySize; ++i)
      {
        index[0] = i;
        if (std::abs(output2->GetPixel(index) - realZeroValue) > valueTolerance)
        {
          std::cout << output2->GetPixel(index)
                    << ": |difference| = " << std::abs(output2->GetPixel(index) - realZeroValue) << std::endl;
          thisTestPassed = false;
        }
      }
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
