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

// Regression test for the per-Update GPU resource leak.
//
// A single reused Vk FFT filter driven for many transforms (the pattern an
// iterative registration uses) previously created and leaked one GPU context
// per Update(), exhausting device memory and aborting with a VkFFT
// out-of-memory error after a few dozen iterations. With the context cached
// across calls, device memory stays flat and the loop runs to completion.

#include "itkVkRealToHalfHermitianForwardFFTImageFilter.h"
#include "itkVkHalfHermitianToRealInverseFFTImageFilter.h"

#include "itkImageRegionIterator.h"
#include "itkTestingMacros.h"

#include <complex>
#include <iostream>
#include <string>

template <typename PrecisionType>
int
runRepeatedTransformLeakTest(unsigned int size, unsigned int iterations)
{
  constexpr unsigned int Dimension{ 3 };
  using RealImageType = itk::Image<PrecisionType, Dimension>;
  using ComplexImageType = itk::Image<std::complex<PrecisionType>, Dimension>;

  typename RealImageType::SizeType regionSize;
  regionSize.Fill(size);
  auto input = RealImageType::New();
  input->SetRegions(regionSize);
  input->Allocate();
  input->FillBuffer(PrecisionType{ 1 });

  // A single forward+inverse pair reused for every iteration, exactly as a
  // registration loop reuses its smoothing filters.
  using ForwardFilterType = itk::VkRealToHalfHermitianForwardFFTImageFilter<RealImageType>;
  using InverseFilterType = itk::VkHalfHermitianToRealInverseFFTImageFilter<ComplexImageType>;
  auto forwardFilter = ForwardFilterType::New();
  auto inverseFilter = InverseFilterType::New();
  forwardFilter->SetDeviceID(0);
  inverseFilter->SetDeviceID(0);
  forwardFilter->SetInput(input);
  inverseFilter->SetInput(forwardFilter->GetOutput());

  for (unsigned int i{ 0 }; i < iterations; ++i)
  {
    // Fresh input data each iteration forces a real recomputation (new CPU
    // buffer contents), reproducing the per-call pattern that leaked.
    input->FillBuffer(static_cast<PrecisionType>((i % 7) + 1));
    input->Modified();
    ITK_TRY_EXPECT_NO_EXCEPTION(inverseFilter->UpdateLargestPossibleRegion());
  }

  // Round-trip sanity: inverse(forward(constant)) is the same constant.
  inverseFilter->GetOutput()->Update();
  const auto                              lastValue = static_cast<PrecisionType>(((iterations - 1) % 7) + 1);
  itk::ImageRegionIterator<RealImageType> it(inverseFilter->GetOutput(),
                                             inverseFilter->GetOutput()->GetBufferedRegion());
  const PrecisionType                     tolerance{ static_cast<PrecisionType>(1e-3) };
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    if (std::abs(it.Get() - lastValue) > tolerance)
    {
      std::cerr << "Round-trip mismatch: got " << it.Get() << " expected " << lastValue << std::endl;
      return EXIT_FAILURE;
    }
  }

  std::cout << "Completed " << iterations << " forward+inverse transforms at " << size << "^3 without GPU resource "
            << "exhaustion." << std::endl;
  return EXIT_SUCCESS;
}

int
itkVkFFTRepeatedTransformLeakTest(int argc, char * argv[])
{
  const std::string  precision{ (argc > 1) ? argv[1] : "float" };
  const unsigned int size{ (argc > 2) ? static_cast<unsigned int>(std::stoul(argv[2])) : 64 };
  const unsigned int iterations{ (argc > 3) ? static_cast<unsigned int>(std::stoul(argv[3])) : 200 };

  if (precision == "double")
  {
    return runRepeatedTransformLeakTest<double>(size, iterations);
  }
  if (precision == "float")
  {
    return runRepeatedTransformLeakTest<float>(size, iterations);
  }
  std::cerr << "Unknown precision '" << precision << "'. Expected 'float' or 'double'." << std::endl;
  return EXIT_FAILURE;
}
