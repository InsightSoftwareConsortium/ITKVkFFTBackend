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

#include "itkVkComplexToComplex1DFFTImageFilter.h"
#include "itkVkComplexToComplexFFTImageFilter.h"
#include "itkVkForward1DFFTImageFilter.h"
#include "itkVkForwardFFTImageFilter.h"
#include "itkVkHalfHermitianToRealInverseFFTImageFilter.h"
#include "itkVkInverse1DFFTImageFilter.h"
#include "itkVkInverseFFTImageFilter.h"
#include "itkVkRealToHalfHermitianForwardFFTImageFilter.h"

#include "itkVkGlobalConfiguration.h"
#include "itkTestingMacros.h"

// Verify FFT interface classes can be instantiated with
// VkFFT backends through ITK object factory override methods

template <typename FFTImageFilterType>
int
itkVkGlobalConfigurationTestProcedure()
{
  // Verify that we can set global configuration properties
  itk::VkGlobalConfiguration::SetDeviceID(1);
  ITK_TEST_SET_GET_VALUE(itk::VkGlobalConfiguration::GetDeviceID(), 1);

  // Verify global configuration properties are picked up by filters by default

  auto fftFilter = FFTImageFilterType::New();
  ITK_TEST_SET_GET_VALUE(fftFilter->GetUseVkGlobalConfiguration(), true);
  ITK_TEST_SET_GET_VALUE(fftFilter->GetDeviceID(), 1);
  fftFilter->SetDeviceID(2);
  ITK_TEST_SET_GET_VALUE(fftFilter->GetDeviceID(), 1);
  itk::VkGlobalConfiguration::SetDeviceID(0);
  ITK_TEST_SET_GET_VALUE(fftFilter->GetDeviceID(), 0);

  // Verify global configuration can be ignored via filter settings
  fftFilter->SetUseVkGlobalConfiguration(false);
  ITK_TEST_SET_GET_VALUE(fftFilter->GetUseVkGlobalConfiguration(), false);
  ITK_TEST_SET_GET_VALUE(fftFilter->GetDeviceID(), 2);
  itk::VkGlobalConfiguration::SetDeviceID(0);
  ITK_TEST_SET_GET_VALUE(fftFilter->GetDeviceID(), 2);
  fftFilter->SetDeviceID(1);
  ITK_TEST_SET_GET_VALUE(fftFilter->GetDeviceID(), 1);

  return EXIT_SUCCESS;
}

int
itkVkGlobalConfigurationTest(int, char *[])
{
  using RealImageType = itk::Image<float, 2>;
  using ComplexImageType = itk::Image<std::complex<float>, 2>;

  itkVkGlobalConfigurationTestProcedure<itk::VkComplexToComplex1DFFTImageFilter<ComplexImageType, ComplexImageType>>();
  itkVkGlobalConfigurationTestProcedure<itk::VkComplexToComplexFFTImageFilter<ComplexImageType, ComplexImageType>>();
  itkVkGlobalConfigurationTestProcedure<itk::VkForward1DFFTImageFilter<RealImageType, ComplexImageType>>();
  itkVkGlobalConfigurationTestProcedure<itk::VkForwardFFTImageFilter<RealImageType, ComplexImageType>>();
  itkVkGlobalConfigurationTestProcedure<
    itk::VkHalfHermitianToRealInverseFFTImageFilter<ComplexImageType, RealImageType>>();
  itkVkGlobalConfigurationTestProcedure<itk::VkInverse1DFFTImageFilter<ComplexImageType, RealImageType>>();
  itkVkGlobalConfigurationTestProcedure<itk::VkInverseFFTImageFilter<ComplexImageType, RealImageType>>();
  itkVkGlobalConfigurationTestProcedure<
    itk::VkRealToHalfHermitianForwardFFTImageFilter<RealImageType, ComplexImageType>>();

  return EXIT_SUCCESS;
}
