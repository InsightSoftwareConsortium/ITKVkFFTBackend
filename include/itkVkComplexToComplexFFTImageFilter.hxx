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

#ifndef itkVkComplexToComplexFFTImageFilter_hxx
#define itkVkComplexToComplexFFTImageFilter_hxx

#include "itkVkCommon.h"
#include "itkVkComplexToComplexFFTImageFilter.h"
#include "vkFFT.h"
#include "itkImageRegionIterator.h"
#include "itkIndent.h"
#include "itkMetaDataObject.h"
#include "itkProgressReporter.h"

namespace itk
{

template <typename TImage>
VkComplexToComplexFFTImageFilter<TImage>::VkComplexToComplexFFTImageFilter()
{
  this->DynamicMultiThreadingOn();
}

template <typename TImage>
void
VkComplexToComplexFFTImageFilter<TImage>::GenerateData()
{
  // get pointers to the input and output
  const InputImageType * const input{ this->GetInput() };
  OutputImageType * const      output{ this->GetOutput() };

  if (!input || !output)
  {
    return;
  }

  // we don't have a nice progress to report, but at least this simple line
  // reports the beginning and the end of the process
  const ProgressReporter progress(this, 0, 1);

  // allocate output buffer memory
  output->SetBufferedRegion(output->GetRequestedRegion());
  output->Allocate();

  const SizeType & inputSize{ input->GetLargestPossibleRegion().GetSize() };

  const InputPixelType * const inputCPUBuffer{ input->GetBufferPointer() };
  OutputPixelType * const      outputCPUBuffer{ output->GetBufferPointer() };
  itkAssertOrThrowMacro(inputCPUBuffer != nullptr, "No CPU input buffer");
  itkAssertOrThrowMacro(outputCPUBuffer != nullptr, "No CPU output buffer");
  const SizeValueType inBytes{ input->GetLargestPossibleRegion().GetNumberOfPixels() * sizeof(InputPixelType) };
  const SizeValueType outBytes{ output->GetLargestPossibleRegion().GetNumberOfPixels() * sizeof(OutputPixelType) };
  itkAssertOrThrowMacro(inBytes == outBytes, "CPU input and output buffers are of different sizes.");

  // Mostly use defaults for VkCommon::VkGPU
  typename VkCommon::VkGPU vkGPU;
  vkGPU.device_id = m_DeviceID;

  // Describe this filter in VkCommon::VkParameters
  typename VkCommon::VkParameters vkParameters;
  if (ImageDimension > 0)
    vkParameters.X = inputSize[0];
  if (ImageDimension > 1)
    vkParameters.Y = inputSize[1];
  if (ImageDimension > 2)
    vkParameters.Z = inputSize[2];
  if (std::is_same<RealType, float>::value)
    vkParameters.P = VkCommon::PrecisionEnum::FLOAT;
  else if (std::is_same<RealType, double>::value)
    vkParameters.P = VkCommon::PrecisionEnum::DOUBLE;
  else
    itkAssertOrThrowMacro(false, "Unsupported type for real numbers.");
  vkParameters.fft = VkCommon::FFTEnum::C2C;
  vkParameters.PSize = sizeof(RealType);
  vkParameters.I = this->GetTransformDirection() == Superclass::TransformDirectionEnum::INVERSE
                     ? VkCommon::DirectionEnum::INVERSE
                     : VkCommon::DirectionEnum::FORWARD;
  vkParameters.normalized = vkParameters.I == VkCommon::DirectionEnum::INVERSE
                              ? VkCommon::NormalizationEnum::NORMALIZED
                              : VkCommon::NormalizationEnum::UNNORMALIZED;

  vkParameters.inputCPUBuffer = inputCPUBuffer;
  vkParameters.inputBufferBytes = inBytes;
  vkParameters.outputCPUBuffer = outputCPUBuffer;
  vkParameters.outputBufferBytes = outBytes;

  const VkFFTResult resFFT{ VkCommon::run(&vkGPU, &vkParameters) };
  if (resFFT != VKFFT_SUCCESS)
  {
    std::ostringstream mesg;
    mesg << "VkFFT third-party library failed with error code " << resFFT << ".";
    itkAssertOrThrowMacro(false, mesg.str());
  }
}

template <typename TImage>
void
VkComplexToComplexFFTImageFilter<TImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "DeviceID: " << m_DeviceID << std::endl;
}

template <typename TImage>
typename VkComplexToComplexFFTImageFilter<TImage>::SizeValueType
VkComplexToComplexFFTImageFilter<TImage>::GetSizeGreatestPrimeFactor() const
{
  return VkCommon::GreatestPrimeFactor;
}

} // end namespace itk

#endif // _itkVkComplexToComplexFFTImageFilter_hxx
