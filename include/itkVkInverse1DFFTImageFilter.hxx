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
#ifndef itkVkInverse1DFFTImageFilter_hxx
#define itkVkInverse1DFFTImageFilter_hxx

#include "itkHalfToFullHermitianImageFilter.h"
#include "itkVkInverse1DFFTImageFilter.h"
#include "itkIndent.h"
#include "itkMetaDataObject.h"
#include "itkProgressReporter.h"
#include "itkMultiThreaderBase.h"

#include <iostream>

namespace itk
{
template <typename TInputImage, typename TOutputImage>
void
VkInverse1DFFTImageFilter<TInputImage, TOutputImage>::GenerateData()
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

  // Mostly use defaults for VkCommon::VkGPU
  typename VkCommon::VkGPU vkGPU;
  vkGPU.device_id = this->GetDeviceID();

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
  vkParameters.fft = VkCommon::FFTEnum::R2FullH;
  vkParameters.PSize = sizeof(RealType);
  vkParameters.I = VkCommon::DirectionEnum::INVERSE;
  vkParameters.normalized = VkCommon::NormalizationEnum::NORMALIZED;
  for (size_t dim{ 0 }; dim < ImageDimension; ++dim)
  {
    if (this->GetDirection() != dim)
    {
      vkParameters.omitDimension[dim] = 1; // omit dimensions other than in the given direction.
    }
  }

  vkParameters.inputCPUBuffer = inputCPUBuffer;
  vkParameters.inputBufferBytes = inBytes;
  vkParameters.outputCPUBuffer = outputCPUBuffer;
  vkParameters.outputBufferBytes = outBytes;

  const VkFFTResult resFFT{ m_VkCommon.Run(vkGPU, vkParameters) };
  if (resFFT != VKFFT_SUCCESS)
  {
    std::ostringstream mesg;
    mesg << "VkFFT third-party library failed with error code " << resFFT << ".";
    itkAssertOrThrowMacro(false, mesg.str());
  }
}

template <typename TInputImage, typename TOutputImage>
void
VkInverse1DFFTImageFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "UseVkGlobalConfiguration: " << m_UseVkGlobalConfiguration << std::endl;
  os << indent << "Local DeviceID: " << m_DeviceID << std::endl;
  os << indent << "Global DeviceID: " << VkGlobalConfiguration::GetDeviceID() << std::endl;
  os << indent << "Preferred DeviceID: " << this->GetDeviceID() << std::endl;
}

template <typename TInputImage, typename TOutputImage>
typename VkInverse1DFFTImageFilter<TInputImage, TOutputImage>::SizeValueType
VkInverse1DFFTImageFilter<TInputImage, TOutputImage>::GetSizeGreatestPrimeFactor() const
{
  return SizeValueType{ m_VkCommon.GetGreatestPrimeFactor() };
}

} // end namespace itk

#endif // _itkVkInverse1DFFTImageFilter_hxx
