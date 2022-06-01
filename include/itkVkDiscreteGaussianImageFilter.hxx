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
#ifndef itkVkDiscreteGaussianImageFilter_hxx
#define itkVkDiscreteGaussianImageFilter_hxx

#include "itkFFTDiscreteGaussianImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkImageToImageFilter.h"
#include "itkVkBlurringPerformanceMetric.h"
#include "itkMakeFilled.h"

namespace itk
{
template <typename TInputImage, typename TOutputImage>
void
VkDiscreteGaussianImageFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion()
{
  ImageToImageFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion();
  auto inputRegion = this->GetOutput()->GetRequestedRegion();

  // Pad the requested region by the kernel radius.
  inputRegion.PadByRadius(this->GetKernelRadius());

  // Crop the output requested region to fit within the largest
  // possible region.
  InputImageType * inputPtr = itkDynamicCastInDebugMode<InputImageType *>(this->GetPrimaryInput());
  bool             wasPartiallyInside = inputRegion.Crop(inputPtr->GetLargestPossibleRegion());
  if (!wasPartiallyInside)
  {
    itkExceptionMacro("Requested region is outside the largest possible region.")
  }
  inputPtr->SetRequestedRegion(inputRegion);
}

template <typename TInputImage, typename TOutputImage>
void
VkDiscreteGaussianImageFilter<TInputImage, TOutputImage>::GenerateData()
{
  TOutputImage * output = this->GetOutput();

  output->SetBufferedRegion(output->GetRequestedRegion());
  output->Allocate();

  // Create an internal image to protect the input image's metdata
  // (e.g. RequestedRegion). The StreamingImageFilter changes the
  // requested region as part of its normal processing.
  auto localInput = TInputImage::New();
  localInput->Graft(this->GetInput());

  BaseBlurringFilterType * smoother = nullptr;

  if (this->GetUseFFT())
  {
    smoother = static_cast<BaseBlurringFilterType *>(m_FFTBlurringFilter.GetPointer());
    m_LastRunUsedFFT = true;
  }
  else
  {
    // Set spatial-specific parameters
    m_SpatialBlurringFilter->SetInputBoundaryCondition(this->GetInputBoundaryCondition());

    smoother = static_cast<BaseBlurringFilterType *>(m_SpatialBlurringFilter.GetPointer());
    m_LastRunUsedFFT = false;
  }

  smoother->SetInput(localInput);
  smoother->SetVariance(this->GetVariance());
  smoother->SetMaximumError(this->GetMaximumError());
  smoother->SetMaximumKernelWidth(this->GetMaximumKernelWidth());
  smoother->SetFilterDimensionality(this->GetFilterDimensionality());
  smoother->SetRealBoundaryCondition(this->GetRealBoundaryCondition());
  smoother->SetUseImageSpacing(this->GetUseImageSpacing());

  // Graft this filters output onto the mini-pipeline so the mini-pipeline
  // has the correct region ivars and will write to this filters bulk data
  // output.
  smoother->GraftOutput(output);

  // Run the mini-pipeline
  smoother->Update();

  // Graft the last output of the mini-pipeline onto this filters output so
  // the final output has the correct region ivars and a handle to the final
  // bulk data
  this->GraftOutput(output);
}


/** Determine whether spatial (default) or FFT blurring gives the
 *  best anticipated performance. */

template <typename TInputImage, typename TOutputImage>
bool
VkDiscreteGaussianImageFilter<TInputImage, TOutputImage>::GetUseFFT() const
{
  return this->GetAnticipatedPerformanceMetric() > m_AnticipatedPerformanceMetricThreshold;
}

template <typename TInputImage, typename TOutputImage>
float
VkDiscreteGaussianImageFilter<TInputImage, TOutputImage>::GetAnticipatedPerformanceMetric() const
{
  // If input has not been set then default to 0
  auto input = this->GetInput();
  auto output = this->GetOutput();
  if (input == nullptr || output == nullptr)
    return 0.0f;

  auto regionSize = output->GetRequestedRegion().GetSize();
  auto kernelSize = this->GetKernelSize();
  return VkBlurringPerformanceMetric<InputImageType>::Compute(regionSize, kernelSize);
}

template <typename TInputImage, typename TOutputImage>
auto
VkDiscreteGaussianImageFilter<TInputImage, TOutputImage>::GetKernelRadius() const -> RadiusType
{
  if (this->GetUseImageSpacing() && this->GetInput() == nullptr)
  {
    // Invalid input set -> default radius 0
    return MakeFilled<RadiusType>(0);
  }

  RadiusType radius;
  for (unsigned int dim = 0; dim < ImageDimension; ++dim)
  {
    radius[dim] = this->Superclass::GetKernelRadius(dim);
  }
  return radius;
}


template <typename TInputImage, typename TOutputImage>
auto
VkDiscreteGaussianImageFilter<TInputImage, TOutputImage>::GetKernelSize() const -> KernelSizeType
{
  KernelSizeType kernelSize;
  auto           radius = GetKernelRadius();
  for (unsigned int dim = 0; dim < ImageDimension; ++dim)
  {
    kernelSize[dim] = radius[dim] * 2 + 1;
  }
  return kernelSize;
}

template <typename TInputImage, typename TOutputImage>
void
VkDiscreteGaussianImageFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Spatial blurring filter member: " << m_SpatialBlurringFilter.GetPointer() << std::endl;
  os << indent << "FFT blurring filter member: " << m_FFTBlurringFilter.GetPointer() << std::endl;
  os << indent << "Kernel radius: " << GetKernelRadius() << std::endl;
  os << indent << "Anticipated performance metric threshold: " << m_AnticipatedPerformanceMetricThreshold << std::endl;
  os << indent << "Anticipated performance metric: " << this->GetAnticipatedPerformanceMetric() << std::endl;
  os << indent << "Last run used FFT: " << m_LastRunUsedFFT << std::endl;
}

} // end namespace itk

#endif // itkVkDiscreteGaussianImageFilter_hxx
