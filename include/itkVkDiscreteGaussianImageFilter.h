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
#ifndef itkVkDiscreteGaussianImageFilter_h
#define itkVkDiscreteGaussianImageFilter_h

#include "itkMacro.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkFFTDiscreteGaussianImageFilter.h"
#include "VkFFTBackendExport.h"

namespace itk
{
/**
 * \class VkDiscreteGaussianImageFilter
 * \brief Blurs an image with discrete gaussian kernels. Dynamically selects
 * the blurring procedure to use in order to minimize runtime.
 *
 * Separable spatial convolution with DiscreteGaussianImageFilter
 * runs quickly for small kernel sizes but scales poorly with
 * increasing kernel size. By contrast ITK FFT convolution accelerated
 * with a VkFFT GPU backend scales slowly with increasing kernel size
 * but is typically outperformed by spatial convolution filters
 * for small kernel sizes.
 *
 * VkDiscreteGaussianImageFilter dynamically selects and runs the blurring
 * procedure expected to give the best performance for the given input image
 * and parameter set. VkBlurringPerformanceMetric is used to quantify the
 * anticipated performance tradeoff between separable spatial blurring and
 * FFT blurring for the filter's output requested region and the Gaussian
 * kernel size that will result from the input kernel parameters defining
 * variance, error, and maximum size. The metric output for the given
 * input parameters are compared with a user-defined threshold at which
 * the approximate performance tradeoff between spatial and VkFFT convolution
 * is observed to occur for the user's system. This threshold can be
 * determined via benchmarking scripts in VkFFTBackend.
 *
 * \sa GaussianOperator
 * \sa DiscreteGaussianImageFilter
 * \sa FFTDiscreteGaussianImageFilter
 *
 * \ingroup ImageEnhancement
 * \ingroup ImageFeatureExtraction
 * \ingroup ITKSmoothing
 * \ingroup VkFFTBackend
 *
 */

template <typename TInputImage, typename TOutputImage = TInputImage>
class VkDiscreteGaussianImageFilter : public DiscreteGaussianImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(VkDiscreteGaussianImageFilter);

  /** Standard class type aliases. */
  using Self = VkDiscreteGaussianImageFilter;
  using Superclass = DiscreteGaussianImageFilter<TInputImage, TOutputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(VkDiscreteGaussianImageFilter, DiscreteGaussianImageFilter);

  /** Image type information. */
  using InputImageType = typename Superclass::InputImageType;
  using OutputImageType = typename Superclass::OutputImageType;

  /** Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same. */
  static constexpr unsigned int ImageDimension = Superclass::ImageDimension;

  /** Typedef to describe kernel parameters */
  using typename Superclass::KernelType;
  using typename Superclass::RadiusType;
  using KernelSizeType = RadiusType;
  using RegionSizeType = typename OutputImageType::SizeType;

  /** Typedef for convolution */
  using BaseBlurringFilterType = DiscreteGaussianImageFilter<InputImageType, OutputImageType>;
  using SpatialBlurringFilterType = DiscreteGaussianImageFilter<InputImageType, OutputImageType>;
  using FFTBlurringFilterType = FFTDiscreteGaussianImageFilter<InputImageType, OutputImageType>;

  /** Threshold value at which spatial and FFT smoothing procedures
   *  are expected to run in approximately equivalent time. */
  itkSetMacro(AnticipatedPerformanceMetricThreshold, float);
  itkGetMacro(AnticipatedPerformanceMetricThreshold, float);

  /** Returns whether spatial or FFT smoothing was used in
   *  the last update. */
  itkGetMacro(LastRunUsedFFT, bool);

  /** Determine whether spatial (default) or FFT blurring gives the
   *  best anticipated performance. */
  bool
  GetUseFFT() const;

  /** Check anticipated performance metric for given input */
  float
  GetAnticipatedPerformanceMetric() const;

  /** Compute the anisotropic N-dimensional radius
   *  of the Gaussian kernel to use based on given
   *  input parameters. */
  RadiusType
  GetKernelRadius() const;

  /** Compute the Gaussian kernel size.
   *  size = radius * 2 + 1 */
  KernelSizeType
  GetKernelSize() const;

protected:
  VkDiscreteGaussianImageFilter() = default;
  ~VkDiscreteGaussianImageFilter() override = default;

  /** Pad region to implementation specifications */
  void
  GenerateInputRequestedRegion() override;

  /** Run either spatial or FFT blurring */
  void
  GenerateData() override;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  float m_AnticipatedPerformanceMetricThreshold = 8.0f;
  bool  m_LastRunUsedFFT = false;

  typename SpatialBlurringFilterType::Pointer m_SpatialBlurringFilter = SpatialBlurringFilterType::New();
  typename FFTBlurringFilterType::Pointer     m_FFTBlurringFilter = FFTBlurringFilterType::New();
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkVkDiscreteGaussianImageFilter.hxx"
#endif

#endif
