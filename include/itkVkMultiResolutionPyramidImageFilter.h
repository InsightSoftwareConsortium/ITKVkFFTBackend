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
#ifndef itkVkMultiResolutionPyramidImageFilter_h
#define itkVkMultiResolutionPyramidImageFilter_h

#include "itkMultiResolutionPyramidImageFilter.h"

#include "itkDiscreteGaussianImageFilter.h"
#include "itkFFTDiscreteGaussianImageFilter.h"
#include "itkVector.h"
#include "itkMacro.h"
#include "VkFFTBackendExport.h"

#include <string>

namespace itk
{

/** \class VkMultiResolutionPyramidImageFilter
 * \brief Creates a multi-resolution pyramid with FFT acceleration
 *
 * VkMultiResolutionPyramidImageFilter re-implements a framework
 * for creating an image pyramid as laid out in
 * MultiResolutionPyramidImageFilter. Conditional logic is added
 * to preemptively select the optimal image smoothing pipeline
 * that is expected to give the best performance for different
 * pyramid levels.
 *
 * Separable spatial convolution with DiscreteGaussianImageFilter
 * runs quickly for small kernel sizes but scales poorly with
 * increasing kernel size. By contrast ITK FFT convolution accelerated
 * with a VkFFT GPU backend scales slowly with increasing kernel size
 * but is typically outperformed by spatial convolution filters
 * for small kernel sizes.
 *
 * VkMultiResolutionPyramidImageFilter allows the user to fix the
 * metric threshold at which a performance tradeoff is expected
 * between spatial and FFT convolution. The exact threshold depends
 * on user hardware and can be estimated through benchmarking with
 * scripts in the ITKVkFFTBackend repository.
 *
 * By mitigating blurring times on levels with large kernel sizes
 * VkMultiResolutionPyramidImageFilter has been observed to run in
 * as little as 50% of the time of its base class.
 *
 * See documentation of MultiResolutionPyramidImageFilter
 * for information on how to specify a multi-resolution schedule.
 *
 * \sa MultiResolutionPyramidImageFilter
 * \sa DiscreteGaussianImageFilter
 * \sa FFTDiscreteGaussianImageFilter
 * \sa ShrinkImageFilter
 * \sa VkBlurringPerformanceMetric
 *
 * \ingroup VkFFTBackend
 * \ingroup PyramidImageFilter
 * \ingroup ITKRegistrationCommon
 */
template <typename TInputImage, typename TOutputImage>
class ITK_TEMPLATE_EXPORT VkMultiResolutionPyramidImageFilter
  : public MultiResolutionPyramidImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(VkMultiResolutionPyramidImageFilter);

  /** Standard class type aliases. */
  using Self = VkMultiResolutionPyramidImageFilter;
  using Superclass = MultiResolutionPyramidImageFilter<TInputImage, TOutputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(VkMultiResolutionPyramidImageFilter, MultiResolutionPyramidImageFilter);

  /** ImageDimension enumeration. */
  static constexpr unsigned int ImageDimension = TInputImage::ImageDimension;

  /** Inherit types from Superclass. */
  using typename Superclass::InputImageType;
  using typename Superclass::OutputImageType;
  using typename Superclass::InputImagePointer;
  using typename Superclass::OutputImagePointer;
  using typename Superclass::InputImageConstPointer;
  using InputSizeType = typename InputImageType::SizeType;
  using OutputPixelType = typename OutputImageType::PixelType;
  using OutputSizeType = typename OutputImageType::SizeType;
  using typename Superclass::ScheduleType;

  using VarianceType = itk::Vector<double, ImageDimension>;
  using KernelSizeType = OutputSizeType;

  /** Types for acceleration.
   *  Assumes and does not verify that FFT backend is accelerated. */
  using BaseSmootherType = DiscreteGaussianImageFilter<OutputImageType, OutputImageType>;
  using SpatialSmootherType = DiscreteGaussianImageFilter<OutputImageType, OutputImageType>;
  using FFTSmootherType = FFTDiscreteGaussianImageFilter<OutputImageType, OutputImageType>;

  /** Set the metric threshold to decide between
   *  accelerated methods such as CPU-based separable smoothing
   *  versus GPU-based FFT smoothing.
   *  We can predictively compare spatial and FFT smoothing
   *  performance using the following metric:
   *
   *  f(i,j,k,x,y,z) = log((i + j + k) * x * y * z)
   *
   *  where i,j,k are the dimensions of the kernel for a given
   *  pyramid level and x,y,z are the dimensions of the
   *  output image region.
   *
   *  The equation above approximates the difference in runtime complexity
   *  between separable spatial Gaussian smoothing and FFT Gaussian smoothing.
   *  Under separable smoothing each pixel [xi,yi,zi] is used in computation
   *  approximately (i + j + k) times. FFT smoothing meanwhile has significant
   *  overhead in setup but scales much more slowly with kernel and image sizes.
   *  As a result there is an approximate threshold where GPU-accelerated
   *  smoothing outperforms spatial smoothing for a given pyramid level.
   *
   *  The default threshold value 8.0 has been empirically determined as
   *  a reasonable approximation such that f(...) < 8.0 indicates that
   *  spatial convolution will run faster while f(...) > 8.0 indicates that
   *  FFT convolution will run faster. The threshold value is not universal
   *  and may need to be adjusted to better match benchmarking results for
   *  particular hardware and expected image sizes so that nuances such as
   *  multithreading and GPU performance may be taken into account.
   *
   * \sa VkBlurringPerformanceMetric
   */
  itkSetMacro(MetricThreshold, float);
  itkGetMacro(MetricThreshold, float);

  /** Set the metric threshold from a certain parameter set describing the input size
   *  and kernel radius threshold that is expected to be equally fast with separable
   *  spatial smoothing and FFT smoothing */
  void
  SetMetricThreshold(const InputSizeType & inputSize, const KernelSizeType & kernelRadius)
  {
    this->SetMetricThreshold(ComputeMetricValue(inputSize, kernelRadius));
  }

  /** Compute the metric value for a given set of inputs */
  float
  ComputeMetricValue(const InputSizeType & inputSize, const KernelSizeType & kernelRadius) const;

  /** Estimate the kernel radius from ilevel settings */
  KernelSizeType
  GetKernelRadius(unsigned int ilevel) const;

  /** Get the kernel variance for the given pyramid level
   *  based on the current schedule */
  VarianceType
  GetVariance(unsigned int ilevel) const;

  /** Get whether FFT smoothing will be used for the given
   *  pyramid level */
  bool
  GetUseFFT(const KernelSizeType & kernelRadius) const;

protected:
  VkMultiResolutionPyramidImageFilter() = default;
  ~VkMultiResolutionPyramidImageFilter() override = default;

  /** Generate the output data. */
  void
  GenerateData() override;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  float                                 m_MetricThreshold = 8.0f;
  typename SpatialSmootherType::Pointer spatialSmoother = SpatialSmootherType::New();
  typename FFTSmootherType::Pointer     fftSmoother = FFTSmootherType::New();
};
} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkVkMultiResolutionPyramidImageFilter.hxx"
#endif

#endif
