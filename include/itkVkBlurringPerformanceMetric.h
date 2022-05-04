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
#ifndef itkVkBlurringPerformanceMetric_h
#define itkVkBlurringPerformanceMetric_h
#include "VkFFTBackendExport.h"

#include "itkLightObject.h"
#include "itkMath.h"

namespace itk
{
/**
 *\class VkBlurringPerformanceMetric
 * \brief Quantify anticipated performance difference between
 *     Gaussian blurring implementations
 *
 * Separable spatial convolution with DiscreteGaussianImageFilter
 * runs quickly for small kernel sizes but scales poorly with
 * increasing kernel size. By contrast ITK FFT convolution accelerated
 * with a VkFFT GPU backend scales slowly with increasing kernel size
 * but is typically outperformed by spatial convolution filters
 * for small kernel sizes.
 *
 * VkBlurringPerformanceMetric quantifies the anticipated performance
 * tradeoff between separable spatial blurring and FFT blurring for
 * a given image region and kernel size. The metric can be paired
 * with a fixed threshold value to dynamically select the smoothing
 * filter expected to give the best performance. The exact threshold
 * at which separable spatial blurring and FFT blurring are equivalent
 * will depend on user hardware and can be estimated through benchmarking
 * with scripts in the ITKVkFFTBackend repository.
 *
 * Separable spatial smoothing runtime scales quickly with increasing kernel size
 * while FFT convolution runtime scales much more slowly with kernel size.
 * The difference in smoothing performance correlates to the metric
 *
 * f(R,K) = log(R0 * R1 * ... * RN * (K0 + K1 + ... + KN))
 *
 * where R = smoothing region size and K = kernel size in N dimensions
 *
 * \sa itkVkDiscreteGaussianImageFilter
 * \sa itkVkMultiResolutionPyramidImageFilter
 *
 * \ingroup ITKFFT
 * \ingroup ITKNumerics
 * \ingroup VkFFTBackend
 */
template <typename TImage, typename TKernel = TImage>
class VkBlurringPerformanceMetric : public LightObject
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(VkBlurringPerformanceMetric);

  /** Standard class type aliases. */
  using Self = VkBlurringPerformanceMetric;
  using Superclass = LightObject;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for class instantiation. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(VkBlurringPerformanceMetric, LightObject);

  using ImageType = TImage;
  using KernelType = TKernel;
  using RegionSizeType = typename ImageType::SizeType;
  using KernelSizeType = typename KernelType::SizeType;

  static constexpr unsigned int ImageDimension = ImageType::ImageDimension;

  /** Compute metric to quantify anticipated performance
   *  tradeoff between VkFFT blurring and separable spatial convolution.
   *
   *  f(R,K) = log(R0 * R1 * ... * RN * (K0 + K1 + ... + KN))
   *
   */
  static float
  Compute(const RegionSizeType & regionSize, const KernelSizeType & kernelSize)
  {
    unsigned int totalKernelSize = 0;
    float        metricValue = 1.0f;
    for (unsigned int dim = 0; dim < ImageDimension; ++dim)
    {
      totalKernelSize += kernelSize[dim];
      metricValue *= regionSize[dim];
    }
    metricValue *= totalKernelSize;
    metricValue = std::log10(metricValue);
    return metricValue;
  }

protected:
  VkBlurringPerformanceMetric() = default;
  ~VkBlurringPerformanceMetric() override = default;
};
} // end namespace itk

#endif // itkVkBlurringPerformanceMetric_h
