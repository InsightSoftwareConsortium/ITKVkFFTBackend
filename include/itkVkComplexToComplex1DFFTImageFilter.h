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

#ifndef itkVkComplexToComplex1DFFTImageFilter_h
#define itkVkComplexToComplex1DFFTImageFilter_h

#include "itkComplexToComplex1DFFTImageFilter.h"
#include "itkFFTImageFilterFactory.h"
#include "itkVkCommon.h"
#include "itkVkGlobalConfiguration.h"

namespace itk
{

/**
 *\class VkComplexToComplex1DFFTImageFilter
 *
 *  \brief Implements an API to enable the 1D Fourier transform or the inverse
 *  Fourier transform of images with complex valued voxels to be computed using
 *  the VkFFT library.
 *
 * This filter is multithreaded and by default supports input images with sizes which are
 * divisible only by primes up to 13.
 *
 * Execution on input images with sizes divisible by primes greater than 13 may succeed
 * with a fallback on Bluestein's algorithm per VkFFT with a cost to performance and output precision.
 *
 * \ingroup FourierTransform
 * \ingroup MultiThreaded
 * \ingroup ITKFFT
 * \ingroup VkFFTBackend
 */
template <typename TInputImage, typename TOutputImage = TInputImage>
class VkComplexToComplex1DFFTImageFilter : public ComplexToComplex1DFFTImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(VkComplexToComplex1DFFTImageFilter);

  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  static_assert(std::is_same<typename TInputImage::PixelType, std::complex<float>>::value ||
                  std::is_same<typename TInputImage::PixelType, std::complex<double>>::value,
                "Unsupported pixel type");
  static_assert(std::is_same<typename TOutputImage::PixelType, std::complex<float>>::value ||
                  std::is_same<typename TOutputImage::PixelType, std::complex<double>>::value,
                "Unsupported pixel type");
  static_assert(TInputImage::ImageDimension >= 1 && TInputImage::ImageDimension <= 3, "Unsupported image dimension");

  /** Standard class type aliases. */
  using Self = VkComplexToComplex1DFFTImageFilter;
  using Superclass = ComplexToComplex1DFFTImageFilter<TInputImage, TOutputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;
  using InputPixelType = typename InputImageType::PixelType;
  using OutputPixelType = typename OutputImageType::PixelType;
  using ComplexType = InputPixelType;
  using RealType = typename ComplexType::value_type;
  using SizeType = typename InputImageType::SizeType;
  using SizeValueType = typename InputImageType::SizeValueType;
  using OutputImageRegionType = typename OutputImageType::RegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(VkComplexToComplex1DFFTImageFilter);

  static constexpr unsigned int ImageDimension{ InputImageType::ImageDimension };

  /** Determine whether local or global properties will be
   *  referenced for setting up GPU acceleration.
   *  Defaults to global so that the user can adjust default properties
   *  in filters constructed through the ITK object factory. */
  itkSetMacro(UseVkGlobalConfiguration, bool);
  itkGetMacro(UseVkGlobalConfiguration, bool);

  /** Local setting for enumerated GPU device to use for FFT.
   *  Ignored if `UseVkGlobalConfiguration` is true. */
  itkSetMacro(DeviceID, uint64_t);

  /** Return the enumerated GPU device to use for FFT
   *  according to current filter settings. */
  uint64_t
  GetDeviceID() const
  {
    return uint64_t{ m_UseVkGlobalConfiguration ? VkGlobalConfiguration::GetDeviceID() : m_DeviceID };
  }

  SizeValueType
  GetSizeGreatestPrimeFactor() const override;

protected:
  VkComplexToComplex1DFFTImageFilter() = default;
  ~VkComplexToComplex1DFFTImageFilter() override = default;

  void
  GenerateData() override;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  bool     m_UseVkGlobalConfiguration{ true };
  uint64_t m_DeviceID{ 0UL };
  VkCommon m_VkCommon{};
};

// Describe whether input/output are real- or complex-valued
// for factory registration
template <>
struct FFTImageFilterTraits<VkComplexToComplex1DFFTImageFilter>
{
  template <typename TUnderlying>
  using InputPixelType = std::complex<TUnderlying>;
  template <typename TUnderlying>
  using OutputPixelType = std::complex<TUnderlying>;
  using FilterDimensions = std::integer_sequence<unsigned int, 3, 2, 1>;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkVkComplexToComplex1DFFTImageFilter.hxx"
#endif

#endif // itkVkComplexToComplex1DFFTImageFilter_h
