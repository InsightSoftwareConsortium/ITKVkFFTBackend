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
#ifndef itkVkHalfHermitianToRealInverseFFTImageFilter_h
#define itkVkHalfHermitianToRealInverseFFTImageFilter_h

#include "itkFFTImageFilterFactory.h"
#include "itkHalfHermitianToRealInverseFFTImageFilter.h"
#include "itkImage.h"
#include "itkVkCommon.h"
#include "itkVkGlobalConfiguration.h"

namespace itk
{
/**
 *\class VkHalfHermitianToRealInverseFFTImageFilter
 *
 * \brief Vk-based inverse Fast Fourier Transform.
 *
 * This filter computes the inverse Fourier transform of an image. The
 * implementation is based on the VkFFT library.
 *
 * This filter is multithreaded and supports input images with sizes which are
 * divisible only by primes up to 13.
 *
 * \ingroup FourierTransform
 * \ingroup MultiThreaded
 * \ingroup ITKFFT
 * \ingroup VkFFTBackend
 *
 * \sa VkGlobalConfiguration
 * \sa HalfHermitianToRealInverseFFTImageFilter
 */
template <typename TInputImage,
          typename TOutputImage = Image<typename TInputImage::PixelType::value_type, TInputImage::ImageDimension>>
class VkHalfHermitianToRealInverseFFTImageFilter
  : public HalfHermitianToRealInverseFFTImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(VkHalfHermitianToRealInverseFFTImageFilter);

  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  static_assert(std::is_same<typename TInputImage::PixelType, std::complex<float>>::value ||
                  std::is_same<typename TInputImage::PixelType, std::complex<double>>::value,
                "Unsupported pixel type");
  static_assert(std::is_same<typename TOutputImage::PixelType, float>::value ||
                  std::is_same<typename TOutputImage::PixelType, double>::value,
                "Unsupported pixel type");
  static_assert(TInputImage::ImageDimension >= 1 && TInputImage::ImageDimension <= 3, "Unsupported image dimension");

  /** Standard class type aliases. */
  using Self = VkHalfHermitianToRealInverseFFTImageFilter;
  using Superclass = HalfHermitianToRealInverseFFTImageFilter<InputImageType, OutputImageType>;
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
  itkTypeMacro(VkHalfHermitianToRealInverseFFTImageFilter, HalfHermitianToRealInverseFFTImageFilter);

  static constexpr unsigned int ImageDimension{ InputImageType::ImageDimension };

  /** Determine whether local or global properties will be
   *  referenced for setting up GPU acceleration.
   *  Defaults to global so that the user can adjust default properties
   *  in filters constructed through the ITK object factory. */
  itkSetMacro(UseVkGlobalConfiguration, bool);
  itkGetMacro(UseVkGlobalConfiguration, bool);

  /** Local platform identifier for accelerated backend.
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
  VkHalfHermitianToRealInverseFFTImageFilter();
  ~VkHalfHermitianToRealInverseFFTImageFilter() override = default;

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
struct FFTImageFilterTraits<VkHalfHermitianToRealInverseFFTImageFilter>
{
  template <typename TUnderlying>
  using InputPixelType = std::complex<TUnderlying>;
  template <typename TUnderlying>
  using OutputPixelType = TUnderlying;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkVkHalfHermitianToRealInverseFFTImageFilter.hxx"
#endif

#endif // itkVkHalfHermitianToRealInverseFFTImageFilter_h
