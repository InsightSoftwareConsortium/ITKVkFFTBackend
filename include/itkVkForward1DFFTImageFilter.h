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
#ifndef itkVkForward1DFFTImageFilter_h
#define itkVkForward1DFFTImageFilter_h

#include "itkFFTImageFilterFactory.h"
#include "itkForward1DFFTImageFilter.h"
#include "itkVkCommon.h"
#include "itkVkGlobalConfiguration.h"

namespace itk
{
/**
 *\class VkForward1DFFTImageFilter
 *
 * \brief Vk-based forward 1D Fast Fourier Transform.
 *
 * This filter computes the forward Fourier transform in one dimension of an image. The
 * implementation is based on the VkFFT library.
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
 *
 * \sa VkGlobalConfiguration
 * \sa Forward1DFFTImageFilter
 */
template <typename TInputImage,
          typename TOutputImage = Image<std::complex<typename TInputImage::PixelType>, TInputImage::ImageDimension>>
class VkForward1DFFTImageFilter : public Forward1DFFTImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(VkForward1DFFTImageFilter);

  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  static_assert(std::is_same<typename TInputImage::PixelType, float>::value ||
                  std::is_same<typename TInputImage::PixelType, double>::value,
                "Unsupported pixel type");
  static_assert(std::is_same<typename TOutputImage::PixelType, std::complex<float>>::value ||
                  std::is_same<typename TOutputImage::PixelType, std::complex<double>>::value,
                "Unsupported pixel type");
  static_assert(TInputImage::ImageDimension >= 1 && TInputImage::ImageDimension <= 3, "Unsupported image dimension");

  /** Standard class type aliases. */
  using Self = VkForward1DFFTImageFilter;
  using Superclass = Forward1DFFTImageFilter<InputImageType, OutputImageType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  using InputPixelType = typename InputImageType::PixelType;
  using OutputPixelType = typename OutputImageType::PixelType;
  using ComplexType = OutputPixelType;
  using RealType = typename ComplexType::value_type;
  using SizeType = typename InputImageType::SizeType;
  using SizeValueType = typename InputImageType::SizeValueType;
  using OutputImageRegionType = typename OutputImageType::RegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(VkForward1DFFTImageFilter, Forward1DFFTImageFilter);

  static constexpr unsigned int ImageDimension = InputImageType::ImageDimension;

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
    return m_UseVkGlobalConfiguration ? VkGlobalConfiguration::GetDeviceID() : m_DeviceID;
  }

  SizeValueType
  GetSizeGreatestPrimeFactor() const override;

protected:
  VkForward1DFFTImageFilter() = default;
  ~VkForward1DFFTImageFilter() override = default;

  void
  GenerateData() override;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  bool     m_UseVkGlobalConfiguration{true};
  uint64_t m_DeviceID{ 0UL };

  VkCommon m_VkCommon{};
};

// Describe whether input/output are real- or complex-valued
// for factory registration
template <>
struct FFTImageFilterTraits<VkForward1DFFTImageFilter>
{
  template <typename TUnderlying>
  using InputPixelType = TUnderlying;
  template <typename TUnderlying>
  using OutputPixelType = std::complex<TUnderlying>;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkVkForward1DFFTImageFilter.hxx"
#endif

#endif // itkVkForward1DFFTImageFilter_h
