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

#ifndef itkVkComplexToComplex1DFFTImageFilter_h
#define itkVkComplexToComplex1DFFTImageFilter_h

#include "itkVkCommon.h"
#include "itkComplexToComplex1DFFTImageFilter.h"

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
 * divisible by primes up to 13. 
 * 
 * Execution on input images with sizes divisible by primes greater than 17 may succeed 
 * with a fallback on Bluestein's algorithm per VkFFT with a cost to performance and output precision.
 *
 * \ingroup FourierTransform
 * \ingroup MultiThreaded
 * \ingroup ITKFFT
 * \ingroup VkFFTBackend
 */
template <typename TImage>
class VkComplexToComplex1DFFTImageFilter : public ComplexToComplex1DFFTImageFilter<TImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(VkComplexToComplex1DFFTImageFilter);

  using InputImageType = TImage;
  using OutputImageType = TImage;
  static_assert(std::is_same<typename TImage::PixelType, std::complex<float>>::value ||
                  std::is_same<typename TImage::PixelType, std::complex<double>>::value,
                "Unsupported pixel type");
  static_assert(TImage::ImageDimension >= 1 && TImage::ImageDimension <= 3, "Unsupported image dimension");

  /** Standard class type aliases. */
  using Self = VkComplexToComplex1DFFTImageFilter;
  using Superclass = ComplexToComplex1DFFTImageFilter<TImage>;
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
  itkTypeMacro(VkComplexToComplex1DFFTImageFilter, ComplexToComplex1DFFTImageFilter);

  static constexpr unsigned int ImageDimension = InputImageType::ImageDimension;

  itkGetMacro(DeviceID, uint64_t);
  itkSetMacro(DeviceID, uint64_t);

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
  uint64_t m_DeviceID{ 0UL };

  VkCommon m_VkCommon{};
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkVkComplexToComplex1DFFTImageFilter.hxx"
#endif

#endif // itkVkComplexToComplex1DFFTImageFilter_h
