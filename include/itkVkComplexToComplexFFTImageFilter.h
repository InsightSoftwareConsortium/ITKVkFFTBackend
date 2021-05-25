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

#ifndef itkVkComplexToComplexFFTImageFilter_h
#define itkVkComplexToComplexFFTImageFilter_h

#include "itkComplexToComplexFFTImageFilter.h"

namespace itk
{

/**
 *\class VkComplexToComplexFFTImageFilter
 *
 *  \brief Implements an API to enable the Fourier transform or the inverse
 *  Fourier transform of images with complex valued voxels to be computed using
 *  the VkFFT library.
 *
 * This filter is multithreaded and supports input images with sizes which are
 * divisible by primes up to 13.
 *
 * \ingroup FourierTransform
 * \ingroup MultiThreaded
 * \ingroup ITKFFT
 * \ingroup VkFFTBackend
 */
template <typename TImage>
class VkComplexToComplexFFTImageFilter : public ComplexToComplexFFTImageFilter<TImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(VkComplexToComplexFFTImageFilter);

  /** Standard class type aliases. */
  using Self = VkComplexToComplexFFTImageFilter;
  using Superclass = ComplexToComplexFFTImageFilter<TImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  using ImageType = TImage;
  using PixelType = typename ImageType::PixelType;
  using ComplexType = PixelType;
  using ValueType = typename ComplexType::value_type;
  using ComplexAsArrayType = ValueType[2];
  using InputImageType = typename Superclass::InputImageType;
  using OutputImageType = typename Superclass::OutputImageType;
  using OutputImageRegionType = typename OutputImageType::RegionType;
  using ImageSizeType = typename ImageType::SizeType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(VkComplexToComplexFFTImageFilter, ComplexToComplexFFTImageFilter);

  static constexpr unsigned int ImageDimension = ImageType::ImageDimension;

  itkGetMacro(DeviceID, uint64_t);
  itkSetMacro(DeviceID, uint64_t);

protected:
  VkComplexToComplexFFTImageFilter();
  ~VkComplexToComplexFFTImageFilter() override = default;

  void
  UpdateOutputData(DataObject * output) override;

  void
  BeforeThreadedGenerateData() override;

  void
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  uint64_t m_DeviceID{ 0UL };
};
} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkVkComplexToComplexFFTImageFilter.hxx"
#endif

#endif // itkVkComplexToComplexFFTImageFilter_h
