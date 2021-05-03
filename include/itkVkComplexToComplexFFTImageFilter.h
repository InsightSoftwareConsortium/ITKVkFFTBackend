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

#include "itkImageToImageFilter.h"
#include "itkComplexToComplexFFTImageFilter.h"


namespace itk
{

/** \class VkComplexToComplexFFTImageFilter
 *
 *  \brief Implements an API to enable the Fourier transform or the inverse
 *  Fourier transform of images with complex valued voxels to be computed using
 *  Vulkan FFT from https://github.com/DTolm/VkFFT.
 *
 * This filter is multithreaded and supports input images with sizes that are not
 * a power of two.
 *
 * \ingroup FourierTransform
 * \ingroup MultiThreaded
 * \ingroup ITKFFT
 *
 */
template <typename TImage>
class ITK_TEMPLATE_EXPORT VkComplexToComplexFFTImageFilter : public ComplexToComplexFFTImageFilter<TImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(VkComplexToComplexFFTImageFilter);

  static constexpr unsigned int ImageDimension = TImage::ImageDimension;

  /** Standard class type aliases. */
  using Self = VkComplexToComplexFFTImageFilter<TImage>;
  using Superclass = ComplexToComplexFFTImageFilter<TImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  using ImageType = TImage;
  using PixelType = typename ImageType::PixelType;
  using InputImageType = typename Superclass::InputImageType;
  using InputRegionType = typename InputImageType::RegionType;
  using OutputImageType = typename Superclass::OutputImageType;
  using OutputRegionType = typename OutputImageType::RegionType;

  /** Run-time type information. */
  itkTypeMacro(VkComplexToComplexFFTImageFilter, ComplexToComplexFFTImageFilter);

  /** Standard New macro. */
  itkNewMacro(Self);

protected:
  VkComplexToComplexFFTImageFilter();
  ~VkComplexToComplexFFTImageFilter() override = default;

  void PrintSelf(std::ostream & os, Indent indent) const override;

  void DynamicThreadedGenerateData(const OutputRegionType & outputRegion) override;

private:
#ifdef ITK_USE_CONCEPT_CHECKING
  // Add concept checking such as
  // itkConceptMacro( FloatingPointPixel, ( itk::Concept::IsFloatingPoint< typename ImageType::PixelType > ) );
#endif
};
} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkVkComplexToComplexFFTImageFilter.hxx"
#endif

#endif // itkVkComplexToComplexFFTImageFilter
