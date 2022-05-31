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

#ifndef itkVkMultiResolutionPyramidImageFilterFactory_h
#define itkVkMultiResolutionPyramidImageFilterFactory_h
#include "VkFFTBackendExport.h"

#include "itkVkMultiResolutionPyramidImageFilter.h"
#include "itkImage.h"
#include "itkObjectFactoryBase.h"
#include "itkVersion.h"

namespace itk
{
/** \class VkMultiResolutionPyramidImageFilterFactory
 *
 * \brief Object Factory implementation for overriding
 *  MultiResolutionPyramidImageFilterFactory with VkMultiResolutionPyramidImageFilterFactory
 *
 * \sa ObjectFactoryBase
 * \sa MultiResolutionPyramidImageFilter
 * \sa VkMultiResolutionPyramidImageFilter
 *
 * \ingroup VkFFTBackend
 * \ingroup ITKRegistration
 * \ingroup FourierTransform
 * \ingroup ITKFFT
 */
class VkMultiResolutionPyramidImageFilterFactory : public itk::ObjectFactoryBase
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(VkMultiResolutionPyramidImageFilterFactory);

  using Self = VkMultiResolutionPyramidImageFilterFactory;
  using Superclass = ObjectFactoryBase;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Class methods used to interface with the registered factories. */
  const char *
  GetITKSourceVersion() const override
  {
    return ITK_SOURCE_VERSION;
  }
  const char *
  GetDescription() const override
  {
    return "An VkMultiResolutionPyramidImageFilterFactory factory";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(VkMultiResolutionPyramidImageFilterFactory, itk::ObjectFactoryBase);

  /** Register one factory of this type  */
  static void
  RegisterOneFactory()
  {
    VkMultiResolutionPyramidImageFilterFactory::Pointer factory = VkMultiResolutionPyramidImageFilterFactory::New();

    ObjectFactoryBase::RegisterFactoryInternal(factory);
  }

protected:
  /** Override base MultiResolutionPyramidImageFilter constructor at runtime to return
   *  an upcast VkMultiResolutionPyramidImageFilter instance through the object factory
   */
  template <typename InputPixelType, typename OutputPixelType, unsigned int D, unsigned int... ImageDimensions>
  void
  OverrideSuperclassType(const std::integer_sequence<unsigned int, D, ImageDimensions...> &)
  {
    using InputImageType = Image<InputPixelType, D>;
    using OutputImageType = Image<OutputPixelType, D>;
    this->RegisterOverride(
      typeid(typename VkMultiResolutionPyramidImageFilter<InputImageType, OutputImageType>::Superclass).name(),
      typeid(VkMultiResolutionPyramidImageFilter<InputImageType, OutputImageType>).name(),
      "VkMultiResolutionPyramidImageFilter Override",
      true,
      CreateObjectFunction<VkMultiResolutionPyramidImageFilter<InputImageType, OutputImageType>>::New());
    OverrideSuperclassType<InputPixelType, OutputPixelType>(std::integer_sequence<unsigned int, ImageDimensions...>{});
  }
  template <typename InputPixelType, typename OutputPixelType>
  void
  OverrideSuperclassType(const std::integer_sequence<unsigned int> &)
  {}

  VkMultiResolutionPyramidImageFilterFactory()
  {
    OverrideSuperclassType<float, float>(std::integer_sequence<unsigned int, 4, 3, 2, 1>{});

    OverrideSuperclassType<double, double>(std::integer_sequence<unsigned int, 4, 3, 2, 1>{});
  }
};

} // namespace itk

#endif // itkVkMultiResolutionPyramidImageFilterFactory_h
