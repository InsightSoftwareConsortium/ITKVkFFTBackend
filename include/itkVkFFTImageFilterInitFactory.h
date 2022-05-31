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
#ifndef itkVkFFTImageFilterInitFactory_h
#define itkVkFFTImageFilterInitFactory_h
#include "VkFFTBackendExport.h"

#include "itkLightObject.h"

namespace itk
{
/**
 *\class VkFFTImageFilterInitFactory
 * \brief Initialize Vk FFT image filter factory backends.
 *
 * The purpose of VkFFTImageFilterInitFactory is to perform
 * one-time registration of factory objects that handle
 * creation of Vk-backend FFT image filter classes
 * through the ITK object factory singleton mechanism.
 *
 * \ingroup FourierTransform
 * \ingroup ITKFFT
 * \ingroup ITKSystemObjects
 * \ingroup VkFFTBackend
 */
class VkFFTBackend_EXPORT VkFFTImageFilterInitFactory : public LightObject
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(VkFFTImageFilterInitFactory);

  /** Standard class type aliases. */
  using Self = VkFFTImageFilterInitFactory;
  using Superclass = LightObject;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(VkFFTImageFilterInitFactory, LightObject);

  /** Mimic factory interface for Python initialization  */
  static void
  RegisterOneFactory()
  {
    RegisterFactories();
  }

  /** Register all Vk FFT factories */
  static void
  RegisterFactories();

protected:
  VkFFTImageFilterInitFactory();
  ~VkFFTImageFilterInitFactory() override = default;
};
} // end namespace itk

#endif
