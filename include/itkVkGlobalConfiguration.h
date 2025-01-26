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
#ifndef itkVkGlobalConfiguration_h
#define itkVkGlobalConfiguration_h

#include "VkFFTBackendExport.h"
#include "itkLightObject.h"
#include "itkMacro.h"

namespace itk
{

/**
 * \class VkGlobalConfigurationGlobals
 *  \brief Implementation detail to reference the VkGlobalConfiguration singleton
 *  \ingroup VkFFTBackend
 */
struct VkGlobalConfigurationGlobals;

/**
 *\class VkGlobalConfiguration
 *
 *  \brief Implements a singleton instance for setting global VkFFT default parameters.
 *
 * \ingroup VkFFTBackend
 */
class VkFFTBackend_EXPORT VkGlobalConfiguration : public LightObject
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(VkGlobalConfiguration);

  /** Standard class type aliases. */
  using Self = VkGlobalConfiguration;
  using Superclass = LightObject;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(VkGlobalConfiguration);

  /** Default accelerated platform identifier */
  static void
  SetDeviceID(const uint64_t id);

  /** Default accelerated platform identifier */
  static uint64_t
  GetDeviceID();

private:
  VkGlobalConfiguration() = default;
  ~VkGlobalConfiguration() override = default;

  /** Access synchronized global singleton */
  static Pointer
  GetInstance();

  itkGetGlobalDeclarationMacro(VkGlobalConfigurationGlobals, PimplGlobals);

  /** This is a singleton pattern New.  There will only be ONE
   * reference to a VkGlobalConfiguration object per process.
   * The single instance will be unreferenced when
   * the program exits. */
  itkFactorylessNewMacro(Self);

  static VkGlobalConfigurationGlobals * m_PimplGlobals;

  uint64_t m_DeviceID{ 0 };
};
} // namespace itk

#endif // itkVkGlobalConfiguration_h
