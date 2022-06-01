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
#include "itkVkGlobalConfiguration.h"

#include <mutex>
#include "itkSingleton.h"

namespace itk
{
struct VkGlobalConfigurationGlobals
{
  VkGlobalConfiguration::Pointer m_Instance{ nullptr };
  std::mutex                     m_CreationLock;
};

itkGetGlobalSimpleMacro(VkGlobalConfiguration, VkGlobalConfigurationGlobals, PimplGlobals);

VkGlobalConfigurationGlobals * VkGlobalConfiguration::m_PimplGlobals;

VkGlobalConfiguration::Pointer
VkGlobalConfiguration::GetInstance()
{
  itkInitGlobalsMacro(PimplGlobals);
  if (!m_PimplGlobals->m_Instance)
  {
    m_PimplGlobals->m_CreationLock.lock();
    // Need to make sure that during gaining access
    // to the lock that some other thread did not
    // initialize the singleton.
    if (!m_PimplGlobals->m_Instance)
    {
      m_PimplGlobals->m_Instance = Self::New();
      if (!m_PimplGlobals->m_Instance)
      {
        std::ostringstream message;
        message << "itk::ERROR: "
                << "VkGlobalConfiguration"
                << " Valid VkGlobalConfiguration instance not created";
        itk::ExceptionObject e_(__FILE__, __LINE__, message.str().c_str(), ITK_LOCATION);
        throw e_; /* Explicit naming to work around Intel compiler bug.  */
      }
    }
    m_PimplGlobals->m_CreationLock.unlock();
  }
  return typename VkGlobalConfiguration::Pointer{ m_PimplGlobals->m_Instance };
}

void
VkGlobalConfiguration::SetDeviceID(const uint64_t id)
{
  itkInitGlobalsMacro(PimplGlobals);
  GetInstance()->m_DeviceID = id;
}

uint64_t
VkGlobalConfiguration::GetDeviceID()
{
  itkInitGlobalsMacro(PimplGlobals);
  return uint64_t{ GetInstance()->m_DeviceID };
}

} // namespace itk
