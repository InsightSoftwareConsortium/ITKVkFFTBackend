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
#include "ITKFFTExport.h"

#include "itkVkComplexToComplex1DFFTImageFilter.h"
#include "itkVkComplexToComplexFFTImageFilter.h"
#include "itkFFTImageFilterFactory.h"
#include "itkVkForward1DFFTImageFilter.h"
#include "itkVkForwardFFTImageFilter.h"
#include "itkVkHalfHermitianToRealInverseFFTImageFilter.h"
#include "itkVkInverse1DFFTImageFilter.h"
#include "itkVkInverseFFTImageFilter.h"
#include "itkVkRealToHalfHermitianForwardFFTImageFilter.h"

#include "itkVkFFTImageFilterInitFactory.h"

namespace itk
{
VkFFTImageFilterInitFactory::VkFFTImageFilterInitFactory()
{
  VkFFTImageFilterInitFactory::RegisterFactories();
}

void VkFFTImageFilterInitFactory::RegisterFactories()
{
  itk::ObjectFactoryBase::RegisterFactory(FFTImageFilterFactory<VkComplexToComplex1DFFTImageFilter>::New(),
                                          itk::ObjectFactoryEnums::InsertionPosition::INSERT_AT_FRONT);
  itk::ObjectFactoryBase::RegisterFactory(FFTImageFilterFactory<VkComplexToComplexFFTImageFilter>::New(),
                                          itk::ObjectFactoryEnums::InsertionPosition::INSERT_AT_FRONT);
  itk::ObjectFactoryBase::RegisterFactory(FFTImageFilterFactory<VkForward1DFFTImageFilter>::New(),
                                          itk::ObjectFactoryEnums::InsertionPosition::INSERT_AT_FRONT);
  itk::ObjectFactoryBase::RegisterFactory(FFTImageFilterFactory<VkForwardFFTImageFilter>::New(),
                                          itk::ObjectFactoryEnums::InsertionPosition::INSERT_AT_FRONT);
  itk::ObjectFactoryBase::RegisterFactory(FFTImageFilterFactory<VkHalfHermitianToRealInverseFFTImageFilter>::New(),
                                          itk::ObjectFactoryEnums::InsertionPosition::INSERT_AT_FRONT);
  itk::ObjectFactoryBase::RegisterFactory(FFTImageFilterFactory<VkInverse1DFFTImageFilter>::New(),
                                          itk::ObjectFactoryEnums::InsertionPosition::INSERT_AT_FRONT);
  itk::ObjectFactoryBase::RegisterFactory(FFTImageFilterFactory<VkInverseFFTImageFilter>::New(),
                                          itk::ObjectFactoryEnums::InsertionPosition::INSERT_AT_FRONT);
  itk::ObjectFactoryBase::RegisterFactory(FFTImageFilterFactory<VkRealToHalfHermitianForwardFFTImageFilter>::New(),
                                          itk::ObjectFactoryEnums::InsertionPosition::INSERT_AT_FRONT);
}

// Undocumented API used to register during static initialization.
// DO NOT CALL DIRECTLY.
void VkFFTBackend_EXPORT
VkFFTImageFilterInitFactoryRegister__Private()
{
  VkFFTImageFilterInitFactory::RegisterFactories();
}

} // end namespace itk
